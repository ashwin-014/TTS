import torch
import onnx
import tensorrt as trt
from cuda import cuda, cudart, nvrtc       # Using Cuda Python
import os
import onnxruntime as ort

print("Torch Version: ", torch.__version__)
print("Onnx Version: ", onnx.__version__)
print("TensorRT Version: ", trt.__version__)
print("OnnxRuntime Version: ", ort.__version__)

# Initialize CUDA Driver API
err, = cuda.cuInit(0)

# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)

# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:   
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def load_model(trt_engine_file):
    trt_logger = trt.Logger()
    trt_logger.min_severity = trt.Logger.VERBOSE
   
    with open(trt_engine_file, "rb") as f:
        trt_runtime = trt.Runtime(trt_logger)
        trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
        trt_context = trt_engine.create_execution_context()

    return {
        "runtime": trt_runtime,
        "engine": trt_engine,
        "context": trt_context
    }

def allocate_binding_buffer(types_dict, shapes_dict):
    '''
    Allocate binding buffers for trt based on provided types and shapes dict
    '''
    return {
        k: torch.empty(
            size=shape,
            dtype=types_dict[k],
            device='cuda',
        )
        for k, shape in shapes_dict.items()
    }

def _allocate_memory(model):
    # print(model)
    """Helper function for binding several inputs at once and pre-allocating the results."""
    # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
    inputs = allocate_binding_buffer(model["dim_specs"]["input_types"], model["dim_specs"]["input_shapes"])
    outputs = allocate_binding_buffer(model["dim_specs"]["output_types"], model["dim_specs"]["output_shapes"])

    bindings = [None] * model["engine"].num_bindings
    device_memory_addresses = {}
    for binding in model["engine"]:
        binding_idx = model["engine"].get_binding_index(binding)
        dtype = trt.nptype(model["engine"].get_binding_dtype(binding))

        if model["engine"].binding_is_input(binding):
            bindings[binding_idx] = int(inputs[binding].data_ptr())
            device_memory_addresses[binding] = int(inputs[binding].data_ptr())
        else:
            bindings[binding_idx] = int(outputs[binding].data_ptr())
            device_memory_addresses[binding] = int(outputs[binding].data_ptr())

    return {
        "bindings": bindings,
        "inputs": inputs,
        "outputs": outputs,
        "device_mem_address": device_memory_addresses
    }

def load_trt_models(trt_path, model_dim_specs):
    model = load_model(trt_path)
    model["data_stream"] = {}
    for binding in model["engine"]:
        err, model["data_stream"][binding] = cuda.cuStreamCreate(1)
    err, model["model_stream"] = cuda.cuStreamCreate(1)
    model["dim_specs"] = model_dim_specs
    model["io"] = _allocate_memory(model)
    return model

def release_trt_resources(model):
    for binding in model["engine"]:
        err, = cuda.cuStreamDestroy(model["data_stream"][binding])

    for node in model["io"]["device_mem_address"].keys():
        err, = cuda.cuMemFree(model["io"]["device_mem_address"][node])    

    err, = cuda.cuStreamDestroy(model["model_stream"])    
    return True

def export_onnx(emb):
    torch.onnx.export(
        model=emb,
        args=(sample),
        f="/tmp/emb.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['x'],
        output_names=['x_emb'],
        dynamic_axes={
            'x': {0: 'bs', 1: 'sequence_length'},
        },
        verbose=False,
    )

    onnx_model = onnx.load("/tmp/emb.onnx")
    onnx.checker.check_model(onnx_model)
    return True

def build_engine(onnx_model_path, engine_model_path, dim_specs):

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # config.builder_optimization_level = 5

    profile = builder.create_optimization_profile()

    for input_name, dims in dim_specs.items():
        profile.set_shape(input_name, dims["min"], dims["opt"], dims["max"])
    config.add_optimization_profile(profile)

    parser = trt.OnnxParser(network, trt_logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
            	print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(engine_model_path, "wb") as f:
        f.write(engineString)

    return True

if __name__ == "__main__":
    sample = torch.randint(0, 100, (24,), dtype=torch.int32)
    sample = sample.repeat(2, 1).to("cuda")

    bs_to_use = sample.shape[0]

    # Checking Sample
    print("Sample Input: \n", sample)
    print("Sample Input Shape: ", sample.shape)
    
    emb = torch.nn.Embedding(101, 512).to('cuda')

    with torch.no_grad():
        emb_sample = emb(sample)

    # Checking torch output
    print("\n\nTorch Output: \n", emb_sample)
    print("Torch Output Shape: ", emb_sample.shape)

    # Exporting ONNX File
    export_onnx(emb)

    # Running ORT
    emb_sess = ort.InferenceSession("/tmp/emb.onnx", providers=["CPUExecutionProvider"])
    ort_out = emb_sess.run(["x_emb"], {"x": sample.cpu().numpy()})

    # Checking ORT output
    print("\n\nOnnxRuntime Output: \n", ort_out[0])
    print("OnnxRuntime Output Shape: ", ort_out[0].shape)

    # Building TensorRT Engine
    dim_specs = {
        "x": {
            "min": (1, 1),
            "opt": (2, 24),
            "max": (6, 400),    
        },
    }
    
    build_engine( "/tmp/emb.onnx", "/tmp/emb.engine", dim_specs )

    # Running TensorRT Engine
    max_bs = 2
    max_sequence_length = 24
    encoder_hidden_size = 512
    max_decoder_sequence_length = 817

    embedding_dim_specs = {
        "input_shapes": {
            "x": (max_bs, max_sequence_length),
        },
        "input_types": {
            "x": torch.int32,
        },
        "output_shapes": {
            "x_emb": (max_bs, max_sequence_length, encoder_hidden_size),
        },
        "output_types": {
            "x_emb": torch.float32,
        }
    }

    emb_engine = load_trt_models("/tmp/emb.engine", embedding_dim_specs)

    emb_engine["io"]["inputs"]["x"][:bs_to_use, :sample.shape[1]] = sample[:bs_to_use, :]

    print(emb_engine["io"]["inputs"]["x"])

    emb_engine["context"].set_binding_shape(emb_engine["engine"].get_binding_index("x"), 
                                            emb_engine["io"]["inputs"]["x"][:bs_to_use, :sample.shape[1]].shape)

    assert emb_engine["context"].all_binding_shapes_specified

    print(emb_engine["context"].get_binding_shape(emb_engine["engine"].get_binding_index("x")))

    # emb_engine["context"].set_input_shape("x", (2, 24))

    emb_engine["context"].execute_v2(bindings=emb_engine["io"]["bindings"])

    # Checking TensorRT Output
    print("\n\nTensorRT Output: \n", emb_engine["io"]["outputs"]["x_emb"][:bs_to_use, :sample.shape[1], :])
    print("TensorRT Output Shape: ", emb_engine["io"]["outputs"]["x_emb"][:bs_to_use, :sample.shape[1], :].shape)

    release_trt_resources(emb_engine)

    