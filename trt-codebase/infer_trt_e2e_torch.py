import os
import time
import pysbd
import numpy as np
import torch
import scipy
import tensorrt as trt
from cuda import cuda, cudart, nvrtc
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
# import torch
from itertools import zip_longest

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
    
    """Helper function for binding several inputs at once and pre-allocating the results."""
    # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
    inputs = allocate_binding_buffer(model["input_types"], model["input_shapes"])
    outputs = allocate_binding_buffer(model["output_types"], model["output_shapes"])

    bindings = [None] * model["engine"].num_bindings
    device_memory_addresses = {}
    # outputs_device = {}
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

def batch():
    tts = TTS(
        model_path="./models/v1/hi/hififastpitch/best_model.pth",
        config_path="./models/v1/hi/hififastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    start = time.time()

    text="नमस्ते आपका नाम क्या है. मुझे ये करना हे भारत हैं नाम."

    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(" > Text splitted to sentences.")
    print(sens)
    tts.tts_to_file(text=sens, speaker=tts.speakers[0], file_path="output.wav")
    print(f"done in {time.time() - start}")


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


def trt_exec(
    inputs: np.array,
    speaker_id: np.array,
    hififastpitch,
) -> list:
    bs_to_use = inputs.shape[0]
    """
    Executes TRT code
    """
    print(inputs.shape)
    print(inputs)
    print(speaker_id)
    start = time.time()

    hififastpitch["io"]["inputs"]["fastpitch/x"][:bs_to_use, :] = np.ascontiguousarray(inputs)[:bs_to_use, :]
    print(hififastpitch["io"]["inputs"]["fastpitch/x"])

    err, = cuda.cuMemcpyHtoDAsync(
        hififastpitch["io"]["device_mem_address"]["fastpitch/x"], 
        hififastpitch["io"]["inputs"]["fastpitch/x"].ctypes.data, 
        hififastpitch["io"]["inputs"]["fastpitch/x"].nbytes,
        hififastpitch["data_stream"]["fastpitch/x"],
    )

    hififastpitch["context"].set_binding_shape(hififastpitch["engine"].get_binding_index("fastpitch/x"), 
                                                hififastpitch["io"]["inputs"]["fastpitch/x"][:bs_to_use, :].shape)
    print(hififastpitch["engine"].get_tensor_shape("fastpitch/x"))
    
    hififastpitch["io"]["inputs"]["fastpitch/speaker_id"] = speaker_id
    err, = cuda.cuMemcpyHtoDAsync(
        hififastpitch["io"]["device_mem_address"]["fastpitch/speaker_id"], 
        hififastpitch["io"]["inputs"]["fastpitch/speaker_id"].ctypes.data, 
        hififastpitch["io"]["inputs"]["fastpitch/speaker_id"].nbytes,
        hififastpitch["data_stream"]["fastpitch/speaker_id"],
    )

    cuda.cuStreamSynchronize(hififastpitch["data_stream"]["fastpitch/x"])
    cuda.cuStreamSynchronize(hififastpitch["data_stream"]["fastpitch/speaker_id"])

    assert hififastpitch["context"].all_binding_shapes_specified
    # hififastpitch["context"].execute_async_v2(bindings=hififastpitch["io"]["bindings"],
    #                                           stream_handle=hififastpitch["model_stream"].getPtr())

    # cuda.cuStreamSynchronize(hififastpitch["model_stream"])

    hififastpitch["context"].execute_v2(bindings=hififastpitch["io"]["bindings"])


    print(hififastpitch["engine"].get_binding_shape("fastpitch/decoder_output"))
    print(hififastpitch["engine"].get_binding_shape("fastpitch/alignments"))

    err, = cuda.cuMemcpyDtoHAsync(
        hififastpitch["io"]["outputs"]["fastpitch/decoder_output"].ctypes.data,
        hififastpitch["io"]["device_mem_address"]["fastpitch/decoder_output"],
        trt.volume((bs_to_use, 80, 817)) * np.dtype(
           trt.nptype(hififastpitch["engine"].get_binding_dtype("fastpitch/decoder_output"))).itemsize,
        hififastpitch["data_stream"]["fastpitch/decoder_output"]
    )

    cuda.cuStreamSynchronize(hififastpitch["data_stream"]["fastpitch/speaker_id"])

    print(hififastpitch["io"]["outputs"]["fastpitch/decoder_output"][:bs_to_use])

    err, = cuda.cuMemcpyDtoHAsync(
        hififastpitch["io"]["outputs"]["fastpitch/alignments"].ctypes.data,
        hififastpitch["io"]["device_mem_address"]["fastpitch/alignments"],
        trt.volume((bs_to_use, 817, 24)) * np.dtype(
            trt.nptype(hififastpitch["engine"].get_binding_dtype("fastpitch/alignments"))).itemsize,
        hififastpitch["data_stream"]["fastpitch/alignments"]
    )

    err, = cuda.cuMemcpyDtoHAsync(
        hififastpitch["io"]["outputs"]["hifigan/o"].ctypes.data,
        hififastpitch["io"]["device_mem_address"]["hifigan/o"],
        trt.volume((bs_to_use, 1, 209152)) * np.dtype(
            trt.nptype(hififastpitch["engine"].get_binding_dtype("hifigan/o"))).itemsize,
        hififastpitch["data_stream"]["hifigan/o"]
    )

    print(hififastpitch["engine"].get_binding_shape("hifigan/o"))

    cuda.cuStreamSynchronize(hififastpitch["data_stream"]["fastpitch/alignments"])
    attn = hififastpitch["io"]["outputs"]["fastpitch/alignments"].sum(axis=(-1, -2))
    attn = (attn - 5) * 256
    attn = attn.astype(np.int32)

    wavs = []
    cuda.cuStreamSynchronize(hififastpitch["data_stream"]["hifigan/o"])
    for i, wave in enumerate(hififastpitch["io"]["outputs"]["hifigan/o"][:bs_to_use]):
        wave = wave.squeeze()[: attn[i]]
        wave = wave.squeeze()
        wavs += list(wave)
        wavs += [0] * 10000

    process_time = time.time() - start
    audio_time = len(wavs) / 22050
    print(f" > Processing time: {process_time}")
    print(f" > Real-time factor: {process_time / audio_time}")
    return wavs

def trt_batch():
    """
    Execution code for TensorRT models
    """
    # Model init
    # --------------------------------------
    hififastpitch = load_model("models/v1/hi/nosqueeze_transpose/hififastpitch.engine")
    hififastpitch["data_stream"] = {}
    for binding in hififastpitch["engine"]:
        err, hififastpitch["data_stream"][binding] = cuda.cuStreamCreate(1)
    err, hififastpitch["model_stream"] = cuda.cuStreamCreate(1)
    print("Initialised hififastpitch...")

    max_bs = 6
    max_encoder_sequence_length = 24
    max_decoder_sequence_length = 817

    # setting shape and types for FastPitch
    hififastpitch["input_shapes"] = {
        "fastpitch/x": (max_bs, max_encoder_sequence_length),
        "fastpitch/speaker_id": (1),
    }
    hififastpitch["input_types"] = {
        "fastpitch/x": np.int32,
        "fastpitch/speaker_id": np.int32,
    }
    hififastpitch["output_shapes"] = {
        "fastpitch/decoder_output": (max_bs, 80, max_decoder_sequence_length),  
        "fastpitch/alignments": (max_bs, max_decoder_sequence_length, max_encoder_sequence_length),
        "fastpitch/pitch": (max_bs, 1, max_encoder_sequence_length),
        "fastpitch/durations_log": (max_bs, 1, max_encoder_sequence_length),
        "hifigan/o": (max_bs, 1, 209152),
    }
    hififastpitch["output_types"] = {
        "fastpitch/decoder_output": np.float32,
        "fastpitch/alignments": np.float32,
        "fastpitch/pitch": np.float32,
        "fastpitch/durations_log": np.float32,
        "hifigan/o": np.float32,
    }
    hififastpitch["io"] = _allocate_memory(hififastpitch)

    # --------------------------------------
    # Text init
    # --------------------------------------
    # text="मेरा. नमस्ते. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नाम भारत हैं."
    # text="नमस्ते आपका नाम क्या है."
    # text = "मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है"

    # text = "मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है"
    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(" > Text splitted to sentences.")
    print(sens)
    
    # --------------------------------------
    # TRT Inputs init
    # --------------------------------------
    from TTS.config import load_config
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    config = load_config("./models/v1/hi/fastpitch/config.json")
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    tok_ids = [tokenizer.text_to_ids(s, language=None) for s in sens]
    inputs = np.array(list(zip_longest(*tok_ids, fillvalue=0)), dtype=np.int32).T
    speaker_id = np.asarray(0, dtype=np.int32)
    print("Initialised inputs...")
    # --------------------------------------

    start = time.time()
    for i in range(100):
        wav = trt_exec(inputs, speaker_id, hififastpitch)
    print(f"done in {time.time() - start}")

    wav = np.array(wav)
    save_wav(wav=wav, path="models/v1/hi/nosqueeze_transpose/trt_output_e2e.wav", sample_rate=22050)

    for binding in hififastpitch["engine"]:
        err, = cuda.cuStreamDestroy(hififastpitch["data_stream"][binding])

    for node in hififastpitch["io"]["device_mem_address"].keys():
        err, = cuda.cuMemFree(hififastpitch["io"]["device_mem_address"][node])

if __name__ == "__main__":
    # batch()
    trt_batch()