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
            size=(np.prod(shape),),
            dtype=types_dict[k],
            device='cuda',
        )
        for k, shape in shapes_dict.items()
    }

def _allocate_memory(model):
    # print(model)
    """Helper function for binding several inputs at once and pre-allocating the results."""
    # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
    inputs = allocate_binding_buffer(model["input_types"], model["input_shapes"])
    # print(model["output_shapes"].keys())
    outputs = allocate_binding_buffer(model["output_types"], model["output_shapes"])
    # print(outputs.keys())

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

def batch():
    tts = TTS(
        model_path="./models/v1/hi/fastpitch/best_model.pth",
        config_path="./models/v1/hi/fastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    start = time.time()
    # text="मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #                     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है.\
    #                      नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #                      नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #                      नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."

    # text="मेरा. नमस्ते. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    text="नमस्ते आपका नाम क्या है. मुझे ये करना हे भारत हैं नाम."
    # text="नमस्ते आपका नाम क्या है."  
    # text = "मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है"
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
    x: np.array,
    speaker_id: np.array,
    fastpitch,
    hifigan,
) -> list:
    bs_to_use = x.shape[0]
    """
    Executes TRT code
    """
    print(x.shape)
    print(x)
    print(speaker_id)
    start = time.time()

    x = torch.tensor(x).to("cuda")
    # x_max = int(torch.max(x, 1).cpu().numpy().tolist())
    speaker_id = torch.tensor(speaker_id).to("cuda")

    # Fastpitch
    fastpitch["io"]["inputs"]["x"][:np.prod([bs_to_use, x.shape[1]])] = x.flatten()[:]
    # print(fastpitch["io"]["inputs"]["x"])


    # err, = cuda.cuMemcpyHtoD(
    #     fastpitch["io"]["device_mem_address"]["x"], 
    #     fastpitch["io"]["inputs"]["x"].ctypes.data, 
    #     fastpitch["io"]["inputs"]["x"].nbytes,
    #     # fastpitch["data_stream"]
    # )
    # fastpitch["io"]["inputs"]["x"][:bs_to_use] = inputs[:bs_to_use]
    # tmp = np.ascontiguousarray(inputs)
    # err, = cuda.cuMemcpyHtoD(
    #     fastpitch["io"]["device_mem_address"]["x"], 
    #     tmp.ctypes.data, 
    #     tmp.nbytes,
    #     # fastpitch["data_stream"]
    # )

    # tmp = np.ascontiguousarray(np.zeros(inputs.shape).astype(np.int32))

    # err, = cuda.cuMemcpyDtoH(
    #     tmp.ctypes.data, 
    #     fastpitch["io"]["device_mem_address"]["x"],
    #     inputs.nbytes,
    #     # fastpitch["data_stream"]
    # )
    
    # print(tmp)

    fastpitch["context"].set_binding_shape(fastpitch["engine"].get_binding_index("x"), x.shape)
    # print(fastpitch["engine"].get_tensor_shape("x"))
    
    # fastpitch["io"]["inputs"]["speaker_id"] = speaker_id
    # err, = cuda.cuMemcpyHtoD(
    #     fastpitch["io"]["device_mem_address"]["speaker_id"], 
    #     fastpitch["io"]["inputs"]["speaker_id"].ctypes.data, 
    #     fastpitch["io"]["inputs"]["speaker_id"].nbytes,
    #     # fastpitch["data_stream"]
    # )
    assert fastpitch["context"].all_binding_shapes_specified
    fastpitch["context"].execute_v2(bindings=fastpitch["io"]["bindings"])

    y_lengths_max = int(torch.max(fastpitch["io"]["outputs"]["y_lengths"][:bs_to_use]).cpu().numpy().tolist())

    print(fastpitch["io"]["outputs"]["x_emb"][:np.prod([bs_to_use, x.shape[1], 512])].reshape([bs_to_use, x.shape[1], 512]))
    print(fastpitch["io"]["outputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])].reshape([bs_to_use, 512, x.shape[1]]))

    # print(np.prod([bs_to_use, x.shape[1]]))
    # print(fastpitch["io"]["outputs"]["o_dr"][:np.prod([bs_to_use, x.shape[1]])].shape)
    print(fastpitch["io"]["outputs"]["o_dr"][:np.prod([bs_to_use, x.shape[1]])].reshape([bs_to_use, x.shape[1]]))
    print(fastpitch["io"]["outputs"]["pitch"][:np.prod([bs_to_use, 1, x.shape[1]])].reshape([bs_to_use, 1, x.shape[1]]))
    print(fastpitch["io"]["outputs"]["o_pitch_emb"][:np.prod([bs_to_use, 512, x.shape[1]])].reshape([bs_to_use, 512, x.shape[1]]))
    print(fastpitch["io"]["outputs"]["x_lengths"][:bs_to_use])
    print(fastpitch["io"]["outputs"]["y_lengths"][:bs_to_use])
    print(fastpitch["io"]["outputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])].reshape([bs_to_use, x.shape[1]]))        
    print(fastpitch["io"]["outputs"]["x_mask_original"][:np.prod([bs_to_use, x.shape[1]])].reshape([bs_to_use, x.shape[1]]))        
    print("Y_MASK: ", fastpitch["io"]["outputs"]["y_mask"][:np.prod([bs_to_use, y_lengths_max])].reshape([bs_to_use, y_lengths_max])) 
    print("Y_MASK_ORIGINAL: ", fastpitch["io"]["outputs"]["y_mask_original"][:np.prod([bs_to_use, y_lengths_max])].reshape([bs_to_use, y_lengths_max]))        
    # print(fastpitch["io"]["outputs"]["o_en_ex_in"][:np.prod([bs_to_use, 512, y_lengths_max])].reshape([bs_to_use, 512, y_lengths_max]))    
    # print(fastpitch["io"]["outputs"]["o_en_ex_out"][:np.prod([bs_to_use, 512, y_lengths_max])].reshape([bs_to_use, 512, y_lengths_max]))    
    print(fastpitch["io"]["outputs"]["model_outputs"][:np.prod([bs_to_use, 80, y_lengths_max])].reshape([bs_to_use, 80, y_lengths_max]))

    # print(fastpitch["engine"].get_binding_shape("model_outputs"))
    # print(fastpitch["engine"].get_binding_shape("alignments"))


    # err, = cuda.cuMemcpyDtoH(
    #     fastpitch["io"]["outputs"]["alignments"].ctypes.data,
    #     fastpitch["io"]["device_mem_address"]["alignments"],
    #     trt.volume((bs_to_use, 817, 24)) * np.dtype(trt.nptype(fastpitch["engine"].get_binding_dtype("alignments"))).itemsize,
    #     # fastpitch["data_stream"]
    # )

    # vocoder_inputs = np.transpose(fastpitch["io"]["outputs"]["model_outputs"][:bs_to_use], (0, 2, 1))
    # print(vocoder_inputs.shape)

    # print(vocoder_inputs[:bs_to_use+1])

    # hifigan["io"]["inputs"]["c"][:bs_to_use] = np.ascontiguousarray(vocoder_inputs)[:bs_to_use]
    
    # err, = cuda.cuMemcpyHtoD(
    #     hifigan["io"]["device_mem_address"]["c"],
    #     hifigan["io"]["inputs"]["c"].ctypes.data,
    #     hifigan["io"]["inputs"]["c"].nbytes,
    #     # hifigan["data_stream"]
    # )

    hifigan["context"].set_binding_shape(hifigan["engine"].get_binding_index("c"), (bs_to_use, 80, 1105))
    # hifigan["io"]["bindings"][hifigan["engine"].get_binding_index("c")] = int(fastpitch["io"]["device_mem_address"]["model_outputs"])
    hifigan["io"]["bindings"][hifigan["engine"].get_binding_index("c")] = int(fastpitch["io"]["device_mem_address"]["model_outputs"])
    assert hifigan["context"].all_binding_shapes_specified
    hifigan["context"].execute_v2(bindings=hifigan["io"]["bindings"])

    # err, = cuda.cuMemcpyDtoH(
    #     hifigan["io"]["outputs"]["o"].ctypes.data,
    #     hifigan["io"]["device_mem_address"]["o"],
    #     trt.volume((bs_to_use, 1, 209152)) * np.dtype(trt.nptype(hifigan["engine"].get_binding_dtype("o"))).itemsize,
    #     # hifigan["data_stream"]
    # )

    # print(hifigan["engine"].get_binding_shape("o"))

    o_shape = hifigan["io"]["outputs"]["o_shape"]
    waveform = hifigan["io"]["outputs"]["o"][:bs_to_use * 1 * o_shape].reshape((bs_to_use, 1, o_shape)).cpu().numpy()

    attn = fastpitch["io"]["outputs"]["alignments"][:np.prod([bs_to_use, y_lengths_max, x.shape[1]])].reshape([bs_to_use, y_lengths_max, x.shape[1]])
    attn = attn.sum(axis=(-1, -2))
    attn = (attn - 5) * 256
    attn = attn.to(torch.int32).to("cpu").numpy()

    wavs = []
    for i, wave in enumerate(waveform):
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
    fastpitch = load_model("models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs.engine")
    # fastpitch = load_model("models/v1/hi/nosqueeze/fastpitch_unsqueeze.engine")
    # fastpitch = load_model("models/v1/hi/nosqueeze/fastpitch.engine")
    # err, fastpitch["data_stream"] = cuda.cuStreamCreate(0)
    # err, fastpitch["model_stream"] = cuda.cuStreamCreate(0)
    print("Initialised fastpitch...")

    hifigan = load_model("models/v1/hi/final_unsqueeze_notranspose/vocoder_with_shape.engine")
    # err, hifigan["data_stream"] = cuda.cuStreamCreate(0)
    # err, hifigan["model_stream"] = cuda.cuStreamCreate(0)
    
    print("Initialised hifigan...")

    max_bs = 12
    num_of_mel_feats = 80
    max_encoder_sequence_length = 200
    max_decoder_sequence_length = 1105
    hidden_size = 512
    vocoder_output_shape = 282880

    # setting shape and types for FastPitch
    fastpitch["input_shapes"] = {
        "x": (max_bs, max_encoder_sequence_length),
        "speaker_id": (1),
    }
    fastpitch["input_types"] = {
        "x": torch.int32,
        "speaker_id": torch.int32,
    }
    fastpitch["output_shapes"] = {
        "model_outputs": (max_bs, num_of_mel_feats, max_decoder_sequence_length),  
        "alignments": (max_bs, max_decoder_sequence_length, max_encoder_sequence_length),
        "pitch": (max_bs, 1, max_encoder_sequence_length),
        "durations_log": (max_bs, 1, max_encoder_sequence_length),
        "x_lengths":(max_bs,),
        "y_lengths":(max_bs,),
        "o_en":(max_bs, hidden_size, max_encoder_sequence_length),
        "x_emb":(max_bs, max_encoder_sequence_length, hidden_size),
        "o_dr":(max_bs, max_encoder_sequence_length),
        "o_pitch_emb": (max_bs, hidden_size, max_encoder_sequence_length),
        "o_en_ex_in": (max_bs, hidden_size, max_decoder_sequence_length),
        "o_en_ex_out": (max_bs, hidden_size, max_decoder_sequence_length),
        "y_mask": (max_bs, max_decoder_sequence_length),
        "y_mask_original": (max_bs, max_decoder_sequence_length),
        "x_mask": (max_bs, 1, max_encoder_sequence_length),
        "x_mask_original": (max_bs, 1, max_encoder_sequence_length),
    }
    fastpitch["output_types"] = {
        "model_outputs": torch.float32,
        "alignments": torch.float32,
        "pitch": torch.float32,
        "durations_log": torch.float32,
        "x_lengths": torch.int64,
        "y_lengths": torch.float32,
        "o_en": torch.float32,
        "x_emb": torch.float32,
        "o_dr": torch.float32,
        "o_pitch_emb": torch.float32,
        "o_en_ex_in": torch.float32,
        "o_en_ex_out": torch.float32,
        "y_mask": torch.float32,
        "y_mask_original": torch.int8,
        "x_mask": torch.float32,
        "x_mask_original": torch.float32,
    }
    fastpitch["io"] = _allocate_memory(fastpitch)

    # setting shape and types for HiFiGAN
    hifigan["input_shapes"] = {
        "c": (max_bs, num_of_mel_feats, max_decoder_sequence_length),
    }
    hifigan["input_types"] = {
        "c": torch.float32,
    }
    hifigan["output_shapes"] = {
        "o": (max_bs, 1, vocoder_output_shape),
        "o_shape": (1)
    }
    hifigan["output_types"] = {
        "o": torch.float32,
        "o_shape": torch.int64
    }
    hifigan["io"] = _allocate_memory(hifigan)
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
    for i in range(1):
        wav = trt_exec(inputs, speaker_id, fastpitch, hifigan)
    print(f"done in {time.time() - start}")

    wav = np.array(wav)
    # save_wav(wav=wav, path="models/v1/hi/nosqueeze/trt_output.wav", sample_rate=22050)
    save_wav(wav=wav, path="models/v1/hi/final_unsqueeze_notranspose/trt_output_torch.wav", sample_rate=22050)

if __name__ == "__main__":
    # batch()
    trt_batch()