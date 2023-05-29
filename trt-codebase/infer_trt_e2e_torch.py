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
    outputs = allocate_binding_buffer(model["output_types"], model["output_shapes"])

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
        model_path="./models/v1/hi/hififastpitch/best_model.pth",
        config_path="./models/v1/hi/hififastpitch/config.json",
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
    hififastpitch,
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

    hififastpitch["io"]["inputs"]["fastpitch/x"][:np.prod([bs_to_use, x.shape[1]])] = x.flatten()[:]

    hififastpitch["context"].set_binding_shape(hififastpitch["engine"].get_binding_index("fastpitch/x"), 
                                                x.shape)
    
    hififastpitch["io"]["inputs"]["fastpitch/speaker_id"][:int(np.prod(speaker_id.shape))] = speaker_id.flatten()[:]

    assert hififastpitch["context"].all_binding_shapes_specified

    hififastpitch["context"].execute_v2(bindings=hififastpitch["io"]["bindings"])

    cudart.cudaDeviceSynchronize()

    print(hififastpitch["io"]["outputs"]["fastpitch/x_emb"][:bs_to_use * 200 * 512 ].reshape(bs_to_use, 200, 512))


    y_lengths_max = int(torch.max(hififastpitch["io"]["outputs"]["fastpitch/y_lengths"][:bs_to_use]).cpu().numpy().tolist())
    print(hififastpitch["io"]["outputs"]["fastpitch/y_lengths"][:bs_to_use])
    attn = hififastpitch["io"]["outputs"]["fastpitch/alignments"][:bs_to_use * y_lengths_max * x.shape[1] ].reshape((bs_to_use, y_lengths_max, x.shape[1]))
    attn = attn.sum(axis=(-1, -2))
    attn = attn - 5
    # attn = attn.cpu().numpy()
    o_shape = hififastpitch["io"]["outputs"]["hifigan/o_shape"]
    print(o_shape)
    waveform = hififastpitch["io"]["outputs"]["hifigan/o"][:bs_to_use * 1 * o_shape].reshape((bs_to_use, 1, o_shape)).cpu().numpy()
    attn = (attn * int(waveform.shape[2]/attn.max())).to(torch.int)


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
    hififastpitch = load_model("models/v1/hi/final_unsqueeze_notranspose/hififastpitch_with_shape.engine")
    hififastpitch["data_stream"] = {}
    for binding in hififastpitch["engine"]:
        err, hififastpitch["data_stream"][binding] = cuda.cuStreamCreate(1)
    err, hififastpitch["model_stream"] = cuda.cuStreamCreate(1)
    print("Initialised hififastpitch...")

    max_bs = 12
    num_of_mel_feats = 80
    max_encoder_sequence_length = 200
    max_decoder_sequence_length = 1105
    hidden_size = 512
    vocoder_output_shape = 282880

    # setting shape and types for FastPitch
    hififastpitch["input_shapes"] = {
        "fastpitch/x": (max_bs, max_encoder_sequence_length),
        "fastpitch/speaker_id": (1),
    }
    hififastpitch["input_types"] = {
        "fastpitch/x": torch.int32,
        "fastpitch/speaker_id": torch.int32,
    }
    hififastpitch["output_shapes"] = {
        "fastpitch/decoder_output": (max_bs, num_of_mel_feats, max_decoder_sequence_length),  
        "fastpitch/alignments": (max_bs, max_decoder_sequence_length, max_encoder_sequence_length),
        "fastpitch/pitch": (max_bs, 1, max_encoder_sequence_length),
        "fastpitch/durations_log": (max_bs, 1, max_encoder_sequence_length),
        "fastpitch/x_lengths":(max_bs,),
        "fastpitch/y_lengths":(max_bs,),
        "fastpitch/o_en":(max_bs, hidden_size, max_encoder_sequence_length),
        "fastpitch/x_emb":(max_bs, max_encoder_sequence_length, hidden_size),
        "fastpitch/o_dr":(max_bs, ),
        "hifigan/o": (max_bs, 1, vocoder_output_shape),
        "hifigan/o_shape": (1),
    }
    hififastpitch["output_types"] = {
        "fastpitch/decoder_output": torch.float32,
        "fastpitch/alignments": torch.float32,
        "fastpitch/pitch": torch.float32,
        "fastpitch/durations_log": torch.float32,
        "fastpitch/x_lengths": torch.int64,
        "fastpitch/y_lengths": torch.float32,
        "fastpitch/o_en": torch.float32,
        "fastpitch/x_emb": torch.float32,
        "fastpitch/o_dr": torch.float32,
        "hifigan/o": torch.float32,
        "hifigan/o_shape": torch.int64,
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
    for i in range(1):
        wav = trt_exec(inputs, speaker_id, hififastpitch)
    print(f"done in {time.time() - start}")

    # print(wav)

    wav = np.array(wav)
    # wav = wav.cpu().numpy()
    save_wav(wav=wav, path="models/v1/hi/final_unsqueeze_notranspose/trt_output_e2e.wav", sample_rate=22050)

    for binding in hififastpitch["engine"]:
        err, = cuda.cuStreamDestroy(hififastpitch["data_stream"][binding])

    for node in hififastpitch["io"]["device_mem_address"].keys():
        err, = cuda.cuMemFree(hififastpitch["io"]["device_mem_address"][node])

if __name__ == "__main__":
    # batch()
    trt_batch()