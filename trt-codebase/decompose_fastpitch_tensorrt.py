import os
import time
import pysbd
import numpy as np
import scipy
import tensorrt as trt
from cuda import cuda, cudart, nvrtc
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
# import torch
from itertools import zip_longest
from datetime import datetime
import logging
import argparse
import torch

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

now = datetime.now()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(console_handler_format))
logger.addHandler(console_handler)

log_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/tensorrt")
os.makedirs(log_file_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_file_dir, f'{now.strftime("%H-%M-%S")}.log'))
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)

length_scale = 1

def parse_args():

    parser = argparse.ArgumentParser(description="Decomposing Fastpitch into indivitual components")

    parser.add_argument(
        "--trt_model_dir",
        type=str,
        required=True,
        help="Directory path of tensorrt models",
    )

    args = parser.parse_args()

    return args

def _set_speaker_input(aux_input):
    d_vectors = aux_input.get("d_vectors", None)
    speaker_ids = aux_input.get("speaker_ids", None)
    g = speaker_ids if speaker_ids is not None else d_vectors
    return g

def format_durations(o_dr_log, x_mask):
    o_dr = (torch.exp(o_dr_log) - 1) * x_mask * length_scale
    o_dr[o_dr < 1] = 1.0
    o_dr = torch.round(o_dr)
    return o_dr

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None) -> None:
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))

def generate_attn(dr, x_mask, y_mask=None):
    # compute decode mask from the durations
    if y_mask is None:
        y_lengths = dr.sum(1).long()
        y_lengths[y_lengths < 1] = 1
        print("DURING GENERATE ATTEN:    -------------")
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
    attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
    return attn

def expand_encoder_outputs(en, dr, x_mask, y_mask):
    attn = generate_attn(dr, x_mask, y_mask)
    # o_en_ex = torch.matmul(attn.squeeze(1).transpose(1, 2).to(en.dtype), en.transpose(1, 2)).transpose(1, 2)
    o_en_ex = torch.matmul(attn.transpose(1, 2).to(en.dtype), en.transpose(1, 2)).transpose(1, 2)
    return o_en_ex, attn

def load_trt_models(trt_model_paths, models_dim_specs):

    models = {}

    for model_name, trt_path in trt_model_paths.items():
        models[model_name] = load_model(trt_path)
        models[model_name]["data_stream"] = {}
        for binding in models[model_name]["engine"]:
            err, models[model_name]["data_stream"][binding] = cuda.cuStreamCreate(1)
        err, models[model_name]["model_stream"] = cuda.cuStreamCreate(1)
        models[model_name]["dim_specs"] = models_dim_specs[model_name]
        models[model_name]["io"] = _allocate_memory(models[model_name])
        
    return models

def release_trt_resources(trt_models):

    models = {}

    for model_name, model in trt_models.items():
        for binding in model["engine"]:
            err, = cuda.cuStreamDestroy(model["data_stream"][binding])

        for node in model["io"]["device_mem_address"].keys():
            err, = cuda.cuMemFree(model["io"]["device_mem_address"][node])    

        err, = cuda.cuStreamDestroy(model["model_stream"])    

    return models

def decompose(
    args,
    trt_model_paths,
    trt_models_dim_specs,
    x,
    speaker_ids=None,
    d_vectors=None,
):

    bs_to_use = x.shape[0]

    fastpitch_hifigan = TTS(
        model_path="./models/v1/hi/fastpitch/best_model.pth",
        config_path="./models/v1/hi/fastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    # fastpitch = fastpitch_hifigan.synthesizer.tts_model
    vocoder = fastpitch_hifigan.synthesizer.vocoder_model

    models = load_trt_models(trt_model_paths, trt_models_dim_specs)

    wavs = []

    with torch.no_grad():
        

        x = torch.tensor(x).to("cuda")
        speaker_ids = torch.tensor(speaker_ids).to("cuda")

        #------------------------------------------ FORWARD PASS -------------------------------------------#

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        g = _set_speaker_input(aux_input)

        # ----- Batching support -----

        x_lengths = torch.tensor(x.shape[1]).repeat(x.shape[0]).to(x.device)
        x_mask_original = sequence_mask(x_lengths, x.shape[1]).to(x.dtype)
        x_mask_original = torch.where(x > 0, x_mask_original, 0)
        # x_mask = torch.unsqueeze(x_mask_original, 1).float()
        x_mask = x_mask_original.float()
        

        # ----- ENCODER - START

        if g is not None:
            g = g.unsqueeze(-1)

        logger.info(f"Embedding Input Shape- {x.shape}")

        models["embedding"]["io"]["inputs"]["x"][:np.prod([bs_to_use, x.shape[1]])] = x.flatten()[:]
        models["embedding"]["context"].set_binding_shape(models["embedding"]["engine"].get_binding_index("x"), x.shape)
        assert models["embedding"]["context"].all_binding_shapes_specified
        models["embedding"]["context"].execute_v2(bindings=models["embedding"]["io"]["bindings"])
        # print(torch.reshape(models["embedding"]["io"]["outputs"]["x_emb"][:np.prod([bs_to_use, x.shape[1], 512])],
        #                     [bs_to_use, x.shape[1], 512]))
        x_emb = torch.reshape(models["embedding"]["io"]["outputs"]["x_emb"][:np.prod([bs_to_use, x.shape[1], 512])],
                              [bs_to_use, x.shape[1], 512])
        x_emb = torch.transpose(x_emb, 1, -1)



        models["encoder"]["io"]["inputs"]["encoder_input_x"][:np.prod([bs_to_use, x.shape[1], 512])] = x_emb.flatten()[:]
        models["encoder"]["context"].set_binding_shape(models["encoder"]["engine"].get_binding_index("encoder_input_x"), x_emb.shape)
        models["encoder"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        models["encoder"]["context"].set_binding_shape(models["encoder"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert models["encoder"]["context"].all_binding_shapes_specified
        models["encoder"]["context"].execute_v2(bindings=models["encoder"]["io"]["bindings"])
        # print(torch.reshape(models["encoder"]["io"]["outputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])],
        #                     [bs_to_use, 512, x.shape[1]]))
        o_en = torch.reshape(models["encoder"]["io"]["outputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])],
                             [bs_to_use, 512, x.shape[1]])
        if g is not None:
            o_en = o_en + g

    #     # ----- ENCODER - END



    #     # ----- DURATION PREDICTION - START

        models["duration_predictor"]["io"]["inputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])] = o_en.flatten()[:]
        models["duration_predictor"]["context"].set_binding_shape(models["duration_predictor"]["engine"].get_binding_index("o_en"), o_en.shape)
        models["duration_predictor"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        models["duration_predictor"]["context"].set_binding_shape(models["duration_predictor"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert models["duration_predictor"]["context"].all_binding_shapes_specified
        models["duration_predictor"]["context"].execute_v2(bindings=models["duration_predictor"]["io"]["bindings"])
        # print(torch.reshape(models["duration_predictor"]["io"]["outputs"]["o_dr_log"][:np.prod([bs_to_use, 1, x.shape[1]])],
        #                     [bs_to_use, 1, x.shape[1]]))
        o_dr_log = torch.reshape(models["duration_predictor"]["io"]["outputs"]["o_dr_log"][:np.prod([bs_to_use, 1, x.shape[1]])],
                                 [bs_to_use, 1, x.shape[1]])

        o_dr = format_durations(o_dr_log, torch.unsqueeze(x_mask, 1)).squeeze(1)    
        o_dr = o_dr * x_mask_original
        y_lengths = o_dr.sum(1)

        # print(y_lengths)
        # print(y_lengths.shape)

        # outputs after duration predictor pass: o_dr_log, o_dr, y_lengths
        # ----- DURATION PREDICTION - END



        # ----- PITCH PREDICTOR - START

        o_pitch = None  # useless statement. For code interpretability during comparison with coqui codebase
        models["pitch_predictor"]["io"]["inputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])] = o_en.flatten()[:]
        models["pitch_predictor"]["context"].set_binding_shape(models["pitch_predictor"]["engine"].get_binding_index("o_en"), o_en.shape)
        models["pitch_predictor"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        models["pitch_predictor"]["context"].set_binding_shape(models["pitch_predictor"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert models["pitch_predictor"]["context"].all_binding_shapes_specified
        models["pitch_predictor"]["context"].execute_v2(bindings=models["pitch_predictor"]["io"]["bindings"])
        # print(torch.reshape(models["pitch_predictor"]["io"]["outputs"]["o_pitch"][:np.prod([bs_to_use, 1, x.shape[1]])],
        #                     [bs_to_use, 1, x.shape[1]]))
        o_pitch = torch.reshape(models["pitch_predictor"]["io"]["outputs"]["o_pitch"][:np.prod([bs_to_use, 1, x.shape[1]])],
                                [bs_to_use, 1, x.shape[1]])



        models["pitch_embedding"]["io"]["inputs"]["o_pitch"][:np.prod([bs_to_use, 1, x.shape[1]])] = o_pitch.flatten()[:]
        models["pitch_embedding"]["context"].set_binding_shape(models["pitch_embedding"]["engine"].get_binding_index("o_pitch"), o_pitch.shape)
        assert models["pitch_embedding"]["context"].all_binding_shapes_specified
        models["pitch_embedding"]["context"].execute_v2(bindings=models["pitch_embedding"]["io"]["bindings"])
        # print(torch.reshape(models["pitch_embedding"]["io"]["outputs"]["o_pitch_emb"][:np.prod([bs_to_use, 512, x.shape[1]])],
        #                     [bs_to_use, 512, x.shape[1]]))
        o_pitch_emb = torch.reshape(models["pitch_embedding"]["io"]["outputs"]["o_pitch_emb"][:np.prod([bs_to_use, 512, x.shape[1]])],
                                    [bs_to_use, 512, x.shape[1]])
        o_en = o_en + o_pitch_emb
        
        # outputs after pitch predictor pass: o_en, o_pitch, o_pitch_emb
        # ----- PITCH PREDICTOR - END

        o_energy = None                     

        # ----- DECODER - START

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        o_en_ex, attn = expand_encoder_outputs(o_en, o_dr, torch.unsqueeze(x_mask, 1), y_mask)
        y_length_max = int(torch.max(y_lengths).cpu().numpy().tolist())

        # print(y_mask)

        # logger.info(f"Decoder Positional Encoder Input Shapes- o_en_ex:{o_en_ex.shape}, y_mask:{y_mask.shape}")

        models["positional_encoder"]["io"]["inputs"]["o_en_ex_in"][:np.prod([bs_to_use, 512, y_length_max])] = o_en_ex.flatten()[:]
        models["positional_encoder"]["context"].set_binding_shape(models["positional_encoder"]["engine"].get_binding_index("o_en_ex_in"), o_en_ex.shape)
        models["positional_encoder"]["io"]["inputs"]["y_mask"][:np.prod([bs_to_use, 1, y_length_max])] = y_mask.flatten()[:]
        models["positional_encoder"]["context"].set_binding_shape(models["positional_encoder"]["engine"].get_binding_index("y_mask"), y_mask.shape)
        assert models["positional_encoder"]["context"].all_binding_shapes_specified
        models["positional_encoder"]["context"].execute_v2(bindings=models["positional_encoder"]["io"]["bindings"])
        # print(torch.reshape(models["positional_encoder"]["io"]["outputs"]["o_en_ex_out"][:np.prod([bs_to_use, 512, y_length_max])],
        #                     [bs_to_use, 512, y_length_max]))
        o_en_ex_out = torch.reshape(models["positional_encoder"]["io"]["outputs"]["o_en_ex_out"][:np.prod([bs_to_use, 512, y_length_max])],
                                    [bs_to_use, 512, y_length_max])



        models["decoder"]["io"]["inputs"]["o_en_ex"][:np.prod([bs_to_use, 512, y_length_max])] = o_en_ex_out.flatten()[:]
        models["decoder"]["context"].set_binding_shape(models["decoder"]["engine"].get_binding_index("o_en_ex"), o_en_ex_out.shape)
        models["decoder"]["io"]["inputs"]["y_mask"][:np.prod([bs_to_use, 1, y_length_max])] = y_mask.flatten()[:]
        models["decoder"]["context"].set_binding_shape(models["decoder"]["engine"].get_binding_index("y_mask"), y_mask.shape)
        assert models["decoder"]["context"].all_binding_shapes_specified
        models["decoder"]["context"].execute_v2(bindings=models["decoder"]["io"]["bindings"])
        # print(torch.reshape(models["decoder"]["io"]["outputs"]["o_de"][:np.prod([bs_to_use, 80, y_length_max])],
        #                     [bs_to_use, 80, y_length_max]))
        o_de = torch.reshape(models["decoder"]["io"]["outputs"]["o_de"][:np.prod([bs_to_use, 80, y_length_max])],
                            [bs_to_use, 80, y_length_max])

        # outputs after decoder pass: y_mask, o_en_ex, attn, o_de
    
        # ----- DECODER - END
    
    
    outputs = {
        "model_outputs": o_de,
        "alignments": attn,
        "pitch": o_pitch,
        "energy": o_energy,
        "durations_log": o_dr_log,
    }

    # print("Fastpitch Decomposition completed successfully!")    

    # # print(o_de)

    # # print(o_de.shape)

    waveform = vocoder.inference(outputs["model_outputs"])

    attn = torch.sum(outputs["alignments"], dim=(-1, -2))
    attn = attn - 5
    attn = (attn * int(waveform.shape[2]/attn.max())).to(torch.int)

    waveform = waveform.cpu().numpy()

    for i, wave in enumerate(waveform):
        wave = wave.squeeze()[:attn[i]]
        wave = wave.squeeze()
        wavs += list(wave)
        wavs += [0] * 10000
    
    # process_time = time.time() - start
    # audio_time = len(wavs) / 22050
    # print(f" > Processing time: {process_time}")
    # print(f" > Real-time factor: {process_time / audio_time}")

    release_trt_resources(models)

    return wavs

if __name__ == "__main__":

    args = parse_args()

    trt_model_paths = {
        "embedding": os.path.join(args.trt_model_dir, "fastpitch_embedding.engine"),
        "encoder": os.path.join(args.trt_model_dir, "fastpitch_encoder.engine"),
        "duration_predictor": os.path.join(args.trt_model_dir, "fastpitch_duration_predictor.engine"),
        "pitch_predictor": os.path.join(args.trt_model_dir, "fastpitch_pitch_predictor.engine"),
        "pitch_embedding": os.path.join(args.trt_model_dir, "fastpitch_pitch_embedding.engine"),
        "positional_encoder": os.path.join(args.trt_model_dir, "fastpitch_positional_encoder.engine"),
        "decoder": os.path.join(args.trt_model_dir, "fastpitch_decoder.engine"),
    }

    max_bs = 6
    max_sequence_length = 400
    encoder_hidden_size = 512
    max_decoder_sequence_length = 817

    trt_models_dim_specs = {
        "embedding": {
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
        },
        "encoder": {
            "input_shapes": {
                "encoder_input_x": (max_bs, encoder_hidden_size, max_sequence_length),
                "x_mask": (max_bs, max_sequence_length),
            },
            "input_types": {
                "encoder_input_x": torch.float32,
                "x_mask": torch.float32,
            },
            "output_shapes": {
                "o_en": (max_bs, encoder_hidden_size, max_sequence_length),
            },
            "output_types": {
                "o_en": torch.float32,
            }
        },
        "duration_predictor": {
            "input_shapes": {
                "o_en": (max_bs, encoder_hidden_size, max_sequence_length),
                "x_mask": (max_bs, max_sequence_length),
            },
            "input_types": {
                "o_en": torch.float32,
                "x_mask": torch.float32,
            },
            "output_shapes": {
                "o_dr_log": (max_bs, 1, max_sequence_length),
            },
            "output_types": {
                "o_dr_log": torch.float32,
            }
        },
        "pitch_predictor": {
            "input_shapes": {
                "o_en": (max_bs, encoder_hidden_size, max_sequence_length),
                "x_mask": (max_bs, max_sequence_length),
            },
            "input_types": {
                "o_en": torch.float32,
                "x_mask": torch.float32,
            },
            "output_shapes": {
                "o_pitch": (max_bs, 1, max_sequence_length),
            },
            "output_types": {
                "o_pitch": torch.float32,
            }
        },
        "pitch_embedding": {
            "input_shapes": {
                "o_pitch": (max_bs, 1, max_sequence_length),
            },
            "input_types": {
                "o_pitch": torch.float32,
            },
            "output_shapes": {
                "o_pitch_emb": (max_bs, encoder_hidden_size, max_sequence_length),
            },
            "output_types": {
                "o_pitch_emb": torch.float32,
            }
        },
        "positional_encoder": {
            "input_shapes": {
                "o_en_ex_in": (max_bs, encoder_hidden_size, max_decoder_sequence_length),
                "y_mask": (max_bs, 1, max_decoder_sequence_length),
            },
            "input_types": {
                "o_en_ex_in": torch.float32,
                "y_mask": torch.float32,
            },
            "output_shapes": {
                "o_en_ex_out": (max_bs, encoder_hidden_size, max_decoder_sequence_length),
            },
            "output_types": {
                "o_en_ex_out": torch.float32,
            }
        },
        "decoder": {
            "input_shapes": {
                "o_en_ex": (max_bs, encoder_hidden_size, max_decoder_sequence_length),
                "y_mask": (max_bs, 1, max_decoder_sequence_length),
            },
            "input_types": {
                "o_en_ex": torch.float32,
                "y_mask": torch.float32,
            },
            "output_shapes": {
                "o_de": (max_bs, 80, max_decoder_sequence_length),
            },
            "output_types": {
                "o_de": torch.float32,
            }
        },
    }

    # --------------------------------------
    # Text init
    # --------------------------------------
    text="मेरा. नमस्ते. \
        नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
        नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
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
    speaker_ids = np.asarray(0, dtype=np.int32)
    print("Initialised inputs...")
    # --------------------------------------

    wav = decompose(
        args=args,
        trt_model_paths=trt_model_paths,
        trt_models_dim_specs=trt_models_dim_specs,
        x=inputs,
        speaker_ids=speaker_ids,
    )

    wav = np.array(wav)
    save_wav(wav=wav, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_decompose_output_tensorrt.wav"), sample_rate=22050)