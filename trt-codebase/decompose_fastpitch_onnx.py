import os
import time
import numpy as np
import scipy
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
import torch
import tensorrt as trt
import logging
from datetime import datetime
import pysbd
from typing import Dict
from itertools import zip_longest
import argparse
import onnx
import onnxruntime as ort
import tensorrt as trt

now = datetime.now()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(console_handler_format))
logger.addHandler(console_handler)

log_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/onnx")
os.makedirs(log_file_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_file_dir, f'{now.strftime("%H-%M-%S")}.log'))
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)

length_scale = 1

def parse_args():

    parser = argparse.ArgumentParser(description="Decomposing Fastpitch into indivitual components")

    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        required=True,
        help="Directory path of onnx models",
    )

    parser.add_argument(
        "--build_embedding",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_encoder",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_duration_predictor",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_pitch_predictor",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_pitch_embedding",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_positional_encoder",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--build_decoder",
        action="store_true",
        help="Whether to export to onnx or not.",
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

def load_models(onnx_model_paths):

    models = {}

    for model_name, onnx_path in onnx_model_paths.items():
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        models[model_name] = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider"]
        )

    return models

def build_engine(onnx_model_path, engine_model_path, dim_specs):

    trt_logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)
    # config.set_flag(trt.BuilderFlag.DEBUG)
    # config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    # config.set_flag(trt.BuilderFlag.REFIT)
    # config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    # config.set_flag(trt.BuilderFlag.TF32)
    # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.DIRECT_IO)
    # config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    # config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    # config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
    # config.set_flag(trt.BuilderFlag.FP8)
    # config.set_flag(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
    # config.set_flag(trt.HardwareCompatibilityLevel.NONE)
    # config.set_flag(trt.HardwareCompatibilityLevel.AMPERE_PLUS)

    config.builder_optimization_level = 5

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

def decompose(
    args,
    onnx_model_paths,
    x,
    speaker_ids=None,
    d_vectors=None,
    engine_save_dir=None
):

    fastpitch_hifigan = TTS(
        model_path="./models/v1/hi/fastpitch/best_model.pth",
        config_path="./models/v1/hi/fastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    fastpitch = fastpitch_hifigan.synthesizer.tts_model
    vocoder = fastpitch_hifigan.synthesizer.vocoder_model
    emb = fastpitch.emb
    encoder = fastpitch.encoder
    duration_predictor = fastpitch.duration_predictor
    pitch_predictor = fastpitch.pitch_predictor
    pitch_emb = fastpitch.pitch_emb
    pos_encoder = fastpitch.pos_encoder
    decoder = fastpitch.decoder

    models = load_models(onnx_model_paths)

    with torch.no_grad():
        

        x = torch.tensor(x).to("cpu")
        speaker_ids = torch.tensor(speaker_ids).to("cpu")

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

        # x_emb = emb(x)                                          # Emebdding forward pass

        x_emb = models["embedding"].run(["x_emb"], {"x": x.cpu().numpy()})
        x_emb = torch.tensor(x_emb[0])

        logger.info(f"Embedding Output Shape- x_emb:{x_emb.shape}")

        if args.build_embedding:
            dim_specs = {
                "x": {
                    "min": (1, 1),
                    "opt": (2, 24),
                    "max": (6, 400),
                },
            }
            if build_engine(
                onnx_model_paths["embedding"],
                os.path.join(engine_save_dir, "fastpitch_embedding.engine"),
                dim_specs
            ):
                logger.info(f"Embedding Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_embedding.engine')}")
            
        encoder_input_x = torch.transpose(x_emb, 1, -1)

        logger.info(f"Encoder Input Shapes - encoder_input_x:{encoder_input_x.shape}, x_mask:{x_mask.shape}")

        # o_en = encoder(encoder_input_x, x_mask)   # Ecoder Forward pass

        o_en = models["encoder"].run(
            ["o_en"],
            {"encoder_input_x": encoder_input_x.cpu().numpy(), "x_mask": x_mask.cpu().numpy()})
        o_en = torch.tensor(o_en[0])

        logger.info(f"Encoder Output Shapes - o_en:{o_en.shape}")

        if g is not None:
            o_en = o_en + g

        if args.build_encoder:
            dim_specs = {
                "encoder_input_x": {
                    "min": (1, 512, 1),
                    "opt": (2, 512, 24),
                    "max": (6, 512, 400),
                },
                "x_mask": {
                    "min": (1, 1),
                    "opt": (2, 24),
                    "max": (6, 400),
                }
            }
            if build_engine(
                onnx_model_paths["encoder"],
                os.path.join(engine_save_dir, "fastpitch_encoder.engine"),
                dim_specs
            ):
                logger.info(f"Encoder Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_encoder.engine')}")

        # outputs after encoder pass : o_en, x_mask, g, x_emb

        # ----- ENCODER - END

        # ----- DURATION PREDICTION - START

        logger.info(f"Duration Predictor Input Shapes- o_en:{o_en.shape}, x_mask:{x_mask.shape}")

        # o_dr_log = duration_predictor(o_en, x_mask)           # Duration Predictor Forward pass

        o_dr_log = models["duration_predictor"].run(["o_dr_log"], 
            {"o_en": o_en.cpu().numpy(), "x_mask": x_mask.cpu().numpy()})
        o_dr_log = torch.tensor(o_dr_log[0])

        logger.info(f"Duration Predictor Output Shapes- o_dr_log:{o_dr_log.shape}")

        if args.build_duration_predictor:
            dim_specs = {
                "o_en": {
                    "min": (1, 512, 1),
                    "opt": (2, 512, 24),
                    "max": (6, 512, 400),
                },
                "x_mask": {
                    "min": (1, 1),
                    "opt": (2, 24),
                    "max": (6, 400),
                }
            }
            if build_engine(
                onnx_model_paths["duration_predictor"],
                os.path.join(engine_save_dir, "fastpitch_duration_predictor.engine"),
                dim_specs
            ):
                logger.info(f"Duration Predictor Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_duration_predictor.engine')}")

        o_dr = format_durations(o_dr_log, torch.unsqueeze(x_mask, 1)).squeeze(1)    
        o_dr = o_dr * x_mask_original
        y_lengths = o_dr.sum(1)

        # outputs after duration predictor pass: o_dr_log, o_dr, y_lengths

        # ----- DURATION PREDICTION - END

        # ----- PITCH PREDICTOR - START

        o_pitch = None  # useless statement. For code interpretability during comparison with coqui codebase

        logger.info(f"Pitch Predictor Input Shapes- o_en:{o_en.shape}, x_mask:{x_mask.shape}")

        # o_pitch = pitch_predictor(o_en, x_mask)             # Pitch Predictor Forward pass

        o_pitch = models["pitch_predictor"].run(
            ["o_pitch"],
            {"o_en": o_en.cpu().numpy(), "x_mask": x_mask.cpu().numpy()},
        )
        o_pitch = torch.tensor(o_pitch[0])

        logger.info(f"Pitch Predictor Output Shapes- o_pitch:{o_pitch.shape}")

        if args.build_pitch_predictor:
            dim_specs = {
                "o_en": {
                    "min": (1, 512, 1),
                    "opt": (2, 512, 24),
                    "max": (6, 512, 400),
                },
                "x_mask": {
                    "min": (1, 1),
                    "opt": (2, 24),
                    "max": (6, 400),
                }
            }
            if build_engine(
                onnx_model_paths["pitch_predictor"],
                os.path.join(engine_save_dir, "fastpitch_pitch_predictor.engine"),
                dim_specs
            ):
                logger.info(f"Pitch Predictor Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_pitch_predictor.engine')}")

        logger.info(f"Pitch Embedding Input Shapes- o_pitch:{o_pitch.shape}")

        # o_pitch_emb = pitch_emb(o_pitch)                    # Pitch Embedding Forward pass

        o_pitch_emb = models["pitch_embedding"].run(
            ["o_pitch_emb"],
            {"o_pitch": o_pitch.cpu().numpy()},
        )
        o_pitch_emb = torch.tensor(o_pitch_emb[0])

        logger.info(f"Pitch Embedding Output Shapes- o_pitch_emb:{o_pitch_emb.shape}")

        if args.build_pitch_embedding:
            dim_specs = {
                "o_pitch": {
                    "min": (1, 1, 1),
                    "opt": (2, 1, 24),
                    "max": (6, 1, 400),
                },
            }
            if build_engine(
                onnx_model_paths["pitch_embedding"],
                os.path.join(engine_save_dir, "fastpitch_pitch_embedding.engine"),
                dim_specs
            ):
                logger.info(f"Pitch Embedding Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_pitch_embedding.engine')}")  

        o_en = o_en + o_pitch_emb
        
        # outputs after pitch predictor pass: o_en, o_pitch, o_pitch_emb

        # ----- PITCH PREDICTOR - END

        o_energy = None                     

        # ----- DECODER - START

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)

        o_en_ex, attn = expand_encoder_outputs(o_en, o_dr, torch.unsqueeze(x_mask, 1), y_mask)

        logger.info(f"Decoder Positional Encoder Input Shapes- o_en_ex:{o_en_ex.shape}, y_mask:{y_mask.shape}")

        # o_en_ex = pos_encoder(o_en_ex, y_mask)              # Positional Encoder Forward pass

        o_en_ex = models["positional_encoder"].run(
            ["o_en_ex_out"],
            {"o_en_ex_in": o_en_ex.cpu().numpy(), "y_mask": y_mask.cpu().numpy()}
        )
        o_en_ex = torch.tensor(o_en_ex[0])

        logger.info(f"Decoder Positional Encoder Output Shapes- o_en_ex:{o_en_ex.shape}")

        if args.build_positional_encoder:
            dim_specs = {
                "o_en_ex_in": {
                    "min": (1, 512, 1),
                    "opt": (2, 512, 134),
                    "max": (6, 512, 400),
                },
                "y_mask": {
                    "min": (1, 1, 1),
                    "opt": (2, 1, 134),
                    "max": (6, 1, 400),
                }
            }
            if build_engine(
                onnx_model_paths["positional_encoder"],
                os.path.join(engine_save_dir, "fastpitch_positional_encoder.engine"),
                dim_specs
            ):
                logger.info(f"Positional Encoder Engine built successfully - \
                                {os.path.join(engine_save_dir, 'fastpitch_positional_encoder.engine')}")

        logger.info(f"Decoder Input Shapes- o_en_ex:{o_en_ex.shape}, y_mask:{y_mask.shape}, g:{g.shape}")

        # o_de = decoder(o_en_ex, y_mask, g=g)                # Decoder Forward pass

        o_de = models["decoder"].run(
            ["o_de"],
            {"o_en_ex": o_en_ex.cpu().numpy(), "y_mask": y_mask.cpu().numpy()}
        )
        o_de = torch.tensor(o_de[0])

        logger.info(f"Decoder Output Shapes- o_de:{o_de.shape}")

        if args.build_decoder:
            dim_specs = {
                "o_en_ex": {
                    "min": (1, 512, 1),
                    "opt": (2, 512, 134),
                    "max": (6, 512, 400),
                },
                "y_mask": {
                    "min": (1, 1, 1),
                    "opt": (2, 1, 134),
                    "max": (6, 1, 400),
                }
            }
            if build_engine(
                onnx_model_paths["decoder"],
                os.path.join(engine_save_dir, "fastpitch_decoder.engine"),
                dim_specs
            ):
                logger.info(f"Decoder Engine built successfully - \
                    {os.path.join(engine_save_dir, 'fastpitch_decoder.engine')}")

    # outputs after decoder pass: y_mask, o_en_ex, attn, o_de
    
    # ----- DECODER - END
    
    
    outputs = {
        "model_outputs": o_de,
        "alignments": attn,
        "pitch": o_pitch,
        "energy": o_energy,
        "durations_log": o_dr_log,
    }

    print("Fastpitch Decomposition completed successfully!")    

    # print(o_de)

    # print(o_de.shape)

    waveform = vocoder.inference(outputs["model_outputs"])

    attn = torch.sum(outputs["alignments"], dim=(-1, -2))
    attn = attn - 5
    attn = (attn * int(waveform.shape[2]/attn.max())).to(torch.int)

    waveform = waveform.cpu().numpy()

    wavs = []
    for i, wave in enumerate(waveform):
        wave = wave.squeeze()[:attn[i]]
        wave = wave.squeeze()
        wavs += list(wave)
        wavs += [0] * 10000
    
    # process_time = time.time() - start
    # audio_time = len(wavs) / 22050
    # print(f" > Processing time: {process_time}")
    # print(f" > Real-time factor: {process_time / audio_time}")
    return wavs

if __name__ == "__main__":

    args = parse_args()

    onnx_model_paths = {
        "embedding": os.path.join(args.onnx_model_dir, "fastpitch_embedding.onnx"),
        "encoder": os.path.join(args.onnx_model_dir, "fastpitch_encoder.onnx"),
        "duration_predictor": os.path.join(args.onnx_model_dir, "fastpitch_duration_predictor.onnx"),
        "pitch_predictor": os.path.join(args.onnx_model_dir, "fastpitch_pitch_predictor.onnx"),
        "pitch_embedding": os.path.join(args.onnx_model_dir, "fastpitch_pitch_embedding.onnx"),
        "positional_encoder": os.path.join(args.onnx_model_dir, "fastpitch_decoder_positional_encoder.onnx"),
        "decoder": os.path.join(args.onnx_model_dir, "fastpitch_decoder.onnx"),
    }

    engine_save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "decomposed_model", "tensorrt")

    os.makedirs(engine_save_dir, exist_ok=True)

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
    speaker_ids = np.asarray(0, dtype=np.int32)
    print("Initialised inputs...")
    # --------------------------------------

    wav = decompose(
        args=args,
        onnx_model_paths=onnx_model_paths,
        x=inputs,
        speaker_ids=speaker_ids,
        engine_save_dir=engine_save_dir
    )

    wav = np.array(wav)
    save_wav(wav=wav, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_decompose_output_onnx.wav"), sample_rate=22050)