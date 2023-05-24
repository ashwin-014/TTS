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

now = datetime.now()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(console_handler_format))
logger.addHandler(console_handler)

log_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/torch")
os.makedirs(log_file_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_file_dir, f'{now.strftime("%H-%M-%S")}.log'))
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)

length_scale = 1

def parse_args():

    parser = argparse.ArgumentParser(description="Decomposing Fastpitch into indivitual components")

    parser.add_argument(
        "--export_embedding",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_encoder",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_duration_predictor",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_pitch_predictor",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_pitch_embedding",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_positional_encoder",
        action="store_true",
        help="Whether to export to onnx or not.",
    )

    parser.add_argument(
        "--export_decoder",
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

def decompose(
    args,
    x,
    speaker_ids=None,
    d_vectors=None,
    decomposed_model_save_path=None,
    # use_torch=True,
    # use_onnx=False,
    # use_trt=False,
):

    
    fastpitch_hifigan = TTS(
        model_path="./models/v1/hi/fastpitch/best_model.pth",
        config_path="./models/v1/hi/fastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    with torch.no_grad():
        fastpitch = fastpitch_hifigan.synthesizer.tts_model
        vocoder = fastpitch_hifigan.synthesizer.vocoder_model
        emb = fastpitch.emb
        encoder = fastpitch.encoder
        duration_predictor = fastpitch.duration_predictor
        # aligner = fastpitch.aligner
        pitch_predictor = fastpitch.pitch_predictor
        pitch_emb = fastpitch.pitch_emb
        pos_encoder = fastpitch.pos_encoder
        decoder = fastpitch.decoder

        x = torch.tensor(x).to("cuda")
        speaker_ids = torch.tensor(speaker_ids).to("cuda")

        #------------------------------------------ FORWARD PASS -------------------------------------------#

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        g = _set_speaker_input(aux_input)

        # ----- Batching support -----

        x_lengths = torch.tensor(x.shape[1]).repeat(x.shape[0]).to(x.device)
        print("DURING X MASK ORIGINAL:    -------------")
        x_mask_original = sequence_mask(x_lengths, x.shape[1]).to(x.dtype)
        x_mask_original = torch.where(x > 0, x_mask_original, 0)
        # x_mask = torch.unsqueeze(x_mask_original, 1).float()
        x_mask = x_mask_original.float()

        # ----- ENCODER - START

        # logger.info(f"Embedding Architecture: {fastpitch.emb} \n\n")

        if g is not None:
            g = g.unsqueeze(-1)

        logger.info(f"Embedding Input Shape- x:{x.shape}")

        x_emb = emb(x)                                          # Emebdding forward pass

        print(x_emb)

        print(x_emb.shape)

        logger.info(f"Embedding Output Shape- x_emb:{x_emb.shape}")

        if args.export_embedding:
            torch.onnx.export(
                model=emb,
                args=(x),
                f=os.path.join(decomposed_model_save_path, "fastpitch_embedding.onnx"),
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
            
        # logger.info(f"Encoder Architecture: {fastpitch.encoder} \n\n")

        encoder_input_x = torch.transpose(x_emb, 1, -1)

        # logger.info(f"Encoder Input Shapes - encoder_input_x:{encoder_input_x.shape}, x_mask:{x_mask.shape}")

        o_en = encoder(encoder_input_x, x_mask)   # Ecoder Forward pass

        # logger.info(f"Encoder Output Shapes - o_en:{o_en.shape}")

        if g is not None:
            o_en = o_en + g

        if args.export_encoder:
            torch.onnx.export(
                model=encoder,
                args=(encoder_input_x, x_mask),
                f=os.path.join(decomposed_model_save_path, "fastpitch_encoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['encoder_input_x', 'x_mask'],
                output_names=['o_en'],
                dynamic_axes={
                    'encoder_input_x': {0: 'bs', 1: 'encoder_hidden_size', 2: 'sequence_length'},
                    # 'x_mask': {0: 'bs', 1: 'bool', 2: 'sequence_length'}
                    'x_mask': {0: 'bs', 1: 'sequence_length'}
                },
            )

        # outputs after encoder pass : o_en, x_mask, g, x_emb

        # ----- ENCODER - END

        # ----- DURATION PREDICTION - START

        # logger.info(f"Duration Predictor Architecture: {fastpitch.duration_predictor} \n\n")

        # logger.info(f"Duration Predictor Input Shapes- o_en:{o_en.shape}, x_mask:{x_mask.shape}")

        o_dr_log = duration_predictor(o_en, x_mask)           # Duration Predictor Forward pass

        # logger.info(f"Duration Predictor Output Shapes- o_dr_log:{o_dr_log.shape}")

        if args.export_duration_predictor:
            torch.onnx.export(
                model=duration_predictor,
                args=(o_en, x_mask),
                f=os.path.join(decomposed_model_save_path, "fastpitch_duration_predictor.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['o_en', 'x_mask'],
                output_names=['o_dr_log'],
                dynamic_axes={
                    'o_en': {0: 'bs', 1: 'encoder_hidden_size', 2: 'sequence_length'},
                    # 'x_mask': {0: 'bs', 1: 'bool', 2: 'sequence_length'}
                    'x_mask': {0: 'bs', 1: 'sequence_length'}
                },
            )

        o_dr = format_durations(o_dr_log, torch.unsqueeze(x_mask, 1)).squeeze(1)    
        # o_dr = format_durations(o_dr_log, x_mask).squeeze(1)    
        o_dr = o_dr * x_mask_original
        y_lengths = o_dr.sum(1)

        # outputs after duration predictor pass: o_dr_log, o_dr, y_lengths

        # ----- DURATION PREDICTION - END

        # ----- PITCH PREDICTOR - START

        # logger.info(f"Pitch Predictor Architecture: {fastpitch.pitch_predictor} \n\n")

        o_pitch = None  # useless statement. For code interpretability during comparison with coqui codebase

        # logger.info(f"Pitch Predictor Input Shapes- o_en:{o_en.shape}, x_mask:{x_mask.shape}")

        o_pitch = pitch_predictor(o_en, x_mask)             # Pitch Predictor Forward pass

        # logger.info(f"Pitch Predictor Output Shapes- o_pitch:{o_pitch.shape}")

        if args.export_pitch_predictor:
            torch.onnx.export(
                model=pitch_predictor,
                args=(o_en, x_mask),
                f=os.path.join(decomposed_model_save_path, "fastpitch_pitch_predictor.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['o_en', 'x_mask'],
                output_names=['o_pitch'],
                dynamic_axes={
                    'o_en': {0: 'bs', 1: 'encoder_hidden_size', 2: 'sequence_length'},
                    # 'x_mask': {0: 'bs', 1: 'bool', 2: 'sequence_length'}
                    'x_mask': {0: 'bs', 1: 'sequence_length'}
                },
            )

        # logger.info(f"Pitch Embedding Architecture: {fastpitch.pitch_emb} \n\n")

        # logger.info(f"Pitch Embedding Input Shapes- o_pitch:{o_pitch.shape}")

        o_pitch_emb = pitch_emb(o_pitch)                    # Pitch Embedding Forward pass

        # logger.info(f"Pitch Embedding Output Shapes- o_pitch_emb:{o_pitch_emb.shape}")

        if args.export_pitch_embedding:
            torch.onnx.export(
                model=pitch_emb,
                args=(o_pitch),
                f=os.path.join(decomposed_model_save_path, "fastpitch_pitch_embedding.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['o_pitch'],
                output_names=['o_pitch_emb'],
                dynamic_axes={
                    'o_pitch': {0: 'bs', 1: 'bool', 2: 'sequence_length'},
                },
            )

        o_en = o_en + o_pitch_emb
        
        # outputs after pitch predictor pass: o_en, o_pitch, o_pitch_emb

        # ----- PITCH PREDICTOR - END

        o_energy = None                     

        # ----- DECODER - START

        # logger.info(f"Positional Encoder Architecture: {fastpitch.pos_encoder} \n\n")

        print(y_lengths.shape)

        print("DURING Y_MASK GENERATION:    -------------")

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)

        # o_en_ex, attn = expand_encoder_outputs(o_en, o_dr, x_mask, y_mask)
        o_en_ex, attn = expand_encoder_outputs(o_en, o_dr, torch.unsqueeze(x_mask, 1), y_mask)

        # logger.info(f"Decoder Positional Encoder Input Shapes- o_en_ex:{o_en_ex.shape}, y_mask:{y_mask.shape}")

        o_en_ex = pos_encoder(o_en_ex, y_mask)              # Positional Encoder Forward pass

        # logger.info(f"Decoder Positional Encoder Output Shapes- o_en_ex:{o_en_ex.shape}")

        if args.export_positional_encoder:
            torch.onnx.export(
                model=pos_encoder,
                args=(o_en_ex, y_mask),
                f=os.path.join(decomposed_model_save_path, "fastpitch_decoder_positional_encoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['o_en_ex_in', 'y_mask'],
                output_names=['o_en_ex_out'],
                dynamic_axes={
                    'o_en_ex_in': {0: 'bs', 1: 'encoder_hidden_size', 2: 'decoder_sequence_length'},
                    'y_mask': {0: 'bs', 1: 'bool', 2: 'decoder_sequence_length'},
                },
            )

        # logger.info(f"Decoder Architecture: {fastpitch.decoder} \n\n")

        # logger.info(f"Decoder Input Shapes- o_en_ex:{o_en_ex.shape}, y_mask:{y_mask.shape}, g:{g.shape}")

        o_de = decoder(o_en_ex, y_mask, g=g)                # Decoder Forward pass

        # logger.info(f"Decoder Output Shapes- o_de:{o_de.shape}")

        if args.export_decoder:
            torch.onnx.export(
                model=decoder,
                args={
                    "x": o_en_ex, 
                    "x_mask": y_mask,
                    "g": g,
                },
                f=os.path.join(decomposed_model_save_path, "fastpitch_decoder.onnx"),
                export_params=True,
                opset_version=16,
                do_constant_folding=True,
                input_names=['o_en_ex', 'y_mask', 'g'],
                output_names=['o_de'],
                dynamic_axes={
                    'o_en_ex': {0: 'bs', 1: 'encoder_hidden_size', 2: 'decoder_sequence_length'},
                    'y_mask': {0: 'bs', 1: 'bool', 2: 'decoder_sequence_length'},
                    # 'g': {0: 'ids'}
                },
            )

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

    decomposed_model_save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "decomposed_model", "onnx")

    os.makedirs(decomposed_model_save_path, exist_ok=True)

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
    # text="मेरा. नमस्ते आपका नाम क्या है."
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
        x=inputs,
        speaker_ids=speaker_ids,
        decomposed_model_save_path=decomposed_model_save_path,
    )

    wav = np.array(wav)
    save_wav(wav=wav, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_decompose_output_torch_unsqueezed.wav"), sample_rate=22050)