import os
import time
import numpy as np
import scipy
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
import torch
import tensorrt as trt
import logging
from datetime import datetime

now = datetime.now()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(console_handler_format))
logger.addHandler(console_handler)

log_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_file_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_file_dir, f'{now.strftime("%H-%M-%S")}.log'))
file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
file_handler.setFormatter(logging.Formatter(file_handler_format))
logger.addHandler(file_handler)


def decompose(args):
    
    fastpitch_hifigan = TTS(
        model_path="./models/v1/hi/fastpitch/best_model.pth",
        config_path="./models/v1/hi/fastpitch/config.json",
        vocoder_path="./models/v1/hi/hifigan/best_model.pth",
        vocoder_config_path="./models/v1/hi/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )

    fastpitch = fastpitch_hifigan.synthesizer.tts_model

    #### ENCODER

    logger.info(f"Embedding Architecture: {fastpitch.emb} \n\n")
    if args.export_embedding:
        torch.onnx.export(
            model=model,
            args=(
                inputs,
                # {"speaker_ids": speaker_id, "d_vectors": d_vector}
                speaker_id
                # d_vector,
                # {"is_final": True}
            ),
            f="fastpitch.onnx",
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['x', 'speaker_id'],
            output_names=['model_outputs', 'alignments', 'pitch', 'durations_log'],
            # output_names=['outputs'],
            dynamic_axes={
                'x': {0: 'batch_size', 1: 'T'},
                # 'model_outputs': {0: 'batch_size', 1: 'O_T'},
                # 'alignments': {0: 'batch_size', 1: 'O_T', 2: 'T_'},
                # 'pitch': {0: 'batch_size'},
                # 'durations_log': {0: 'batch_size', 1: 'T'}
            },
            # verbose=True,
        )
        pass
        
    logger.info(f"Encoder Architecture: {fastpitch.encoder} \n\n")
    if args.export_encoder:
        pass

    ### DURATION PREDICTION

    logger.info(f"Duration Predictor Architecture: {fastpitch.duration_predictor} \n\n")
    if args.export_duration_predictor:
        pass

    ### ALIGNER

    logger.info(f"Aligner Architecture: {fastpitch.aligner} \n\n")
    if args.export_aligner:
        pass

    ### PITCH PREDICTOR

    logger.info(f"Pitch Predictor Architecture: {fastpitch.pitch_predictor} \n\n")
    if args.export_pitch_predictor:
        pass

    ### DECODER

    logger.info(f"Positional Encoder Architecture: {fastpitch.pos_encoder} \n\n")
    if args.export_positional_encoder:
        pass

    logger.info(f"Decoder Architecture: {fastpitch.decoder} \n\n")
    if args.export_decoder:
        pass
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    # batch()
    decompose()