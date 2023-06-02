import os
import time
import logging
from datetime import datetime

import numpy as np
import scipy
import pysbd
from itertools import zip_longest
import torch
from torch import nn
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
# import tensorrt as trt
# from cuda import cuda, cudart, nvrtc

from TTS.api import TTS
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer


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


def format_durations(o_dr_log, x_mask):
    o_dr = (torch.exp(o_dr_log) - 1) * x_mask * length_scale
    o_dr[o_dr < 1] = 1.0
    o_dr = torch.round(o_dr)
    return o_dr

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None) -> None:
    logger.info(f"save wav: {wav.shape}")
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


def generate_attn(dr, x_mask, y_mask=None):
    # compute decode mask from the durations
    if y_mask is None:
        y_lengths = dr.sum(1).long()
        y_lengths[y_lengths < 1] = 1
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
        # models[model_name] = ort.InferenceSession(
        #     onnx_path, providers=["CUDAExecutionProvider"]
        # )
    return models


class tts_model(nn.Module):
    def __init__(self):
        super().__init__()
        fastpitch_hifigan = TTS(
            model_path="./models/v1/hi/fastpitch/best_model.pth",
            config_path="./models/v1/hi/fastpitch/config.json",
            vocoder_path="./models/v1/hi/hifigan/best_model.pth",
            vocoder_config_path="./models/v1/hi/hifigan/config.json",
            progress_bar=True,
            gpu=True,
        )
        self.tokenizer = fastpitch_hifigan.synthesizer.tts_model.tokenizer

        fastpitch = fastpitch_hifigan.synthesizer.tts_model
        self.vocoder = fastpitch_hifigan.synthesizer.vocoder_model
        self.emb = fastpitch.emb
        self.encoder = fastpitch.encoder
        self.duration_predictor = fastpitch.duration_predictor
        # aligner = fastpitch.aligner
        self.pitch_predictor = fastpitch.pitch_predictor
        self.pitch_emb = fastpitch.pitch_emb
        self.pos_encoder = fastpitch.pos_encoder
        self.decoder = fastpitch.decoder

    def forward(self, x, speaker_ids):
        with torch.no_grad():
            x_lengths = torch.tensor(x.shape[1]).repeat(x.shape[0]).to(x.device)
            x_mask_original = sequence_mask(x_lengths, x.shape[1]).to(x.dtype)
            x_mask_original = torch.where(x > 0, x_mask_original, 0)
            x_mask = torch.unsqueeze(x_mask_original, 1).float()

            g = speaker_ids.unsqueeze(-1)
            x_emb = self.emb(x)
            logger.info(f"Embedding Output Shape- x_emb:{x_emb.shape}")

            encoder_input_x = torch.transpose(x_emb, 1, -1)
            logger.info(f"Encoder Input Shapes - encoder_input_x:{encoder_input_x.shape}")
            o_en = self.encoder(encoder_input_x, x_mask)
            logger.info(f"Encoder Output Shapes - o_en:{o_en.shape}")

            o_en = o_en + g

            o_dr_log = self.duration_predictor(o_en, x_mask)
            logger.info(f"Duration Predictor Output Shapes- o_dr_log:{o_dr_log.shape}")

            o_dr = format_durations(o_dr_log, x_mask).squeeze(1)
            o_dr = o_dr * x_mask_original
            y_lengths = o_dr.sum(1)

            o_pitch = self.pitch_predictor(o_en, x_mask)
            logger.info(f"Pitch Predictor Output Shapes- o_pitch:{o_pitch.shape}")

            o_pitch_emb = self.pitch_emb(o_pitch)
            logger.info(f"Pitch Embedding Output Shapes- o_pitch_emb:{o_pitch_emb.shape}")

            o_en = o_en + o_pitch_emb

            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
            o_en_ex, attn = expand_encoder_outputs(o_en, o_dr, x_mask, y_mask)
            logger.info(f"Expand encoder outputs shapes- attn: {attn.shape}")

            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
            logger.info(f"Decoder Positional Encoder Output Shapes- o_en_ex: {o_en_ex.shape}")

            o_de = self.decoder(o_en_ex, y_mask, g=g)
            logger.info(f"Decoder Output Shapes- o_de:{o_de.shape}")

            waveform = self.vocoder.inference(o_de)
            logger.info(f"Vocoder Output Shapes- waveform:{waveform.shape}")
            attn = torch.sum(attn, dim=(-1, -2)).to(torch.int)
            attn = attn - 5
            multiplier = waveform.size(2) / attn.max()
            multiplier = multiplier.to(torch.int)
            attn = attn * multiplier
            return waveform, attn.unsqueeze(1)

    def predict(self, x, speaker_id):
        return self.forward(x, speaker_id)


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor


def id_to_torch(aux_id, cuda=False):
    if aux_id is not None:
        aux_id = np.asarray(aux_id)
        aux_id = torch.from_numpy(aux_id)
    if cuda:
        return aux_id.cuda()
    return aux_id


def export_onnx():
    text = ["नमस्ते आपका नाम क्या है"]
    model = tts_model()

    language_name = "en"
    use_cuda = True
    speaker_id = 0

    st = time.perf_counter()

    speaker_id = id_to_torch(speaker_id, cuda=use_cuda)
    # convert text to sequence of token IDs
    tok_ids = [model.tokenizer.text_to_ids(t, language=language_name) for t in text]
    inputs = np.array(list(zip_longest(*tok_ids, fillvalue=0)), dtype=np.int32).T
    inputs = numpy_to_torch(inputs, torch.long, cuda=use_cuda)

    inputs = inputs.to("cuda")
    speaker_id = torch.tensor(speaker_id).to("cuda")
    logger.info(f"inputs: {inputs}, speaker id: {speaker_id}")
    waveform, attn = model.forward(inputs)
    logger.info(f"waveform: {waveform.shape}, attn: {attn.shape}")

    st = time.perf_counter()
    waveform = waveform.cpu().numpy()
    logger.info(f"waveform: {waveform.shape}")

    st = time.perf_counter()
    wavs = []
    for i, wave in enumerate(waveform):
        # logger.info(f"wave: {wave.shape}, attn: {attn.shape}, {wave.squeeze().shape}")
        wave = wave.squeeze()[:attn[i]]
        wavs += list(wave)
        wavs += [0] * 10000

    wav = np.array(wavs)
    logger.info(f"wav: {wav}, wav.shape: {wav.shape}")
    save_wav(wav=wav, path="output.wav", sample_rate=22050)

    torch.onnx.export(
            model=model,
            args=(inputs, speaker_id),
            f="full_tts_model.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['inputs', 'speaker_id'],
            output_names=['waveform', 'durations'],
            dynamic_axes={
                'inputs': {0: 'batch_size', 1: 'T'},
                'waveform': {0: 'batch_size', 2: 'O_T'},
                'durations': {0: 'batch_size'},
            },
            verbose=True,
        )
    print("exported!! ")


def export_onnx_fp16():
    onnx_model = onnx.load("full_tts_model.onnx")
    onnx.checker.check_model(onnx_model)
    model_fp16 = float16.convert_float_to_float16(onnx_model)
    onnx.save(model_fp16, "full_tts_model_fp16.onnx")


def test_onnx():
    DEVICE_NAME = "cuda"
    DEVICE_NAME = "cpu"
    DEVICE_INDEX = 0
    sess_options = ort.SessionOptions()
    # sess_options.enable_profiling=True
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # exproviders = ['CPUExecutionProvider']
    exproviders = [('CUDAExecutionProvider', {'cudnn_conv_use_max_workspace': 1}), 'CPUExecutionProvider']

    tts_ort_sess = ort.InferenceSession(
        "full_tts_model.onnx", sess_options, providers=exproviders
    )
    onnx.checker.check_model(onnx_model)
    print(onnx_model.graph.input, onnx_model.graph.output)

    text="मेरा. नमस्ते. नाम भारत हैं. नमस्ते आपका नाम क्या है नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."

    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(sens)

    config = load_config("./models/v1/hi/fastpitch/config.json")
    tokenizer, _ = TTSTokenizer.init_from_config(config)

    tok_ids = [tokenizer.text_to_ids(s, language=None) for s in sens]
    inputs = np.array(list(zip_longest(*tok_ids, fillvalue=0)), dtype=np.int64).T
    speaker_id = np.asarray(0, dtype=np.int64)
    print("Initialised inputs...")

    inputs_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        inputs, DEVICE_NAME, DEVICE_INDEX
    )
    speaker_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        speaker_id, DEVICE_NAME, DEVICE_INDEX
    )
    tts_io_binding = tts_ort_sess.io_binding()
    tts_io_binding.bind_input(
        name="inputs",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=inputs_ortvalue.shape(),
        buffer_ptr=inputs_ortvalue.data_ptr(),
    )
    tts_io_binding.bind_input(
        name="speaker_id",
        device_type=speaker_ortvalue.device_name(),
        device_id=0,
        element_type=speaker_id.dtype,
        shape=speaker_ortvalue.shape(),
        buffer_ptr=speaker_ortvalue.data_ptr(),
    )

    tts_io_binding.bind_output(
        name="waveform",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 817, 80),
    )
    tts_io_binding.bind_output(
        name="durations",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 817, inputs_ortvalue.shape()[1]),
    )

    overall_start = time.perf_counter()
    for i in range(100):
        start = time.perf_counter()
        tts_ort_sess.run_with_iobinding(tts_io_binding)
        outputs_ = tts_io_binding.get_outputs()
        waveform = outputs_[0].numpy()
        attn = outputs_[1].numpy()

        # outputs_ = tts_ort_sess.run(
        #     ["waveform", "durations"],
        #     {"inputs": inputs, "speaker_id": speaker_id},
        # )
        # waveform = outputs_[0]
        # attn = outputs_[1]

        wavs = []
        for i, wave in enumerate(waveform):
            # logger.info(f"wave: {wave.shape}, attn: {attn.shape}, {wave.squeeze().shape}")
            wave = wave.squeeze()[:attn[i]]
            wavs += list(wave)
            wavs += [0] * 10000

        process_time = time.perf_counter() - start
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / (waveform.shape[0] * waveform.shape[-1] / 22050)}")

    process_time = time.perf_counter() - overall_start
    print(f" > Processing time: {process_time}")

    wav = np.array(wavs)
    logger.info(f"wav: {wav}, wav.shape: {wav.shape}")
    save_wav(wav=wav, path="output.wav", sample_rate=22050)


if __name__ == "__main__":
    export_onnx()
    export_onnx_fp16()
    test_onnx()
