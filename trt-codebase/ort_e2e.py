import time
from itertools import zip_longest
import torch
import pysbd
import onnx
import onnxruntime as ort
import numpy as np
import scipy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
# from typing import List, Any, Dict


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

    text = "मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है नमस्ते आपका नाम क्या है नाम भारत हैं नाम हैं भारत नाम भारत हैं नाम भारत हैं नाम भारत हैं नाम हैं भारत नाम भारत हैं नाम भारत हैं"
    # text="मेरा. नमस्ते आपका नाम क्या है. नाम भारत हैं नाम हैं भारत नाम भारत हैं नाम भारत हैं \
    #     नाम भारत हैं नाम भारत हैं नाम भारत हैं नाम भारत हैं नाम भारत हैं नाम भारत हैं नाम भारत हैं. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # from TTS.utils.synthesizer import Synthesizer
    # s = Synthesizer()
    # sens = s.split_into_sentences(text)

    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(" > Text splitted to sentences.")
    print(sens)
    # for i in range(100):
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


def ort_exec(
    inputs: np.array,
    speaker_id: np.array,
    fastpitch_ort_sess: ort.InferenceSession,
    hifigan_ort_sess: ort.InferenceSession,
) -> list:
    """
    Executes ORT code
    """
    print(inputs.shape)
    print(inputs)
    print(speaker_id)
    start = time.time()
    # Fastpitch
    outputs_ = fastpitch_ort_sess.run(
        # ["model_outputs", "alignments", "pitch", "durations_log"],
        ['model_outputs', 'alignments', 'pitch',
        'durations_log', "x_lengths", "y_lengths",
        "o_en", "x_emb", "o_dr", "o_pitch_emb",
        "o_en_ex_in", "o_en_ex_out", "y_mask", 
        "y_mask_original", "x_mask", "x_mask_original"],
        {"x": inputs, "speaker_id": speaker_id},
    )
    
    # model_outputs_ = torch.Tensor(outputs_[0])
    # attn = torch.Tensor(outputs_[1])
    model_outputs_ = outputs_[0]
    print(model_outputs_.shape)
    attn = outputs_[1]
    print(attn.shape)
    # print("--> ", model_outputs_.shape, attn.shape)

    # Vocoder
    # vocoder_inputs = model_outputs_.transpose(1, 2)
    # waveform = hifigan_ort_sess.run(None, {"c": vocoder_inputs.cpu().numpy()})
    # waveform = torch.Tensor(waveform).squeeze(0)
    # vocoder_inputs = np.transpose(model_outputs_, (0, 2, 1))
    # print(vocoder_inputs.shape)
    # print(vocoder_inputs)
    # waveform = hifigan_ort_sess.run(["o", "o_shape"], {"c": vocoder_inputs})
    waveform = hifigan_ort_sess.run(["o", "o_shape"], {"c": model_outputs_})
    print(len(waveform[0]))
    waveform = np.squeeze(waveform[0], axis=0)
    print("squeezed: ", waveform.shape)

    # attn = torch.sum(attn, dim=(-1, -2))
    attn = attn.sum(axis=(-1, -2))
    print(attn.shape)
    # TODO: get a permanent solution ta a mask level to silence 5 mel frames
    attn = (attn - 5) * 256
    # attn = attn.to(torch.int)
    attn = attn.astype(np.int32)

    # waveform = waveform.cpu().numpy()
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


def onnx_batch():
    """
    Execution code for onnx models
    """
    # Model init
    # --------------------------------------
    # onnx_model = onnx.load("models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_hardcoded_long.onnx")
    # onnx_model = onnx.load("models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_hardcoded.onnx")
    onnx_model = onnx.load("models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_dynamic.onnx")
    onnx.checker.check_model(onnx_model)
    fastpitch_ort_sess = ort.InferenceSession(
        # "models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_hardcoded_long.onnx", providers=["CUDAExecutionProvider"]
        # "models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_hardcoded.onnx", providers=["CUDAExecutionProvider"]
        "models/v1/hi/final_unsqueeze_notranspose/fastpitch_unsqueeze_all_outputs_dynamic.onnx", providers=["CUDAExecutionProvider"]
    )
    print("Initialised fastpitch...")

    # onnx_model = onnx.load("nosqueeze/vocoder.onnx")
    onnx_model = onnx.load("models/v1/hi/final_unsqueeze_notranspose/vocoder_with_shape.onnx")
    onnx.checker.check_model(onnx_model)
    # print(onnx_model.graph.input, onnx_model.graph.output)
    hifigan_ort_sess = ort.InferenceSession(
        # "onnx/vocoder.onnx", providers=["CUDAExecutionProvider"]
        "models/v1/hi/final_unsqueeze_notranspose/vocoder_with_shape.onnx", providers=["CUDAExecutionProvider"]
    )
    print("Initialised hifigan...")
    # --------------------------------------

    # Text init
    # --------------------------------------
    # text="मेरा. नमस्ते. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
    #     नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    # text="नमस्ते आपका नाम क्या है आपका नाम क्या है आपका नाम क्या है."
    text="नमस्ते"    

    # text = "मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है"
    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(" > Text splitted to sentences.")
    print(sens)
    # --------------------------------------

    # ORT Inputs init
    # --------------------------------------
    from TTS.config import load_config
    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    config = load_config("./models/v1/hi/fastpitch/config.json")
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    tok_ids = [tokenizer.text_to_ids(s, language=None) for s in sens]
    inputs = np.array(list(zip_longest(*tok_ids, fillvalue=0)), dtype=np.int64).T
    speaker_id = np.asarray(0, dtype=np.int64)
    print("Initialised inputs...")
    # --------------------------------------

    start = time.time()
    for i in range(1):
        wav = ort_exec(inputs, speaker_id, fastpitch_ort_sess, hifigan_ort_sess)
    print(f"done in {time.time() - start}")

    wav = np.array(wav)
    save_wav(wav=wav, path="models/v1/hi/final_unsqueeze_notranspose/onnx_output_h.wav", sample_rate=22050)


if __name__ == "__main__":
    # batch()
    onnx_batch()
