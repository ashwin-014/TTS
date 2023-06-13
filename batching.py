import time
from itertools import zip_longest

# import torch
import pysbd
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
import numpy as np
import scipy

from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer

DEVICE_NAME = "cuda"
DEVICE_INDEX = 0


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
    start = time.time()
    # Fastpitch
    outputs_ = fastpitch_ort_sess.run(
        ["model_outputs", "alignments", "pitch", "durations_log"],
        {"x": inputs, "speaker_id": speaker_id},
    )
    model_outputs_ = outputs_[0]
    attn = outputs_[1]
    print(model_outputs_.shape, attn.shape, outputs_[2].shape, outputs_[3].shape)

    # Vocoder
    vocoder_inputs = np.transpose(model_outputs_, (0, 2, 1))
    waveform = hifigan_ort_sess.run(None, {"c": vocoder_inputs})[0]
    attn = attn.sum(axis=(-1, -2))
    # TODO: get a permanent solution at a mask level to silence 5 mel frames
    attn = (attn - 5) * 256
    attn = attn.astype(np.int32)

    wavs = []
    # print(waveform.shape)
    for i, wave in enumerate(waveform):
        wave = wave.squeeze()[: attn[i]]
        wave = wave.squeeze()
        wavs += list(wave)
        wavs += [0] * 10000

    process_time = time.time() - start
    audio_time = len(wavs) / 22050
    print(f" > Processing time: {process_time}")
    print(f" > Real-time factor: {process_time / (waveform.shape[0] * waveform.shape[-1] / 22050)}")
    return wavs


def ort_exec_io_bound(
    fastpitch_ort_sess: ort.InferenceSession,
    fastpitch_io_binding,
    hifigan_ort_sess: ort.InferenceSession,
    hifigan_io_binding,
) -> list:
    """
    Executes ORT code
    """

    start = time.time()
    # Fastpitch
    fastpitch_ort_sess.run_with_iobinding(fastpitch_io_binding)
    outputs_ = fastpitch_io_binding.get_outputs()

    model_outputs_ = outputs_[0].numpy()
    attn = outputs_[1].numpy()

    # Vocoder
    vocoder_inputs = np.transpose(model_outputs_, (0, 2, 1))
    vocoder_inputs_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        vocoder_inputs, DEVICE_NAME, DEVICE_INDEX
    )
    hifigan_io_binding.bind_input(
        name="c",
        device_type=vocoder_inputs_ortvalue.device_name(),
        device_id=0,
        element_type=vocoder_inputs.dtype,
        shape=vocoder_inputs_ortvalue.shape(),
        buffer_ptr=vocoder_inputs_ortvalue.data_ptr(),
    )
    hifigan_io_binding.bind_output(
        name="o",
        device_type=vocoder_inputs_ortvalue.device_name(),
        device_id=0,
        element_type=vocoder_inputs.dtype,
        shape=(vocoder_inputs_ortvalue.shape()[0], 1, 209152),
    )
    hifigan_ort_sess.run_with_iobinding(hifigan_io_binding)
    waveform = hifigan_io_binding.get_outputs()[0].numpy()
    # waveform = np.squeeze(waveform, axis=0)

    attn = attn.sum(axis=(-1, -2))
    # TODO: get a permanent solution at a mask level to silence 5 mel frames
    attn = (attn - 5) * 256
    attn = attn.astype(np.int32)

    wavs = []
    for i, wave in enumerate(waveform):
        wave = wave.squeeze()[: attn[i]]
        wave = wave.squeeze()
        wavs += list(wave)
        wavs += [0] * 10000

    process_time = time.time() - start
    audio_time = len(wavs) / 22050
    print(f" > Processing time: {process_time}")
    print(f" > Real-time factor: {process_time / (waveform.shape[0] * waveform.shape[-1] / 22050)}")
    return wavs


def onnx_batch():
    """
    Execution code for onnx models
    """
    onnx_model = onnx.load("fastpitch_fp16.onnx")
    onnx.checker.check_model(onnx_model)
    # model_fp16 = float16.convert_float_to_float16(onnx_model)
    # onnx.save(model_fp16, "fastpitch_fp16.onnx")
    fastpitch_ort_sess = ort.InferenceSession(
        "fastpitch_fp16.onnx", sess_options, providers=exproviders
    )
    print("Initialised fastpitch...")

    onnx_model = onnx.load("vocoder_fp16.onnx")
    onnx.checker.check_model(onnx_model)
    # model_fp16 = float16.convert_float_to_float16(onnx_model)
    # onnx.save(model_fp16, "vocoder_fp16.onnx")
    hifigan_ort_sess = ort.InferenceSession(
        "vocoder_fp16.onnx", sess_options, providers=exproviders
    )
    print("Initialised hifigan...")
    # --------------------------------------

    # Text init
    # --------------------------------------
    # text = "ओम नम शिवाय ओम. ओम नम शिवाय"
    text="मेरा. नाम भारत हैं. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
                        नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है.\
                         नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
                         नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. \
                         नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है."
    seg = pysbd.Segmenter(language="en", clean=True)
    sens = seg.segment(text)
    print(" > Text splitted to sentences.")
    print(sens)
    # --------------------------------------

    # ORT Inputs init
    # --------------------------------------
    config = load_config("./models/v1/hi/fastpitch/config.json")
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    tok_ids = [tokenizer.text_to_ids(s, language=None) for s in sens]
    inputs = np.array(list(zip_longest(*tok_ids, fillvalue=0)), dtype=np.int64).T
    speaker_id = np.asarray(0, dtype=np.int64)
    print("Initialised inputs...")
    # --------------------------------------

    inputs_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        inputs, DEVICE_NAME, DEVICE_INDEX
    )
    speaker_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        speaker_id, DEVICE_NAME, DEVICE_INDEX
    )
    fastpitch_io_binding = fastpitch_ort_sess.io_binding()
    fastpitch_io_binding.bind_input(
        name="x",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=inputs_ortvalue.shape(),
        buffer_ptr=inputs_ortvalue.data_ptr(),
    )
    fastpitch_io_binding.bind_input(
        name="speaker_id",
        device_type=speaker_ortvalue.device_name(),
        device_id=0,
        element_type=speaker_id.dtype,
        shape=speaker_ortvalue.shape(),
        buffer_ptr=speaker_ortvalue.data_ptr(),
    )

    fastpitch_io_binding.bind_output(
        name="model_outputs",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 817, 80),
    )
    fastpitch_io_binding.bind_output(
        name="alignments",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 817, inputs_ortvalue.shape()[1]),
    )
    fastpitch_io_binding.bind_output(
        name="pitch",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 1, inputs_ortvalue.shape()[1]),
    )
    fastpitch_io_binding.bind_output(
        name="durations_log",
        device_type=inputs_ortvalue.device_name(),
        device_id=0,
        element_type=inputs.dtype,
        shape=(inputs_ortvalue.shape()[0], 1, inputs_ortvalue.shape()[1]),
    )

    hifigan_io_binding = hifigan_ort_sess.io_binding()

    start = time.time()
    for i in range(100):
        # wav = ort_exec(inputs, speaker_id, fastpitch_ort_sess, hifigan_ort_sess)
        wav = ort_exec_io_bound(fastpitch_ort_sess, fastpitch_io_binding, hifigan_ort_sess, hifigan_io_binding)
    print(f"done in {time.time() - start}")

    wav = np.array(wav)
    save_wav(wav=wav, path="output.wav", sample_rate=22050)


def export_vocoders_for_lang(lang):

    import os
    import json
    import tempfile

    checkpoint_folder = f"~/tts/deployment/models/indo_aryan/{lang}"
    tts_config_path = os.path.join(checkpoint_folder, "fastpitch/config.json")
    tts_config = json.load(open(tts_config_path))
    speakers_file = tts_config_path.replace("config.json", "speakers.pth")
    tts_config["model_args"]["speakers_file"] = speakers_file
    tts_config["speakers_file"] = speakers_file

    # Write the config file to a temporary path so that we can pass it to the Synthesizer class
    patched_tts_config_file = tempfile.NamedTemporaryFile(suffix=".json", mode='w', encoding='utf-8', delete=False)
    patched_tts_config_file.write(json.dumps(tts_config))
    patched_tts_config_file.close()

    tts = TTS(
        model_path=f"{checkpoint_folder}/fastpitch/best_model.pth",
        config_path=patched_tts_config_file.name,
        vocoder_path=f"{checkpoint_folder}/hifigan/best_model.pth",
        vocoder_config_path=f"{checkpoint_folder}/hifigan/config.json",
        progress_bar=True,
        gpu=True,
    )
    texts = {
        "as": "",
        "hi": "मेरा. नाम भारत हैं. नमस्ते आपका नाम हैं",
        "bn": "",
        "gu": "",
        "mr": "",
        "or": "",
        "pa": "",
        "raj": "",
    }
    seg = pysbd.Segmenter(language="en", clean=True)

    start = time.time()
    sens = seg.segment(texts[lang])
    print(" > Text splitted to sentences.")
    print(sens)

    wavs, _ = tts.tts(text=sens, speaker=tts.speakers[0])
    for j, wav in enumerate(wavs):
        wav = np.array(wav)
        # print(wav, wav.shape)
        # wav = wav[~np.isnan(wav)]
        wav = wav[wav != -1]
        print(wav, wav.shape)
        save_wav(wav=wav, path=f"outputs/output_{lang}_{j}.wav", sample_rate=22050)
    print(f"done in {time.time() - start}")


if __name__ == "__main__":
    # batch()
    # onnx_batch()
    # onnx_model = onnx.load("../triton/tts_hi_batched/1/models/hififastpitch.onnx")
    # onnx.checker.check_model(onnx_model)
    # print(onnx_model.graph.input, onnx_model.graph.output)

    langs = [
        "as",
        "hi",
        "bn",
        "gu",
        "mr",
        "or",
        "pa",
        "raj",
    ]
    for lang in langs:
        export_vocoders_for_lang(lang)
