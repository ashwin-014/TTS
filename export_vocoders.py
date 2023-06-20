import os
import shutil
import json
import time
import tempfile

import pysbd
import numpy as np
import scipy
from TTS.api import TTS


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


def export_vocoders_for_lang(lang):
    checkpoint_folder = f"/home/app/models/misc/checkpoints/{lang}"
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
        "as": "মোৰ নাম ভাৰত. নমস্কাৰ, তোমাৰ নাম",
        "hi": "मेरा नाम भारत हैं. नमस्ते आपका नाम हैं",
        "bn": "আমার নাম ভারত. হ্যালো, আপনার নাম",
        "gu": "મારું નામ ભારત છે. નમસ્તે, તમારું નામ છે",
        "mr": "माझे नाव भारत आहे. नमस्कार, तुमचे नाव आहे",
        "or": "ମୋ ନାଁ ଭାରତ. ନମସ୍କାର, ଆପଣଙ୍କ ନାମ",
        "pa": "ਮੇਰਾ ਨਾਮ ਭਾਰਤ ਹੈ. ਹੈਲੋ, ਤੁਹਾਡਾ ਨਾਮ ਹੈ",
        "raj": "",
        "ta": "காலை  வணக்கம்‌. யாம் ‌எப்படி இருக்கிறீர்கள்‌?",
        "te": "గుడ్ మార్నింగ్ మీరు ఎలా ఉన్నారు?",
        "kn": "ಗುಡ್ ಮಾರ್ನಿಂಗ್ ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "സുപ്രഭാതം, എങ്ങനെയുണ്ട്?",
        "mni": "ꯒꯨꯗ ꯃꯣꯔꯅꯤꯡ ꯀꯔꯝ ꯇꯧꯒꯦ",
        "brx": "गुड मर्निं, नों बोरै?",
        "en+hi": "Good morning, how are you?"
    }
    input_text = texts[lang]
    seg = pysbd.Segmenter(language="en", clean=True)

    if lang == "mni":

        from ai4bharat.transliteration import XlitEngine
        xlit_langs = set()
        xlit_langs.add(lang)
        xlit_engine = XlitEngine(xlit_langs, beam_width=6)
        input_text = xlit_engine.translit_sentence(input_text, lang)

        # Manipuri was trained using the Central-govt's Bangla script
        # So convert the words in native state-govt script to Eastern-Nagari
        from aksharamukha.transliterate import process as aksharamukha_xlit
        input_text = aksharamukha_xlit("MeeteiMayek", "Bengali", input_text)

    start = time.time()
    sens = seg.segment(input_text)
    print(" > Text splitted to sentences.")
    print(sens)

    wavs, _ = tts.tts(text=sens, speaker=tts.speakers[0])

    os.makedirs(f"/home/app/models/misc/checkpoints/{lang}/outputs", exist_ok=True)
    for j, wav in enumerate(wavs):
        wav = np.array(wav)
        # print(wav, wav.shape)
        # wav = wav[~np.isnan(wav)]
        wav = wav[wav != -1]
        print(wav, wav.shape)
        save_wav(wav=wav, path=f"/home/app/models/misc/checkpoints/{lang}/outputs/output_{j}.wav", sample_rate=22050)
    print(f"done in {time.time() - start}")
    os.makedirs(f"./onnx/{lang}", exist_ok=True)
    shutil.move("./onnx/vocoder.onnx", f"/home/app/models/misc/checkpoints/{lang}/hifigan/vocoder.onnx")
    shutil.move("./onnx/vocoder_fp16.onnx", f"/home/app/models/misc/checkpoints/{lang}/hifigan/vocoder_fp16.onnx")


if __name__ == "__main__":
    langs = [
        # "as",
        # "hi",
        # "bn",
        # "gu",
        # "mr",
        # "or",
        # "pa",
        # "raj",
        # "ta",
        # "kn",
        # "ml",
        # "te",
        # "mni",
        # "brx",
        "en+hi"
    ]
    for lang in langs:
        export_vocoders_for_lang(lang)
