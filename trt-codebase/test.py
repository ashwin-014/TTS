import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TTS.api import TTS
import time
from itertools import zip_longest
import torch
import pysbd
import onnx
import onnxruntime as ort
import numpy as np
import scipy

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
# text = "मेरा."
# text="मेरा. नमस्ते आपका नाम क्या है."
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

# from onnxsim import simplify
# import onnx

# model = onnx.load("./fastpitch.onnx")

# model_simp, check = simplify(model)

# onnx.save(model, "./fastpitch-simplified.onnx")