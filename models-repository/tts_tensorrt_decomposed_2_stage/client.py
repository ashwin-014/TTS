import os
import numpy as np
from glob import glob
import math
import tritonclient.http as http_client
import sys
import time
import gevent.ssl
import random
import scipy
from tritonclient.utils import serialize_byte_tensor

def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None) -> None:
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))

triton_http_client = http_client.InferenceServerClient(
    url="localhost:8000",
    connection_timeout=180.0,
    network_timeout=180.0
)
headers = {}
results = []
start_time = time.time()
for i in range(1):
    serialized = np.array([
        # "नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है.".encode("utf-8"), 
        # "नाम क्या है. नाम क्या है. नाम क्या है.".encode("utf-8")
        "नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम ".encode("utf-8"),
    ], dtype=np.object_)
    serialized = np.expand_dims(serialized, axis=0)
    input0 = http_client.InferInput("TEXT", serialized.shape, "BYTES")
    input0.set_data_from_numpy(serialized, binary_data=True)
    output0 = http_client.InferRequestedOutput('WAVEFORM')
    response = triton_http_client.infer("tts_tensorrt_decomposed_2_stage", model_version='1',inputs=[input0], request_id=str(1), outputs=[output0], headers=headers)
    result_response = response.get_response()
    batch_result = response.as_numpy("WAVEFORM")
    len_result = response.as_numpy("WAVEFORM_LENGTH")

    total_time = time.time() - start_time

    print(total_time)

    for i in range(batch_result.shape[0]):
        save_wav(wav=batch_result[i], path=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"triton_tensorrt_decomposed_output_{i}.wav"), sample_rate=22050)



