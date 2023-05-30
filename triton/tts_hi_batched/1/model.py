# import torch
from itertools import zip_longest

import pysbd
import onnx
import onnxruntime as ort
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class TritonPythonModel:

    def initialize(self, args):
        """
        This function initializes pre-trained ResNet50 model.
        """
        self.device = 'cuda' if args["model_instance_kind"] == "GPU" else 'cpu'
        self.model = TTS(
            model_path="./models/v1/hi/fastpitch/best_model.pth",
            config_path="./models/v1/hi/fastpitch/config.json",
            vocoder_path="./models/v1/hi/hifigan/best_model.pth",
            vocoder_config_path="./models/v1/hi/hifigan/config.json",
            progress_bar=True,
            gpu=True if args["model_instance_kind"] == "GPU" else False,
        )
        self.seg = pysbd.Segmenter(language="en", clean=True)

    def half_batch_execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            inputs = self.seg.segment(input_tensor.as_numpy().tolist()[0][0].decode("utf-8"))
            result = self.model.tts(text=inputs, speaker=self.model.speakers[0])
            out_tensor = pb_utils.Tensor("OUTPUT0", np.array(result).astype(np.float32))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
    
    def execute(self, requests):  # full_batch_
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        inputs = []
        input_lens = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            sents = self.seg.segment(input_tensor.as_numpy().tolist()[0][0].decode("utf-8"))
            input_lens.append(len(sents))
            inputs.extend(sents)

        result = self.model.tts(text=inputs, speaker=self.model.speakers[0])
        final_results = []
        prev_ip_len = 0
        for ip_len in input_lens:
            final_results.append(result[prev_ip_len:ip_len])
            prev_ip_len = ip_len

        out_tensor = pb_utils.Tensor("OUTPUT0", np.array(final_results).astype(np.float32))
        responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
    
    def old_execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            inputs = input_tensor.as_numpy().tolist()[0][0].decode("utf-8")
            result = self.model.tts(text=inputs, speaker=self.model.speakers[0])
            out_tensor = pb_utils.Tensor("OUTPUT0", np.array(result).astype(np.float32))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def onnx_execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        inputs = []
        input_lens = []
        speaker_ids = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            sents = self.seg.segment(input_tensor.as_numpy().tolist()[0][0].decode("utf-8"))
            input_lens.append(len(sents))

            tok_ids = [self.tokenizer.text_to_ids(s, language=None) for s in sents]
            inps = list(zip_longest(*tok_ids, fillvalue=0))
            speaker_ids.append(np.asarray(0, dtype=np.int64))
            inputs.extend(inps)
            # print("Initialised inputs...")

        inputs = np.array(inputs, dtype=np.int64).T
        speaker_ids = np.array(speaker_ids, dtype=np.int64)
        outputs_ = self.ort_session.run(
            # ["fastpitch/alignments", "fastpitch/pitch", "fastpitch/durations_log", "fastpitch/decoder_output", "hifigan/o"],
            ["hifigan/o"],
            {"fastpitch/x": inputs, "fastpitch/speaker_id": speaker_ids},
        )
        outputs = np.squeeze(outputs_[0], axis=1)
        final_results = []
        for ip_len in input_lens:
            final_results.append(outputs[:ip_len])
            del outputs_[:ip_len]

        out_tensor = pb_utils.Tensor("OUTPUT0", np.array(final_results).astype(np.float32))
        responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses