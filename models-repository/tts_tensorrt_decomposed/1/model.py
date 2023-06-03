# import torch
import os
import time
from itertools import zip_longest
from datetime import datetime
import tensorrt as trt
from cuda import cuda, cudart, nvrtc
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
from TTS.api import TTS
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.helpers import average_over_durations, generate_path, maximum_path, sequence_mask
import torch
from config import TRT_MODELS_DIM_SPECS
import logging
import json

now = datetime.now()

length_scale = 1

class TritonPythonModel:

    def initialize(self, args):
        
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get WAVEFORM configuration
        waveform_config = pb_utils.get_output_config_by_name(
            model_config, "WAVEFORM")

        # Convert Triton types to numpy types
        self.waveform_dtype = pb_utils.triton_string_to_numpy(
            waveform_config['data_type'])

        # Get WAVEFORM configuration
        waveform_length_config = pb_utils.get_output_config_by_name(
            model_config, "WAVEFORM_LENGTH")

        # Convert Triton types to numpy types
        self.waveform_length_dtype = pb_utils.triton_string_to_numpy(
            waveform_length_config['data_type'])

        trt_model_dir: str = os.path.join(args["model_repository"], args["model_version"], "models")
        trt_model_paths = {
            "embedding": os.path.join(trt_model_dir, "fastpitch_embedding.engine"),
            "encoder": os.path.join(trt_model_dir, "fastpitch_encoder.engine"),
            "duration_predictor": os.path.join(trt_model_dir, "fastpitch_duration_predictor.engine"),
            "pitch_predictor": os.path.join(trt_model_dir, "fastpitch_pitch_predictor.engine"),
            "pitch_embedding": os.path.join(trt_model_dir, "fastpitch_pitch_embedding.engine"),
            "positional_encoder": os.path.join(trt_model_dir, "fastpitch_positional_encoder.engine"),
            "decoder": os.path.join(trt_model_dir, "fastpitch_decoder.engine"),
            "vocoder": os.path.join(trt_model_dir, "vocoder.engine"),
        }
        self.models = self.load_trt_models(trt_model_paths, TRT_MODELS_DIM_SPECS)
        config = load_config(os.path.join(trt_model_dir, "fastpitch_config.json"))
        self.tokenizer, _ = TTSTokenizer.init_from_config(config)

        # WARMUP - pretty dumb way but, I was not able to use triton warmup due to some weird data error
        self.warmup()

    def warmup(self):
        warmup_batch = [
            "नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है.",
        ]

        tok_ids = [self.tokenizer.text_to_ids(s, language=None) for s in warmup_batch]
        inps = list(zip_longest(*tok_ids, fillvalue=0))
        inputs = np.array(inps, dtype=np.int64).T
        speaker_ids = [np.asarray(0, dtype=np.int64)]
        speaker_ids = np.array(speaker_ids, dtype=np.int64)
        for i in range(5):
            _ = self.run_trt(x=inputs, speaker=speaker_ids) 
            print(f"Warmup Epoch {i} done")

    def execute(self, requests):  # full_batch_
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        responses = []
        inputs = []
        input_lens = []
        speaker_ids = []
        tok_ids = []
        
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            inputs = input_tensor.as_numpy().tolist()[0]
            print(inputs)
            input_lens += [len(inputs)]
            sents = [inputs[i].decode("utf-8") for i in range(len(inputs))]
            print(sents)
            tok_ids.extend([self.tokenizer.text_to_ids(s, language=None) for s in sents])

        inps = list(zip_longest(*tok_ids, fillvalue=0))
        inputs = np.array(inps, dtype=np.int64).T

        speaker_ids.append(np.asarray(0, dtype=np.int64))
        speaker_ids = np.array(speaker_ids, dtype=np.int64)

        results = self.run_trt(x=inputs, speaker=speaker_ids)

        for ip_len in input_lens:
            padded_out0 = list(zip_longest(*results[:ip_len], fillvalue=0))
            out0_tensor = pb_utils.Tensor("WAVEFORM", np.array(padded_out0).T.astype(self.waveform_dtype))
            out1_tensor = [len(ip) for ip in results[:ip_len]]
            out1_tensor = pb_utils.Tensor("WAVEFORM_LENGTH", np.array(out1_tensor).astype(self.waveform_length_dtype))
            responses.append(pb_utils.InferenceResponse([out0_tensor, out1_tensor]))
            del results[:ip_len]

        return responses
    
    def run_trt(self, x, speaker):
        wavs = []
            
        x = torch.tensor(x).to("cuda")
        speaker_ids = torch.tensor(speaker).to("cuda")

        bs_to_use = x.shape[0]

        #------------------------------------------ FORWARD PASS -------------------------------------------#

        aux_input = {"d_vectors": None, "speaker_ids": speaker_ids}
        g = self._set_speaker_input(aux_input)

        # ----- Batching support -----

        x_lengths = torch.tensor(x.shape[1]).repeat(x.shape[0]).to(x.device)
        x_mask_original = sequence_mask(x_lengths, x.shape[1]).to(x.dtype)
        x_mask_original = torch.where(x > 0, x_mask_original, 0)
        x_mask = x_mask_original.float()


        # ----- ENCODER - START

        if g is not None:
            g = g.unsqueeze(-1)

        self.models["embedding"]["io"]["inputs"]["x"][:np.prod([bs_to_use, x.shape[1]])] = x.flatten()[:]
        self.models["embedding"]["context"].set_binding_shape(self.models["embedding"]["engine"].get_binding_index("x"), x.shape)
        assert self.models["embedding"]["context"].all_binding_shapes_specified
        self.models["embedding"]["context"].execute_v2(bindings=self.models["embedding"]["io"]["bindings"])
        x_emb = torch.reshape(self.models["embedding"]["io"]["outputs"]["x_emb"][:np.prod([bs_to_use, x.shape[1], 512])],
                            [bs_to_use, x.shape[1], 512])
        x_emb = torch.transpose(x_emb, 1, -1)


        self.models["encoder"]["io"]["inputs"]["encoder_input_x"][:np.prod([bs_to_use, x.shape[1], 512])] = x_emb.flatten()[:]
        self.models["encoder"]["context"].set_binding_shape(self.models["encoder"]["engine"].get_binding_index("encoder_input_x"), x_emb.shape)
        self.models["encoder"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        self.models["encoder"]["context"].set_binding_shape(self.models["encoder"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert self.models["encoder"]["context"].all_binding_shapes_specified
        self.models["encoder"]["context"].execute_v2(bindings=self.models["encoder"]["io"]["bindings"])
        o_en = torch.reshape(self.models["encoder"]["io"]["outputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])],
                            [bs_to_use, 512, x.shape[1]])
        if g is not None:
            o_en = o_en + g

        # ----- ENCODER - END



        # ----- DURATION PREDICTION - START

        self.models["duration_predictor"]["io"]["inputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])] = o_en.flatten()[:]
        self.models["duration_predictor"]["context"].set_binding_shape(self.models["duration_predictor"]["engine"].get_binding_index("o_en"), o_en.shape)
        self.models["duration_predictor"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        self.models["duration_predictor"]["context"].set_binding_shape(self.models["duration_predictor"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert self.models["duration_predictor"]["context"].all_binding_shapes_specified
        self.models["duration_predictor"]["context"].execute_v2(bindings=self.models["duration_predictor"]["io"]["bindings"])
        o_dr_log = torch.reshape(self.models["duration_predictor"]["io"]["outputs"]["o_dr_log"][:np.prod([bs_to_use, 1, x.shape[1]])],
                                [bs_to_use, 1, x.shape[1]])

        o_dr = self.format_durations(o_dr_log, torch.unsqueeze(x_mask, 1)).squeeze(1)    
        o_dr = o_dr * x_mask_original
        y_lengths = o_dr.sum(1)

        # ----- DURATION PREDICTION - END



        # ----- PITCH PREDICTOR - START

        self.models["pitch_predictor"]["io"]["inputs"]["o_en"][:np.prod([bs_to_use, 512, x.shape[1]])] = o_en.flatten()[:]
        self.models["pitch_predictor"]["context"].set_binding_shape(self.models["pitch_predictor"]["engine"].get_binding_index("o_en"), o_en.shape)
        self.models["pitch_predictor"]["io"]["inputs"]["x_mask"][:np.prod([bs_to_use, x.shape[1]])] = x_mask.flatten()[:]
        self.models["pitch_predictor"]["context"].set_binding_shape(self.models["pitch_predictor"]["engine"].get_binding_index("x_mask"), x_mask.shape)
        assert self.models["pitch_predictor"]["context"].all_binding_shapes_specified
        self.models["pitch_predictor"]["context"].execute_v2(bindings=self.models["pitch_predictor"]["io"]["bindings"])
        o_pitch = torch.reshape(self.models["pitch_predictor"]["io"]["outputs"]["o_pitch"][:np.prod([bs_to_use, 1, x.shape[1]])],
                                [bs_to_use, 1, x.shape[1]])

        self.models["pitch_embedding"]["io"]["inputs"]["o_pitch"][:np.prod([bs_to_use, 1, x.shape[1]])] = o_pitch.flatten()[:]
        self.models["pitch_embedding"]["context"].set_binding_shape(self.models["pitch_embedding"]["engine"].get_binding_index("o_pitch"), o_pitch.shape)
        assert self.models["pitch_embedding"]["context"].all_binding_shapes_specified
        self.models["pitch_embedding"]["context"].execute_v2(bindings=self.models["pitch_embedding"]["io"]["bindings"])
        o_pitch_emb = torch.reshape(self.models["pitch_embedding"]["io"]["outputs"]["o_pitch_emb"][:np.prod([bs_to_use, 512, x.shape[1]])],
                                    [bs_to_use, 512, x.shape[1]])
        o_en = o_en + o_pitch_emb
        
        # ----- PITCH PREDICTOR - END

        o_energy = None                     

        # ----- DECODER - START

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        o_en_ex, attn = self.expand_encoder_outputs(o_en, o_dr, torch.unsqueeze(x_mask, 1), y_mask)
        y_length_max = int(torch.max(y_lengths).cpu().numpy().tolist())

        self.models["positional_encoder"]["io"]["inputs"]["o_en_ex_in"][:np.prod([bs_to_use, 512, y_length_max])] = o_en_ex.flatten()[:]
        self.models["positional_encoder"]["context"].set_binding_shape(self.models["positional_encoder"]["engine"].get_binding_index("o_en_ex_in"), o_en_ex.shape)
        self.models["positional_encoder"]["io"]["inputs"]["y_mask"][:np.prod([bs_to_use, 1, y_length_max])] = y_mask.flatten()[:]
        self.models["positional_encoder"]["context"].set_binding_shape(self.models["positional_encoder"]["engine"].get_binding_index("y_mask"), y_mask.shape)
        assert self.models["positional_encoder"]["context"].all_binding_shapes_specified
        self.models["positional_encoder"]["context"].execute_v2(bindings=self.models["positional_encoder"]["io"]["bindings"])
        o_en_ex_out = torch.reshape(self.models["positional_encoder"]["io"]["outputs"]["o_en_ex_out"][:np.prod([bs_to_use, 512, y_length_max])],
                                    [bs_to_use, 512, y_length_max])

        self.models["decoder"]["io"]["inputs"]["o_en_ex"][:np.prod([bs_to_use, 512, y_length_max])] = o_en_ex_out.flatten()[:]
        self.models["decoder"]["context"].set_binding_shape(self.models["decoder"]["engine"].get_binding_index("o_en_ex"), o_en_ex_out.shape)
        self.models["decoder"]["io"]["inputs"]["y_mask"][:np.prod([bs_to_use, 1, y_length_max])] = y_mask.flatten()[:]
        self.models["decoder"]["context"].set_binding_shape(self.models["decoder"]["engine"].get_binding_index("y_mask"), y_mask.shape)
        assert self.models["decoder"]["context"].all_binding_shapes_specified
        self.models["decoder"]["context"].execute_v2(bindings=self.models["decoder"]["io"]["bindings"])
        o_de = torch.reshape(self.models["decoder"]["io"]["outputs"]["o_de"][:np.prod([bs_to_use, 80, y_length_max])],
                            [bs_to_use, 80, y_length_max])
    
        # ----- DECODER - END

        self.models["vocoder"]["io"]["inputs"]["c"][:np.prod([bs_to_use, 80, y_length_max])] = o_de.flatten()[:]
        self.models["vocoder"]["context"].set_binding_shape(self.models["vocoder"]["engine"].get_binding_index("c"), o_de.shape)
        assert self.models["vocoder"]["context"].all_binding_shapes_specified
        self.models["vocoder"]["context"].execute_v2(bindings=self.models["vocoder"]["io"]["bindings"])
        o_shape = self.models["vocoder"]["io"]["outputs"]["o_shape"]
        waveform = torch.reshape(self.models["vocoder"]["io"]["outputs"]["o"][:np.prod([bs_to_use, 1, o_shape.cpu().numpy()[0]])],
                                 [bs_to_use, 1, o_shape])

        attn = torch.sum(attn, dim=(-1, -2))
        attn = attn - 5
        attn = (attn * int(waveform.shape[2]/attn.max())).to(torch.int)
        waveform = waveform.cpu().numpy()

        for i, wave in enumerate(waveform):
            wave = wave.squeeze()[:attn[i]]
            wave = wave.squeeze()
            wavs.append(wave)

        return wavs

    @staticmethod
    def _set_speaker_input(aux_input):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)
        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    @staticmethod
    def format_durations(o_dr_log, x_mask):
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def expand_encoder_outputs(self, en, dr, x_mask, y_mask):
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.matmul(attn.transpose(1, 2).to(en.dtype), en.transpose(1, 2)).transpose(1, 2)
        return o_en_ex, attn

    @staticmethod
    def load_model(trt_engine_file):
        trt_logger = trt.Logger()
        trt_logger.min_severity = trt.Logger.VERBOSE

        print(trt_engine_file, flush=True)
    
        with open(trt_engine_file, "rb") as f:
            trt_runtime = trt.Runtime(trt_logger)
            trt_engine = trt_runtime.deserialize_cuda_engine(f.read())
            trt_context = trt_engine.create_execution_context()

        return {
            "runtime": trt_runtime,
            "engine": trt_engine,
            "context": trt_context
        }

    @staticmethod
    def allocate_binding_buffer(types_dict, shapes_dict):
        '''
        Allocate binding buffers for trt based on provided types and shapes dict
        '''
        return {
            k: torch.empty(
                size=(np.prod(shape),),
                dtype=types_dict[k],
                device='cuda',
            )
            for k, shape in shapes_dict.items()
        }

    def _allocate_memory(self, model):
        # print(model)
        """Helper function for binding several inputs at once and pre-allocating the results."""
        # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
        inputs = self.allocate_binding_buffer(model["dim_specs"]["input_types"], model["dim_specs"]["input_shapes"])
        outputs = self.allocate_binding_buffer(model["dim_specs"]["output_types"], model["dim_specs"]["output_shapes"])

        bindings = [None] * model["engine"].num_bindings
        device_memory_addresses = {}
        for binding in model["engine"]:
            binding_idx = model["engine"].get_binding_index(binding)
            dtype = trt.nptype(model["engine"].get_binding_dtype(binding))

            if model["engine"].binding_is_input(binding):
                bindings[binding_idx] = int(inputs[binding].data_ptr())
                device_memory_addresses[binding] = int(inputs[binding].data_ptr())
            else:
                bindings[binding_idx] = int(outputs[binding].data_ptr())
                device_memory_addresses[binding] = int(outputs[binding].data_ptr())

        return {
            "bindings": bindings,
            "inputs": inputs,
            "outputs": outputs,
            "device_mem_address": device_memory_addresses
        }

    def load_trt_models(self, trt_model_paths, models_dim_specs):

        models = {}

        for model_name, trt_path in trt_model_paths.items():
            models[model_name] = self.load_model(trt_path)
            models[model_name]["data_stream"] = {}
            for binding in models[model_name]["engine"]:
                err, models[model_name]["data_stream"][binding] = cuda.cuStreamCreate(1)
            err, models[model_name]["model_stream"] = cuda.cuStreamCreate(1)
            models[model_name]["dim_specs"] = models_dim_specs[model_name]
            models[model_name]["io"] = self._allocate_memory(models[model_name])
            
        return models

    @staticmethod
    def release_trt_resources(trt_models):

        models = {}

        for model_name, model in trt_models.items():
            for binding in model["engine"]:
                err, = cuda.cuStreamDestroy(model["data_stream"][binding])

            for node in model["io"]["device_mem_address"].keys():
                err, = cuda.cuMemFree(model["io"]["device_mem_address"][node])    

            err, = cuda.cuStreamDestroy(model["model_stream"])    

        return models