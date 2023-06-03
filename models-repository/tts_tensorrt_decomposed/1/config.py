import torch

MAX_BS = 6
MAX_ENCODER_SEQUENCE_LENGTH = 400
HIDDEN_SIZE = 512
MAX_DECODER_SEQUENCE_LENGTH = 817
MAX_VOCODER_OUTPUT_LENGTH = 282880

TRT_MODELS_DIM_SPECS = {
    "embedding": {
        "input_shapes": {
            "x": (MAX_BS, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "x": torch.int32,
        },
        "output_shapes": {
            "x_emb": (MAX_BS, MAX_ENCODER_SEQUENCE_LENGTH, HIDDEN_SIZE),
        },
        "output_types": {
            "x_emb": torch.float32,
        }
    },
    "encoder": {
        "input_shapes": {
            "encoder_input_x": (MAX_BS, HIDDEN_SIZE, MAX_ENCODER_SEQUENCE_LENGTH),
            "x_mask": (MAX_BS, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "encoder_input_x": torch.float32,
            "x_mask": torch.float32,
        },
        "output_shapes": {
            "o_en": (MAX_BS, HIDDEN_SIZE, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_en": torch.float32,
        }
    },
    "duration_predictor": {
        "input_shapes": {
            "o_en": (MAX_BS, HIDDEN_SIZE, MAX_ENCODER_SEQUENCE_LENGTH),
            "x_mask": (MAX_BS, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "o_en": torch.float32,
            "x_mask": torch.float32,
        },
        "output_shapes": {
            "o_dr_log": (MAX_BS, 1, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_dr_log": torch.float32,
        }
    },
    "pitch_predictor": {
        "input_shapes": {
            "o_en": (MAX_BS, HIDDEN_SIZE, MAX_ENCODER_SEQUENCE_LENGTH),
            "x_mask": (MAX_BS, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "o_en": torch.float32,
            "x_mask": torch.float32,
        },
        "output_shapes": {
            "o_pitch": (MAX_BS, 1, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_pitch": torch.float32,
        }
    },
    "pitch_embedding": {
        "input_shapes": {
            "o_pitch": (MAX_BS, 1, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "o_pitch": torch.float32,
        },
        "output_shapes": {
            "o_pitch_emb": (MAX_BS, HIDDEN_SIZE, MAX_ENCODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_pitch_emb": torch.float32,
        }
    },
    "positional_encoder": {
        "input_shapes": {
            "o_en_ex_in": (MAX_BS, HIDDEN_SIZE, MAX_DECODER_SEQUENCE_LENGTH),
            "y_mask": (MAX_BS, 1, MAX_DECODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "o_en_ex_in": torch.float32,
            "y_mask": torch.float32,
        },
        "output_shapes": {
            "o_en_ex_out": (MAX_BS, HIDDEN_SIZE, MAX_DECODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_en_ex_out": torch.float32,
        }
    },
    "decoder": {
        "input_shapes": {
            "o_en_ex": (MAX_BS, HIDDEN_SIZE, MAX_DECODER_SEQUENCE_LENGTH),
            "y_mask": (MAX_BS, 1, MAX_DECODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "o_en_ex": torch.float32,
            "y_mask": torch.float32,
        },
        "output_shapes": {
            "o_de": (MAX_BS, 80, MAX_DECODER_SEQUENCE_LENGTH),
        },
        "output_types": {
            "o_de": torch.float32,
        }
    },
    "vocoder": {
        "input_shapes": {
            "c": (MAX_BS, 80, MAX_DECODER_SEQUENCE_LENGTH),
        },
        "input_types": {
            "c": torch.float32,
        },
        "output_shapes": {
            "o": (MAX_BS, 1, MAX_VOCODER_OUTPUT_LENGTH),
            "o_shape": (1)
        },
        "output_types": {
            "o": torch.float32,
            "o_shape": torch.int64
        }
    },
}