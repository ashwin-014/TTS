import numpy as np
from tritonclient.utils import serialize_byte_tensor

# serialized = "नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है".encode("utf-8")
serialized = serialize_byte_tensor(np.array([
    "नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है. नमस्ते आपका नाम क्या है".encode("utf-8")
], dtype=np.object_))

with open("./warmup-input", "wb") as fh:
    fh.write(serialized)