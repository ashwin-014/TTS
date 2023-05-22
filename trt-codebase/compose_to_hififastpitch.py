import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
import onnx_graphsurgeon as gs
import numpy as np
import argparse

# fastpitch = onnx.load_model("./nosqueeze_transpose/fastpitch_noidentity.onnx")
# fastpitch_graph = gs.import_onnx(fastpitch)

# model_outputs_node = [node for node in fastpitch_graph.nodes if node.name == "/decoder/decoder/Mul_1"][0]

# # print(model_outputs_node)

# identity_out = gs.Variable("decoder_output", shape=(model_outputs_node.outputs[0].shape), dtype=np.float32)
# identity = gs.Node(op="Identity", inputs=model_outputs_node.outputs, outputs=[identity_out])

# fastpitch_graph.nodes.append(identity)
# fastpitch_graph.outputs += [identity_out]
# fastpitch_graph.cleanup().toposort()

# onnx.save(gs.export_onnx(fastpitch_graph), "./nosqueeze_transpose/fastpitch.onnx")

fastpitch = onnx.load_model("models/v1/hi/nosqueeze_transpose/fastpitch.onnx")
onnx.checker.check_model(fastpitch)
hifigan = onnx.load_model("models/v1/hi/nosqueeze_transpose/vocoder.onnx")
onnx.checker.check_model(hifigan)

fastpitch = onnx.compose.add_prefix(fastpitch, prefix="fastpitch/")
hifigan = onnx.compose.add_prefix(hifigan, prefix="hifigan/")

combined_model = onnx.compose.merge_models(
    fastpitch, hifigan,
    io_map=[("fastpitch/model_outputs", "hifigan/c")]
)

combined_model = shape_inference.infer_shapes(combined_model)

onnx.save(combined_model, "models/v1/hi/nosqueeze_transpose/hififastpitch.onnx")