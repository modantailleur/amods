"""
Step 2 of 2 in reproducing docs/models/dns64.int8.onnx - run
scripts/quantize_dns64.py first (step 1), then this script on its output.

Shrinks the model further by converting its remaining Conv/ConvTranspose
weights (float32, ~67MB - everything step 1's LSTM-only int8 quantization
deliberately left alone) to float16. LSTM stays int8 (untouched); the
graph's actual inputs/outputs stay float32 so callers (denoiser-dns64.js)
don't change.

Why float16 for Conv instead of also int8: onnxruntime's dynamic
quantization turns a quantized Conv into ConvInteger, which has no
implementation in onnxruntime-web's WASM backend (see quantize_dns64.py's
docstring - this is why Conv was left alone there); float16 has no such
op-identity problem, since Conv itself stays Conv.

This does NOT meaningfully speed up inference (Conv - unmodified compute
path - dominates runtime either way, confirmed by benchmarking the LSTM-only
int8 model against the original: only a ~15-20% difference). The entire
point is download size, which is what actually matters for a model users
fetch over a mobile connection before the app can start.

Why this is done manually instead of via onnxconverter_common's
convert_float_to_float16 (a generic whole-graph converter, and the obvious
first choice): that function needs onnx.shape_inference.infer_shapes to
figure out where float32/float16 boundaries fall, and shape_inference hangs
on DynamicQuantizeLSTM (a com.microsoft-domain contrib op from the prior
int8 step it doesn't understand) - confirmed here to take 10+ minutes
without finishing. Skipping that pass via disable_shape_infer=True makes
conversion instant (~0.5s) but breaks correctness two different ways,
also confirmed here: (a) it leaves some pre-existing Cast nodes' declared
output type stale, and separately (b) with that op_block_list'd too, it
leaves some ops (e.g. Sub) with mismatched float/float16 inputs and no
Cast between them - a genuinely invalid graph, not just stale metadata.
Blocking "Cast" in op_block_list to fix (a) reintroduces a 300+ second hang
of its own.

So instead: only touch Conv/ConvTranspose nodes, each turned into a
self-contained float32-in/float32-out island (Cast to float16 on the way
in, float16 weights, Cast back to float32 on the way out). No other node in
the graph ever sees a float16 tensor, so there is no boundary-placement
problem left to solve and no shape/type inference is needed at all - this
script never even loads onnx.shape_inference.

Verified: loads and runs correctly in both onnxruntime (Python, this
script's own verification step below) and separately in real headless
Chrome via onnxruntime-web (922ms inference, no NaN/Inf, correct shape).
Output quality vs. step 1's int8-only model: 0.99995 cosine similarity on a
random test input - the float16 rounding is not perceptible.

Usage:
    python scripts/quantize_dns64.py            # step 1
    python scripts/float16_denoiser_conv.py     # step 2 (this file)
Reads docs/models/dns64.int8.lstmonly.onnx (step 1's output), writes
docs/models/dns64.int8.onnx - the file docs/index.html's denoiser dropdown
actually ships.

Requires: pip install onnx onnxruntime numpy
"""
import os
import time

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "docs", "models", "dns64.int8.lstmonly.onnx")
DST = os.path.join(HERE, "..", "docs", "models", "dns64.int8.onnx")

MODEL_LEN = 32085  # fixed input/output length for this graph - see denoiser-dns64.js
TARGET_OPS = {"Conv", "ConvTranspose"}


def convert_conv_weights_to_fp16(model):
    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    converted_init_names = set()
    new_initializers = []
    new_nodes = []
    n_converted = 0

    for i, node in enumerate(graph.node):
        if node.op_type not in TARGET_OPS:
            new_nodes.append(node)
            continue

        # A few Conv nodes (near the int8-quantized LSTM boundary) get their
        # weight/bias from existing Cast node outputs instead of plain
        # initializers - skip converting those rather than risk mis-handling
        # whatever that pattern actually represents; it's a small minority
        # of the 24 Conv/ConvTranspose nodes, so leaving them float32 costs
        # little size and avoids a real correctness risk.
        if not all(inp in initializer_map for inp in node.input[1:]):
            new_nodes.append(node)
            continue
        n_converted += 1

        # Cast the node's data input (its first input only - W/B are
        # handled below, not cast at runtime, just stored as float16
        # directly) to float16.
        x_name = node.input[0]
        cast_in_out = f"{x_name}__fp16in_{i}"
        cast_in = onnx.helper.make_node(
            "Cast", inputs=[x_name], outputs=[cast_in_out], to=TensorProto.FLOAT16, name=f"CastIn_{i}"
        )
        new_nodes.append(cast_in)

        new_inputs = [cast_in_out]
        for inp_name in node.input[1:]:
            if inp_name in initializer_map and inp_name not in converted_init_names:
                arr = numpy_helper.to_array(initializer_map[inp_name])
                fp16_init = numpy_helper.from_array(arr.astype(np.float16), name=inp_name)
                new_initializers.append(fp16_init)
                converted_init_names.add(inp_name)
            new_inputs.append(inp_name)

        orig_output = node.output[0]
        fp16_output = f"{orig_output}__fp16out_{i}"
        fp16_node = onnx.NodeProto()
        fp16_node.CopyFrom(node)
        fp16_node.input[:] = new_inputs
        fp16_node.output[:] = [fp16_output]
        new_nodes.append(fp16_node)

        cast_out = onnx.helper.make_node(
            "Cast", inputs=[fp16_output], outputs=[orig_output], to=TensorProto.FLOAT, name=f"CastOut_{i}"
        )
        new_nodes.append(cast_out)

    kept_initializers = [init for init in graph.initializer if init.name not in converted_init_names]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    graph.initializer.extend(new_initializers)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Not maintained by the surgery above (harmless to drop - onnxruntime
    # re-derives what it needs at session-creation time, the same way it
    # already has to for DynamicQuantizeLSTM, which isn't in here either).
    del graph.value_info[:]

    return n_converted


def main():
    print(f"loading {SRC} ({os.path.getsize(SRC) / 1e6:.1f} MB)")
    model = onnx.load(SRC)

    print("converting Conv/ConvTranspose to float16 islands (cast in, float16 weights+compute, cast out)...")
    t0 = time.time()
    n_converted = convert_conv_weights_to_fp16(model)
    print(f"converted {n_converted} nodes in {time.time() - t0:.1f}s")

    onnx.save(model, DST)
    print(f"saved {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")

    print("verifying with onnxruntime (CPU EP)...")
    import onnxruntime as ort

    session = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mix = rng.uniform(-0.1, 0.1, size=(1, 1, MODEL_LEN)).astype(np.float32)
    t0 = time.time()
    (denoised,) = session.run(None, {"mix": mix})
    elapsed_ms = (time.time() - t0) * 1000
    assert denoised.shape == (1, 1, MODEL_LEN), denoised.shape
    assert denoised.dtype == np.float32, denoised.dtype
    assert np.isfinite(denoised).all(), "output contains NaN/Inf"
    print(f"OK: output shape {denoised.shape}, dtype {denoised.dtype}, no NaN/Inf, inference took {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    main()
