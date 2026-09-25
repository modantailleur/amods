"""
Shrink docs/models/dns64.onnx into docs/models/dns64.int8.onnx, the
"quantized" option in the web GUI's denoiser dropdown (docs/index.html).

Only the model's LSTM weights (50% of its total weight bytes - verified by
summing each op type's initializer tensor sizes) are dynamically quantized
to int8; Conv/ConvTranspose (the other 50%) are deliberately left alone.
This is NOT the smallest theoretically possible result - quantizing Conv
too would shrink the file further (~50MB instead of ~84MB) - but it is the
largest reduction that actually WORKS in a browser:

  - onnxruntime's dynamic quantization implements a quantized Conv via a
    ConvInteger node, which has NO implementation in onnxruntime-web's WASM
    execution provider. Loading a Conv-quantized model in an actual browser
    fails outright ("Could not find an implementation for ConvInteger") -
    confirmed by testing it in real headless Chrome, even though the exact
    same file loads and runs fine in onnxruntime-node (Node's native
    backend has broader op coverage than the WASM backend browsers
    actually use - testing only in Node is NOT sufficient to confirm
    browser compatibility).
  - The quantized LSTM op (DynamicQuantizeLSTM) IS implemented in the WASM
    backend - confirmed the same way, in real headless Chrome, including
    running real inference on it (not just loading the session).
  - Combining this with a float16 conversion of the remaining Conv/
    ConvTranspose weights (for further size reduction) was also attempted,
    but onnxconverter_common's float16 conversion took 10+ minutes without
    finishing on this model (twice) - impractical, abandoned rather than
    left half-done. If revisited, that's the next thing to try.

Verified (see this session's history): the resulting file loads and runs
correctly in real headless Chrome via onnxruntime-web, producing
non-NaN output of the expected shape.

Usage:
    python scripts/quantize_dns64.py
Reads docs/models/dns64.onnx, writes docs/models/dns64.int8.onnx.

Requires: pip install onnx onnxruntime
"""
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "docs", "models", "dns64.onnx")
DST = os.path.join(HERE, "..", "docs", "models", "dns64.int8.onnx")


def main():
    print(f"loading {SRC} ({os.path.getsize(SRC) / 1e6:.1f} MB)")
    print("quantizing LSTM weights to int8 (Conv is left alone - see module docstring for why)...")
    quantize_dynamic(SRC, DST, weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"])
    print(f"done: {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
