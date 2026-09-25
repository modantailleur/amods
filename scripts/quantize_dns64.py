"""
Step 1 of 2 in an EARLIER, now-superseded pipeline for shrinking
docs/models/dns64.onnx - kept for reference/history, not what's currently
shipped. See scripts/quantize_dns64_max.py for what docs/index.html's
denoiser dropdown actually ships today (also int8 for Conv/ConvTranspose,
not just float16, for a smaller file - this pipeline's own LSTM-only int8
quantization step is still relevant background for why that script's Conv
step exists, so it's kept rather than deleted). If reproducing this older
pipeline anyway: run scripts/float16_denoiser_conv.py on this script's own
output next - see that script for step 2.

This step dynamically quantizes only the model's LSTM weights (50% of its
total weight bytes - verified by summing each op type's initializer tensor
sizes) to int8; Conv/ConvTranspose (the other 50%, handled by step 2
instead) are deliberately left alone here:

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

This script's own output (~84MB) is an intermediate file. In this old
pipeline, scripts/float16_denoiser_conv.py shrinks the remaining
Conv/ConvTranspose weights further via float16 (not int8 - dynamic
quantization can't get Conv to int8 in a WASM-compatible way, per above)
down to ~50MB. scripts/quantize_dns64_max.py takes a different, better
approach for that step now (static QDQ int8 quantization instead of
dynamic - a different quantization format sidesteps the ConvInteger
problem), reaching ~34MB - see that script for what's actually shipped.

Verified: the resulting file loads and runs correctly in real headless
Chrome via onnxruntime-web, producing non-NaN output of the expected shape.

Usage:
    python scripts/quantize_dns64.py
Reads docs/models/dns64.onnx, writes docs/models/dns64.int8.lstmonly.onnx.

Requires: pip install onnx onnxruntime
"""
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "docs", "models", "dns64.onnx")
DST = os.path.join(HERE, "..", "docs", "models", "dns64.int8.lstmonly.onnx")


def main():
    print(f"loading {SRC} ({os.path.getsize(SRC) / 1e6:.1f} MB)")
    print("quantizing LSTM weights to int8 (Conv is left alone - see module docstring for why)...")
    quantize_dynamic(SRC, DST, weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"])
    print(f"done: {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
