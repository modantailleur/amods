// docs/index.html's "Denoiser" dropdown value -> ONNX model file, relative
// to docs/js/. Single source of truth for main.js (which decides whether to
// spin up docs/js/denoiser-worker.js at all, and with which model) - see
// worker-engine.js's header comment for why the denoiser itself no longer
// lives there.
// dns64.int8.onnx (~50MB, despite the name) is the result of running BOTH
// scripts/quantize_dns64.py (LSTM weights -> int8) AND, on that script's
// output, scripts/float16_denoiser_conv.py (remaining Conv/ConvTranspose
// weights -> float16) - see those scripts for the full reproduction steps
// and why each is needed.
export const DENOISER_MODEL_PATHS = {
  original: '../models/dns64.onnx',
  int8: '../models/dns64.int8.onnx',
};
