// docs/index.html's "Denoiser" dropdown value -> ONNX model file, relative
// to docs/js/. Single source of truth for main.js (which decides whether to
// spin up docs/js/denoiser-worker.js at all, and with which model) - see
// worker-engine.js's header comment for why the denoiser itself no longer
// lives there.
// dns64.int8.onnx (~34MB) is produced by scripts/quantize_dns64_max.py -
// LSTM weights quantized dynamically to int8, Conv/ConvTranspose weights
// quantized statically (QDQ format, calibrated on real speech) to int8 too.
// Quality is signal-level dependent - near-lossless on normal/loud speech,
// more noticeable on quiet passages - see that script's docstring for
// measured numbers before changing this further.

export const DENOISER_MODEL_PATHS = {
  original: '../models/dns64.onnx',
  int8: '../models/dns64.int8.onnx',
};
