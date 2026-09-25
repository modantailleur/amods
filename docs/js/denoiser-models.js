// docs/index.html's "Denoiser" dropdown value -> ONNX model file, relative
// to docs/js/. Single source of truth for main.js (which decides whether to
// spin up docs/js/denoiser-worker.js at all, and with which model) - see
// worker-engine.js's header comment for why the denoiser itself no longer
// lives there.
export const DENOISER_MODEL_PATHS = {
  original: '../models/dns64.onnx',
  int8: '../models/dns64.int8.onnx',
};
