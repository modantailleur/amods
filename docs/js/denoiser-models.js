// docs/index.html's "Denoiser" dropdown value -> ONNX model file, relative
// to docs/js/. Single source of truth for main.js (which decides whether to
// spin up docs/js/denoiser-worker.js at all, and with which model) - see
// worker-engine.js's header comment for why the denoiser itself no longer
// lives there.
// dns64.int8.float16conv.onnx (~50MB) is produced by quantize_dns64.py +
// float16_denoiser_conv.py - LSTM weights quantized dynamically to int8,
// Conv/ConvTranspose weights converted to float16 (~lossless).
//
// dns64.int8.onnx (~34MB, produced by scripts/quantize_dns64_max.py -
// same LSTM step, but Conv/ConvTranspose quantized to int8 too, via static
// QDQ quantization) is smaller but noticeably noisier in practice - not
// currently used by the dropdown below, kept on disk/in the repo in case
// that trade-off becomes preferable later. See quantize_dns64_max.py's
// docstring for the measured quality numbers behind that call.

export const DENOISER_MODEL_PATHS = {
  original: '../models/dns64.onnx',
  int8: '../models/dns64.int8.float16conv.onnx',
  // "FbdeDM" - the de-esser (see deesser.js) runs as a separate JS
  // pre-processing step before denoising, not a different model, so it
  // shares FbDM's own model file.
  fbdedm: '../models/dns64.int8.float16conv.onnx',
};
