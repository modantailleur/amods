// Port of FbDM in src/amods/models/denoiser.py, using the exported
// dns64.onnx graph via onnxruntime-web instead of the `denoiser` PyPI
// package's live PyTorch model.
//
// Unlike the VAD, this graph has a FIXED input/output shape [1, 1, 32085]
// (verified via onnx.load - not dynamic), and 32085 samples at the model's
// 16kHz rate is ~2.005s - matching amods's own pending_voice_max_duration
// (2s) almost exactly, since _feed_memory (granspeechmask.py) always calls
// the denoiser on that whole buffer as one batch pass, never per audio
// callback. That's what makes running this here, on a ~2s cadence rather
// than every ~50ms, workable at all in a browser.
import { resampleLinear } from './resample.js';

const MODEL_SR = 16000;
const MODEL_LEN = 32085;

export class DnsDenoiser {
  /** @param {ort.InferenceSession} session - loaded from dns64.onnx */
  constructor(session, { sr = 48000 } = {}) {
    this.session = session;
    this.sr = sr;
  }

  /**
   * x: Float32Array mono audio at this.sr, any length.
   * Returns a Float32Array of the same length as x, resampled back to
   * this.sr - matching FbDM.predict's contract. The model itself only
   * ever sees exactly MODEL_LEN samples at MODEL_SR; x is resampled, then
   * padded/trimmed to fit, and the result is resampled/trimmed back.
   */
  async predict(x) {
    let noisy = this.sr !== MODEL_SR ? resampleLinear(x, this.sr, MODEL_SR) : x;

    const fixed = new Float32Array(MODEL_LEN);
    const copyLen = Math.min(noisy.length, MODEL_LEN);
    fixed.set(noisy.subarray(0, copyLen));
    // If noisy is longer than MODEL_LEN, samples beyond copyLen are simply
    // dropped (matching a fixed-size graph's own hard limit); if shorter,
    // the tail stays zero-padded.

    const inputTensor = new ort.Tensor('float32', fixed, [1, 1, MODEL_LEN]);
    const results = await this.session.run({ mix: inputTensor });
    const denoised = results.denoised.data; // Float32Array, length MODEL_LEN

    let enhanced = denoised;
    if (this.sr !== MODEL_SR) {
      enhanced = resampleLinear(denoised, MODEL_SR, this.sr);
    }

    // Trim/pad back to x's original length, same as the Python version
    // implicitly does by resampling a same-length buffer back.
    if (enhanced.length === x.length) return enhanced;
    const out = new Float32Array(x.length);
    out.set(enhanced.subarray(0, Math.min(enhanced.length, x.length)));
    return out;
  }
}
