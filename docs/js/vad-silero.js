// Port of SileroVAD in src/amods/models/vad.py, using the raw ONNX graph
// (silero_vad.onnx) via onnxruntime-web instead of the silero-vad PyPI
// package. The graph takes `input` [batch, sequence] + a recurrent `state`
// [2, batch, 128] tensor that must be carried across calls (the PyPI
// package hides this bookkeeping inside its own model wrapper; here it's
// done explicitly, since we're calling the graph directly).
import { resampleLinear } from './resample.js';

const VAD_SR = 16000; // Silero VAD's fixed sampling rate
const FRAME_SIZE = 512; // samples per frame at 16kHz

export class SileroVAD {
  /** @param {ort.InferenceSession} session - loaded from silero_vad.onnx */
  constructor(session, { logitThreshold = null, sr = 16000 } = {}) {
    this.session = session;
    this.sr = sr;
    this.logitThreshold = logitThreshold;
    this._resetState();
  }

  _resetState() {
    // [2, batch=1, 128], matches the graph's `state` input shape.
    this.state = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
  }

  /**
   * x: Float32Array of audio in [-1, 1] at this.sr.
   * Returns bool (speech?) based on mean(frame_is_speech) > logitThreshold,
   * or the raw speech ratio if logitThreshold is null - matching the
   * Python wrapper's contract exactly.
   */
  async predict(x) {
    if (x.length === 0) return false;

    let audio = x;
    if (this.sr !== VAD_SR) audio = resampleLinear(audio, this.sr, VAD_SR);

    // Cut from the beginning if not an exact multiple of FRAME_SIZE, same
    // as the Python wrapper (keeps the most recent, freshest audio).
    const remainder = audio.length % FRAME_SIZE;
    const trimmed = remainder !== 0 ? audio.subarray(remainder) : audio;
    const nFrames = Math.floor(trimmed.length / FRAME_SIZE);
    if (nFrames === 0) return this.logitThreshold !== null ? false : 0.0;

    // Note: unlike some public silero_vad.onnx exports, this graph's only
    // inputs are `input` and `state` (verified via onnx.load) - no `sr`
    // tensor, since it's fixed to 16kHz at export time.
    let sum = 0;
    for (let f = 0; f < nFrames; f++) {
      const frame = trimmed.subarray(f * FRAME_SIZE, (f + 1) * FRAME_SIZE);
      const inputTensor = new ort.Tensor('float32', Float32Array.from(frame), [1, FRAME_SIZE]);
      const feeds = { input: inputTensor, state: this.state };
      const results = await this.session.run(feeds);
      this.state = results.stateN;
      sum += results.output.data[0];
    }
    const speechRatio = sum / nFrames;

    return this.logitThreshold !== null ? speechRatio > this.logitThreshold : speechRatio;
  }
}
