// Port of TenVAD in src/amods/models/vad.py, using the official TEN VAD
// WebAssembly build (vendored at docs/vendor/ten-vad/, from
// https://github.com/TEN-framework/ten-vad's lib/Web output - Apache 2.0,
// with an additional-conditions clause; see that repo's LICENSE/NOTICES.
// "Powered by ten-vad", per that license's suggested attribution).
//
// ten_vad.js only exports the raw C API (createVADModule -> a WASM module
// exposing _ten_vad_create/_ten_vad_process/_ten_vad_destroy, compiled with
// EMSCRIPTEN_KEEPALIVE, not C++/Embind bindings) - there is no shipped
// high-level JS wrapper (the project's own ten_vad.d.ts documents the shape
// of one, but that class isn't actually in ten_vad.js). The class below is
// that wrapper, written against the raw API directly per ten_vad.h's
// documented contract.
import createVADModule from '../vendor/ten-vad/ten_vad.js';
import { resampleLinear } from './resample.js';

const VAD_SR = 16000; // TEN VAD's fixed sampling rate (see the .py wrapper's own vad_sr)
const HOP_SIZE = 256; // matches src/amods/models/vad.py's TenVAD(hop_size=256)

export class TenVAD {
  /** @param {object} opts - { logitThreshold, sr, name, debug } */
  constructor({ logitThreshold = null, sr = 16000, name = '?', debug = false } = {}) {
    this.sr = sr;
    this.logitThreshold = logitThreshold;
    this.name = name; // DIAGNOSTIC - see predict()'s [VADDIAG] logging below
    this.debug = debug; // DIAGNOSTIC gate - only worker-engine.js's debugTelemetry (i.e. debug.html) sessions pass true, so index.html never logs any of this
    this.__callCount = 0; // DIAGNOSTIC
    this.module = null;
    this.handle = 0;
    this._audioPtr = 0;
    this._probPtr = 0;
    this._flagPtr = 0;
    // predict() awaits this, so any call before init actually finishes just
    // waits rather than racing the WASM module's own async instantiation.
    this._ready = this._init();
  }

  async _init() {
    // A FRESH WASM module instance per TenVAD instance, deliberately never
    // shared between two instances (e.g. worker-engine.js's sourceVad and
    // concealerVad) even though this API looks fully stateless/parameterized
    // via explicit handles - vad-silero.js's own SileroVAD instances shared
    // one onnxruntime-web InferenceSession under the same "it's just passed
    // explicit tensors, should be safe" assumption, and that turned out to
    // silently corrupt one of them the moment both were ever in play at
    // once. Instantiating this ~283KB wasm module twice is a trivial cost
    // next to repeating that bug.
    this.module = await createVADModule();
    const handlePtr = this.module._malloc(4);
    const rc = this.module._ten_vad_create(handlePtr, HOP_SIZE, 0.0);
    // getValue()/setValue() aren't actually exported by this build (despite
    // the project's own ten_vad.d.ts documenting them as if they were -
    // that file is manually maintained and describes an intended wrapper
    // shape, not this specific build's real exports) - read the raw HEAP32
    // view directly instead. handlePtr is byte-addressed; HEAP32 is a
    // 4-byte-per-element view of the same buffer, hence >> 2.
    this.handle = this.module.HEAP32[handlePtr >> 2];
    this.module._free(handlePtr);
    if (rc !== 0 || !this.handle) throw new Error('ten_vad_create failed');

    // Reusable scratch buffers - predict() runs many times per second, so
    // allocating fresh int16/float/int buffers on every call would churn
    // the WASM heap for no reason.
    this._audioPtr = this.module._malloc(HOP_SIZE * 2); // int16 samples
    this._probPtr = this.module._malloc(4); // float* out_probability
    this._flagPtr = this.module._malloc(4); // int* out_flag (unused - see predict())
  }

  /**
   * x: Float32Array of audio in [-1, 1] at this.sr.
   * Returns bool (speech?) based on mean(frame_is_speech) > logitThreshold,
   * or the raw speech ratio if logitThreshold is null - matching the
   * Python wrapper's contract exactly (and SileroVAD's, for drop-in parity).
   */
  async predict(x) {
    await this._ready;
    if (x.length === 0) return this.logitThreshold !== null ? false : 0.0;

    let audio = x;
    if (this.sr !== VAD_SR) audio = resampleLinear(audio, this.sr, VAD_SR);

    // Cut from the beginning if not an exact multiple of HOP_SIZE, same
    // convention as vad-silero.js/the Python wrapper (keeps the most
    // recent, freshest audio).
    const remainder = audio.length % HOP_SIZE;
    const trimmed = remainder !== 0 ? audio.subarray(remainder) : audio;
    const nFrames = Math.floor(trimmed.length / HOP_SIZE);
    if (nFrames === 0) return this.logitThreshold !== null ? false : 0.0;

    const base = this._audioPtr >> 1; // int16 element index, not byte offset
    let sum = 0;
    for (let f = 0; f < nFrames; f++) {
      const frame = trimmed.subarray(f * HOP_SIZE, (f + 1) * HOP_SIZE);
      // Re-read HEAP16 fresh every frame (not hoisted above the loop) - if
      // the WASM instance's memory ever grows (Emscripten's
      // ALLOW_MEMORY_GROWTH reallocates the whole buffer when it does), a
      // reference captured before that point would silently point at a
      // detached, stale buffer.
      const heap16 = this.module.HEAP16;
      for (let i = 0; i < HOP_SIZE; i++) {
        // float [-1,1] -> int16, matching the Python wrapper's
        // `np.clip(x, -1, 1)` then `(x * 32767).astype('int16')` exactly
        // (astype truncates toward zero, hence Math.trunc here, not round).
        const s = Math.max(-1, Math.min(1, frame[i]));
        heap16[base + i] = Math.trunc(s * 32767);
      }
      const rc = this.module._ten_vad_process(this.handle, this._audioPtr, HOP_SIZE, this._probPtr, this._flagPtr);
      if (rc !== 0) throw new Error('ten_vad_process failed');
      // HEAPF32 is a 4-byte-per-element view of the same buffer as the
      // byte-addressed _probPtr, hence >> 2 (see the constructor's HEAP32 note).
      sum += this.module.HEAPF32[this._probPtr >> 2];
    }
    const speechRatio = sum / nFrames;

    // DIAGNOSTIC (debug.html only - see this.debug above) - mirrors
    // vad-silero.js's own [VADDIAG] logging exactly, same reasoning: this is
    // what caught the shared-session bug there, and debugging VAD behavior
    // live is an ongoing need, not a one-off.
    if (this.debug) {
      this.__callCount++;
      if (this.__callCount % 20 === 0 || speechRatio > 0.3) {
        console.error(`[VADDIAG ${this.name}] call#${this.__callCount} speechRatio=${speechRatio.toFixed(4)} threshold=${this.logitThreshold}`);
      }
    }
    return this.logitThreshold !== null ? speechRatio > this.logitThreshold : speechRatio;
  }
}
