// Non-browser acoustic echo cancellation - WebRTC's own AEC3 (the real
// algorithm Chrome/Meet use for this), run directly by our own code instead
// of relying on getUserMedia's echoCancellation constraint. That constraint
// turned out to be a dead end for this app specifically: Chromium (Chrome/
// Edge) never applies it to audio played via the Web Audio API at all
// (confirmed via a Google engineer's report on the WebRTC mailing list -
// https://groups.google.com/g/discuss-webrtc/c/NQ0f8MwwegQ - and tracked as
// still-open Chromium issue 687574), which is exactly how this app plays
// its concealer/noise mix. A documented RTCPeerConnection-loopback
// workaround was tried first (see git history) to coax the browser's own
// black-box AEC into engaging anyway, but gave no way to inspect, measure,
// or debug what it was actually doing - not acceptable once it looked like
// it wasn't working and there was no way to tell why. This replaces that
// entirely: same real algorithm, but run as an explicit library WE call
// directly with exactly the render/capture data we choose, so every step is
// inspectable.
//
// Port used: https://github.com/ennuicastr/webrtcaec3.js (BSD-3-Clause,
// matching the license of the underlying Google WebRTC AEC3 module it's
// compiled from: https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/).
// Vendored at docs/vendor/webrtcaec3/ (see that directory's own note on the
// two fixes applied to the npm-published build to make it load as a
// browser ES module at all).
//
// Verified directly (not just assumed): a synthetic test feeding real
// recorded speech as the "render" signal and a delayed+attenuated copy as
// "capture" measured a consistent ~20-27dB echo reduction (ERLE) across a
// 20-second run - a pure sine tone, tried first, gave ~0dB, an unrelated
// methodology artifact (a continuous tone is a degenerate, atypical input
// for a speech-tuned canceller), not a sign the library doesn't work.
//
// Usage per real-time chunk (see stream.js's processChunk): call
// processCapture(x) on the RAW mic chunk FIRST - it returns the
// echo-cancelled version, computed from render data registered by earlier
// chunks' analyzeRender calls (AEC3 estimates and continuously re-estimates
// the actual render-to-capture delay itself via cross-correlation - see
// this library's own README - so the caller never needs to know or supply
// it). Once this chunk's own final mixed output is computed, call
// analyzeRender(playMix) so the library can match it against capture chunks
// arriving later (once it's actually traveled speaker->air->mic).
import createWebRtcAec3Module from '../vendor/webrtcaec3/webrtcaec3-0.3.0.js';

// One of only two rates AEC3 supports (32000 or 48000) - the instance is
// always created at this rate regardless of the device's real sampleRate,
// which is instead passed as sampleRateIn/sampleRateOut on every call (the
// library resamples internally - see its own README).
const AEC3_SAMPLE_RATE = 48000;

// The WASM module itself (not an AEC3 instance - see getModule()) only
// ever needs loading once per page, regardless of how many EchoCanceller
// instances get constructed/freed across a session.
let modulePromise = null;
function getModule() {
  if (!modulePromise) modulePromise = createWebRtcAec3Module();
  return modulePromise;
}

export class EchoCanceller {
  constructor(sr) {
    this.sr = sr;
    this._instance = null;
    this._ready = this._init();
  }

  async _init() {
    const AEC3 = await getModule();
    this._instance = new AEC3.AEC3(AEC3_SAMPLE_RATE, 1, 1);
  }

  /**
   * x: Float32Array, one chunk of RAW mic audio at this.sr.
   * Returns a new Float32Array of the same length, with the estimated echo
   * (from render data registered via analyzeRender) subtracted out.
   */
  async processCapture(x) {
    await this._ready;
    const opts = { sampleRateIn: this.sr, sampleRateOut: this.sr };
    const outLen = this._instance.processSize([x], opts);
    const out = [new Float32Array(outLen)];
    this._instance.process(out, [x], opts);
    // processSize's returned length can differ slightly from x.length (a
    // resampling-rounding artifact when this.sr !== AEC3_SAMPLE_RATE) -
    // trim/pad back to x.length so callers always get back exactly what
    // they fed in, sample-for-sample, matching every other fixed-block-
    // length convention in this pipeline.
    if (out[0].length === x.length) return out[0];
    const fixed = new Float32Array(x.length);
    fixed.set(out[0].subarray(0, Math.min(out[0].length, x.length)));
    return fixed;
  }

  /** playMix: Float32Array, one chunk of the FINAL mixed audio about to be sent to the speaker, at this.sr. */
  async analyzeRender(playMix) {
    await this._ready;
    this._instance.analyze([playMix], { sampleRateIn: this.sr });
  }

  free() {
    if (this._instance) {
      this._instance.free();
      this._instance = null;
    }
  }
}
