// Continuously tracks "how loud is this signal, roughly over the last N
// seconds" for the Noise bed's new "Sensitivity" slider (see stream.js's
// own comment on noiseLevelSensitivitySeconds) - used once for the live
// source signal and once for the noise generator's own raw output, then
// their ratio scales the noise's level to follow the source's loudness
// over time (see stream.js's processChunk).
//
// Reuses the same K-weighting + gating approach as granspeechmask.js's
// energy-matching (see k-weighting.js and gatedMeanSquare's own header
// comments there) rather than a plain flat RMS, for the same reasons: a
// stretch that's mostly silence plus one brief loud burst shouldn't drag
// the measured level down to "mostly silent", and all frequencies
// shouldn't count equally toward perceived loudness.
//
// Unlike granspeechmask.js's one-shot measurements (a single stored clip,
// or a single ~2s snapshot of recent history), this runs continuously for
// the life of a session, updated once per real-time chunk - so instead of
// gating a single big buffer in one pass, each chunk is K-weighted through
// a PERSISTENT filter (see k-weighting.js's StreamingKWeighting - carries
// its own history correctly across calls, so its startup transient only
// ever happens once, not every chunk) and pushed into a small (~100ms)
// rolling buffer, which IS re-gated (via granspeechmask.js's own
// gatedMeanSquare, unchanged) every chunk - cheap, since it's a short,
// fixed-size buffer, not the long history. That short-term gated reading
// is then smoothed over time via an EMA (exponential moving average) -
// see setTimeConstant's own comment for why EMA instead of a literal
// multi-second rolling buffer: a sensitivity slider's whole 1-60s range
// costs the exact same (one float) either way, with no large buffer to
// maintain regardless of how slow/long the chosen time constant is.
import { StreamingKWeighting } from './k-weighting.js';
import { gatedMeanSquare } from './granspeechmask.js';

const HOP_SECONDS = 0.05; // matches this app's real-time chunk cadence (CHUNK_SIZE_AT_48K @ 48kHz) - the EMA is updated once per chunk, not per sample
// Short enough to be cheap (a few hundred samples, not multiple seconds),
// long enough to contain several of gatedMeanSquare's own 20ms sub-windows
// so its "exclude the quiet ones" gating has enough windows to work with
// (GATE_WINDOW_MS=20 in granspeechmask.js - 100ms gives 5 of them).
const SHORT_WINDOW_SECONDS = 0.1;

export class LevelTracker {
  constructor(sr) {
    this.sr = sr;
    this.kWeighting = new StreamingKWeighting(sr);
    this.windowSize = Math.max(1, Math.round(sr * SHORT_WINDOW_SECONDS));
    this.window = new Float64Array(this.windowSize); // rolling - holds the most recent windowSize K-weighted samples
    this.windowFilled = 0; // how much of `window` has real audio yet (ramps up to windowSize once, then stays there)
    this.levelMeanSq = 0; // the EMA itself - mean-square (power), not RMS, so it can be updated by a plain weighted-average of other mean-square values
    this.emaAlpha = 1; // set by setTimeConstant below; 1 (no smoothing at all) until that's actually called
  }

  /** seconds: the EMA's time constant (see this module's own header - NOT a literal window length, see the LevelTracker-wide comment for why). Clamped the same way SpeechShapedNoise.setSensitivitySeconds is. */
  setTimeConstant(seconds) {
    const clamped = Math.max(0.05, seconds);
    this.emaAlpha = 1 - Math.exp(-HOP_SECONDS / clamped);
  }

  /**
   * Runs the window-feeding + short-term gated measurement that both
   * update() and updateGated() below need, but stops short of committing
   * it to the EMA - callers decide whether/how to do that themselves.
   * Always advances the rolling window regardless of that decision, so
   * the window's own content never falls behind the real audio.
   */
  _measure(x) {
    const weighted = this.kWeighting.process(x);
    // Shift the rolling window left by weighted.length, zero-filling the
    // tail, then copy the new samples in - same ring-buffer-via-shift
    // technique as speech-shaped-noise.js's own texture buffer.
    const n = weighted.length;
    if (n >= this.windowSize) {
      this.window.set(weighted.subarray(n - this.windowSize));
      this.windowFilled = this.windowSize;
    } else {
      this.window.copyWithin(0, n);
      this.window.set(weighted, this.windowSize - n);
      this.windowFilled = Math.min(this.windowSize, this.windowFilled + n);
    }
    // Only gate over the portion that's actually real audio so far (early
    // on, right after construction, the rest of `window` is still zero-
    // filled padding, which would otherwise count as "silence" and get
    // gated out anyway, but there's no reason to feed gatedMeanSquare more
    // than what's real).
    const active = this.windowFilled === this.windowSize ? this.window : this.window.subarray(this.windowSize - this.windowFilled);
    return gatedMeanSquare(active, this.sr);
  }

  /** Feed one chunk of audio (Float32Array, mono, at this.sr) - call exactly once per real-time chunk. */
  update(x) {
    const shortTermMeanSq = this._measure(x);
    this.levelMeanSq += this.emaAlpha * (shortTermMeanSq - this.levelMeanSq);
  }

  /**
   * Like update(), but asymmetric: a chunk that would pull the level DOWN
   * is always accepted (so it keeps decaying freely toward whatever's
   * actually quiet, e.g. real silence/room noise once speech stops), while
   * a chunk that would push it UP is only accepted when speechDetected is
   * true. This is "Purity"'s mechanism (see stream.js's processChunk) -
   * unlike an earlier version of this gate that simply froze the average
   * outright on non-speech chunks (which never let it fall at all, since
   * a frozen value never decays on its own), letting falls through
   * unconditionally is what actually pulls the tracked level down during
   * real silence, while still refusing to let ambient noise/false
   * triggers inflate it back up.
   */
  updateGated(x, speechDetected) {
    const shortTermMeanSq = this._measure(x);
    const candidate = this.levelMeanSq + this.emaAlpha * (shortTermMeanSq - this.levelMeanSq);
    if (candidate <= this.levelMeanSq || speechDetected) {
      this.levelMeanSq = candidate;
    }
  }

  /** Current smoothed level, as RMS (not mean-square/power) - directly comparable to another LevelTracker's own getRms() for a ratio. */
  getRms() {
    return Math.sqrt(this.levelMeanSq);
  }
}
