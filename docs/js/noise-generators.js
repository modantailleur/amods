// Plain white/pink noise - alternatives to speech-shaped noise
// (speech-shaped-noise.js) for the "Noise controls" panel's noise-type
// dropdown. Unlike SpeechShapedNoise, these have no relationship to the
// live mic at all - no spectral shape to track, so no FFT, no segments,
// no crossfade, just a continuous per-sample generator. They share
// SpeechShapedNoise's { reset(), feed(x), nextBlock(frames) } interface
// anyway, purely so stream.js can hold all three and swap the active one
// uniformly - feed(x) is a deliberate no-op on both.
//
// WHITE_CALIBRATED_SCALE/PINK_CALIBRATED_SCALE below put each generator's
// own native output at 0.1 RMS - calibrated empirically (2M-sample Monte
// Carlo, see scratch calibration script) against each one's own native
// RMS, not derived analytically (pink noise's filtered RMS in particular
// has no simple closed form worth hand-deriving). 0.1 was chosen to sit
// in the same ballpark SpeechShapedNoise's own mic-tracking output
// naturally lands in, so switching the dropdown wouldn't jump wildly in
// loudness - but both were still reported as "way too loud" at full
// ("Noise" slider = 1) level on top of that.
//
// NOISE_COLOR_TRIM_DB is the one knob to change that: a flat dB trim
// applied on top of the 0.1 RMS calibration, same for both colors. Raise
// it toward 0 for louder, more negative for quieter (every -6dB roughly
// halves the amplitude; -20dB is a factor of 10 quieter).
// Exported so water-noise-generator.js can calibrate the "Water flow"
// color to the same ballpark loudness via the same trim, rather than
// picking its own independent constant.
export const NOISE_COLOR_TRIM_DB = -30;
export const NOISE_COLOR_TARGET_RMS = 0.1; // the pre-trim RMS target every color is calibrated to
const WHITE_CALIBRATED_SCALE = 0.17318; // native uniform-[-1,1] RMS (1/sqrt(3) ~= 0.577), calibrated to 0.1 RMS
const PINK_CALIBRATED_SCALE = 0.05845; // native Kellet-filter RMS (no closed form), calibrated to 0.1 RMS
const NOISE_COLOR_TRIM = 10 ** (NOISE_COLOR_TRIM_DB / 20);
// Every color's known, calibrated native RMS (post-trim) - exported so
// stream.js's "Sensitivity" level-following can compare the live source
// against this fixed constant instead of measuring the noise generator's
// own output, which only stays representative for stationary synthetic
// noise (white/pink) and not for a real, dynamically-varying recording
// (water) - see stream.js's own comment at its use site.
export const NOISE_COLOR_REFERENCE_RMS = NOISE_COLOR_TARGET_RMS * NOISE_COLOR_TRIM;
const WHITE_SCALE = WHITE_CALIBRATED_SCALE * NOISE_COLOR_TRIM;
const PINK_SCALE = PINK_CALIBRATED_SCALE * NOISE_COLOR_TRIM;

export class WhiteNoiseGenerator {
  constructor(sr) {
    this.sr = sr;
  }

  reset() {}

  /** No-op - white noise has no relationship to the live mic. */
  feed() {}

  nextBlock(frames) {
    const out = new Float32Array(frames);
    for (let i = 0; i < frames; i++) out[i] = (Math.random() * 2 - 1) * WHITE_SCALE;
    return out;
  }
}

/**
 * Paul Kellet's "economy" pink noise approximation - three one-pole
 * filters applied in parallel to a shared white-noise input, summed. A
 * well-known, cheap (no FFT) way to get a good-enough ~1/f (pink) power
 * spectrum - this specific set of pole/gain constants is the commonly
 * published "economy" variant (slightly less accurate at the very
 * lowest frequencies than Kellet's full version, which uses an
 * additional pole; not worth the extra cost here).
 */
export class PinkNoiseGenerator {
  constructor(sr) {
    this.sr = sr;
    this.b0 = 0;
    this.b1 = 0;
    this.b2 = 0;
  }

  reset() {
    this.b0 = 0;
    this.b1 = 0;
    this.b2 = 0;
  }

  /** No-op - pink noise has no relationship to the live mic. */
  feed() {}

  nextBlock(frames) {
    const out = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      const white = Math.random() * 2 - 1;
      this.b0 = 0.99765 * this.b0 + white * 0.099046;
      this.b1 = 0.963 * this.b1 + white * 0.2965164;
      this.b2 = 0.57 * this.b2 + white * 1.0526913;
      out[i] = (this.b0 + this.b1 + this.b2 + white * 0.1848) * PINK_SCALE;
    }
    return out;
  }
}
