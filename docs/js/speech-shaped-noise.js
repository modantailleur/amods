// Continuous "speech-shaped noise" (SSN) background bed - an independent
// masking layer, ADDITIVE on top of GranSpeechMask's normal concealer
// clips (see stream.js's processChunk), not a modification of them. This
// is what the project's supervisor actually meant by "concealing noise" -
// concealer-noise.js's blur/noiseShape techniques (applied to individual
// memory clips) were a misunderstanding of that request and are left in
// place but no longer wired to anything live; this is the real feature.
//
// Mechanism, by explicit request: every HOP_SECONDS, generate a fresh
// noise segment and crossfade it with ONLY the immediately preceding one,
// over CROSSFADE_SECONDS - not a multi-way overlap-add (fixed, not tied
// to the sensitivity slider). An earlier version used continuous
// Hann-windowed WOLA with a long window and several (3-7) frames summed
// at any instant, which measured cheap but sounded like "someone speaking
// in a huge cathedral" - unsurprising in hindsight, since that's close to
// literally synthesizing diffuse overlapping reflections. This version
// only ever has TWO segments overlapping (the current one's tail fading
// out, the next one's head fading in) - simple addition of two
// edge-tapered segments, no WOLA normalization involved at all. The taper
// itself is an EQUAL-POWER (sin/cos) curve, not GranSpeechMask's own
// raised-cosine applyFade: applyFade's ramp sums amplitudes to exactly 1
// through the overlap, which is lossless for blending two COHERENT copies
// of the same signal (what GranSpeechMask's overlapping concealer clips
// are) but produces a real, audible ~3dB power dip at the crossfade's
// midpoint for INDEPENDENT signals, which is what two separately
// random-phase-generated segments are - independent variances ADD, not
// amplitudes, so a sum-of-SQUARES-to-1 curve (sin^2+cos^2=1, exactly) is
// what actually keeps the result's loudness constant through the overlap
// here. Confirmed audibly (reported as a clear dip with the amplitude
// curve) before switching.
//
// Each segment is reconstructed by keeping a magnitude spectrum but
// giving every bin an independent random phase (same "noise-vocoded
// speech" principle as concealer-noise.js's noiseShapeConcealer). The
// magnitude isn't just the current snapshot's own - it's an exponential
// moving average (avgMagHalf) over snapshots, which is what the
// "Sensitivity" slider controls the time constant of: long sensitivity
// makes the noise's spectral character drift slowly (several seconds of
// memory); short makes it track near-instantaneous content. This keeps
// "sensitivity" meaningful as a longer memory without having to actually
// analyze an ever-growing window every tick (a literal whole-buffer FFT
// over the sensitivity slider's full range, benchmarked directly,
// measured up to ~60ms per call at its 10s top end - already longer than
// the 50ms hop it would have needed to run inside of, on the single
// thread that also has to keep up with real-time audio and VAD).
//
// avgMagHalf's own OVERALL level directly reflects how loud the live mic
// currently is, so the noise's own loudness naturally swells while the
// speaker is loud and fades during pauses, same as its spectral shape
// does - an earlier revision forced avgMagHalf's level to a fixed
// constant before synthesis every tick specifically to prevent that, but
// that real-time, 50ms-scale level adaptation was explicitly asked to be
// removed (not needed) - avgMagHalf now goes into synthesis as-is. The
// "Noise"/"Concealer level" sliders (in stream.js) still scale the final
// output on top of whatever this naturally produces.
//
// rmsCorrection below corrects for only keeping the first segmentSamples
// of the padded-to-power-of-2 nFft-long random-phase ifft output (its
// energy spreads roughly evenly across the FULL nFft span in expectation,
// so keeping only part of it keeps roughly that fraction) - calibrated
// once, empirically, against a flat test spectrum. KNOWN LIMITATION: that
// calibration doesn't generalize exactly to every possible spectral shape
// - a narrowband/tonal spectrum (an extreme, e.g. a pure sine used in
// testing) truncates far more severely than a broadband one (measured
// ~64x too quiet for a sine, against the flat-spectrum calibration's
// ~1.0x), so in principle the correction can be off for strongly tonal
// content. Two attempted "fixes" for that were tried and both reverted:
// (1) smoothing the correction over a slow EMA/ramp - revealed a separate
// feedback bug that made the gain drift away monotonically; (2) measuring
// and correcting each segment's OWN realized RMS exactly, every tick -
// fixed the average level but could demand a huge (~64x) correction for a
// genuinely narrowband tick, which blew the segment's own PEAK (not just
// its RMS) far past the rest of the signal's range. Removing the
// truncation entirely (segmentSamples = nFft, nothing discarded) fixed
// that in isolated testing but was reported as sounding "completely
// catastrophic" live - periodic severe level drops, and heavy overlap
// artifacts at higher Sensitivity - so it was reverted too, back to this
// simpler, empirically-fine-in-practice fixed correction. If the
// narrowband case turns out to matter audibly in practice, revisit this
// - but don't reintroduce either of the two fixes above without finding
// what actually made the "eliminate truncation" version sound wrong
// first, since its own isolated tests looked clean.
import { fft } from './mel.js';
import { ifft } from './stft.js';

export const SSN_MIN_SENSITIVITY_S = 0.2;
export const SSN_MAX_SENSITIVITY_S = 10;
// Fixed, explicit request - not tied to the sensitivity slider.
const HOP_SECONDS = 0.05;
const CROSSFADE_SECONDS = 0.025;

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Equal-power fade: tapers BOTH ends of x via sin(pi/2*t)/cos(pi/2*t)
 * curves, not GranSpeechMask's own amplitude-linear applyFade - see this
 * module's header comment for why (independent-signal crossfades need
 * sum-of-SQUARES-to-1, not sum-of-amplitudes-to-1, to avoid an audible
 * mid-crossfade loudness dip).
 */
function equalPowerFadeRamp(fadeSize) {
  const ramp = new Float64Array(fadeSize);
  for (let i = 0; i < fadeSize; i++) {
    const t = fadeSize === 1 ? 0 : i / (fadeSize - 1);
    ramp[i] = Math.sin((Math.PI / 2) * t);
  }
  return ramp;
}

function applyEqualPowerFade(x, fadeSize) {
  const n = x.length;
  if (fadeSize <= 0 || n < 2) return x;
  fadeSize = Math.min(fadeSize, Math.floor(n / 2));
  if (fadeSize <= 0) return x;
  const y = Float32Array.from(x);
  const ramp = equalPowerFadeRamp(fadeSize);
  for (let i = 0; i < fadeSize; i++) {
    y[i] *= ramp[i]; // fade-in: 0 -> 1 via sin(pi/2*t)
    y[n - 1 - i] *= ramp[i]; // fade-out: 1 -> 0, read backwards (equivalent to cos(pi/2*t) forward)
  }
  return y;
}

/**
 * Builds a conjugate-symmetric nFft spectrum from a half-spectrum
 * magnitude array (bins 0..nFft/2 inclusive), giving every INTERIOR bin an
 * independent uniform-random phase. DC (k=0) and Nyquist (k=nFft/2, nFft
 * is always even here) are kept REAL - a random CONTINUOUS angle there
 * would give them a spurious nonzero imaginary part with no mirror
 * counterpart to cancel it, breaking conjugate symmetry at exactly those
 * two bins (the ifft would then come out genuinely complex there, not
 * just real - the same bug class concealer-noise.js's own header
 * describes hitting elsewhere) - a random SIGN keeps the "random phase"
 * spirit while staying real, since a real bin's only two valid phases are
 * 0 and pi.
 */
function randomPhaseSpectrum(magHalf, nFft, halfLen) {
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  re[0] = magHalf[0] * (Math.random() < 0.5 ? 1 : -1);
  re[halfLen - 1] = magHalf[halfLen - 1] * (Math.random() < 0.5 ? 1 : -1);
  for (let k = 1; k < halfLen - 1; k++) {
    const angle = Math.random() * 2 * Math.PI;
    re[k] = magHalf[k] * Math.cos(angle);
    im[k] = magHalf[k] * Math.sin(angle);
  }
  for (let k = halfLen; k < nFft; k++) {
    const mirror = nFft - k;
    re[k] = re[mirror];
    im[k] = -im[mirror];
  }
  return { re, im };
}

/**
 * Random-phase reconstruction spreads a fixed total energy roughly evenly
 * (in expectation, not on any single draw) across the FULL nFft-sample
 * ifft output. Each segment only keeps the first segmentSamples of that
 * (nFft is padded up from segmentSamples to a power of 2 for fft()), so it
 * only keeps roughly that fraction of the total energy - calibrated
 * empirically across many independent draws against a FLAT test spectrum
 * (see this module's header for the known narrowband-content caveat).
 */
function calibrateSegmentRmsCorrection(nFft, halfLen, segmentSamples) {
  const targetRms = 1;
  // Flat spectrum whose Parseval-implied RMS over the FULL nFft samples is
  // exactly targetRms: sum_i x[i]^2 = (1/nFft)*sum_k|X[k]|^2 = C^2 for flat
  // |X[k]|=C over nFft bins, so C=sqrt(nFft)*targetRms gives sum_i x[i]^2 =
  // nFft*targetRms^2, i.e. mean-square = targetRms^2.
  const magHalf = new Float64Array(halfLen).fill(Math.sqrt(nFft) * targetRms);

  // Scaled inversely with nFft so this one-time construction-time cost
  // stays roughly constant regardless of hop/crossfade (ifft cost per
  // trial grows with nFft).
  const trials = Math.max(20, Math.round(1.2e6 / nFft));
  let sumSq = 0;
  for (let t = 0; t < trials; t++) {
    const { re, im } = randomPhaseSpectrum(magHalf, nFft, halfLen);
    ifft(re, im);
    for (let i = 0; i < segmentSamples; i++) sumSq += re[i] * re[i];
  }
  const actualRms = Math.sqrt(sumSq / (trials * segmentSamples));
  return actualRms > 1e-12 ? targetRms / actualRms : 1;
}

export class SpeechShapedNoise {
  constructor(sr, sensitivitySeconds = 2) {
    this.sr = sr;
    this.hopSamples = Math.max(1, Math.round(sr * HOP_SECONDS));
    this.crossfadeSamples = Math.max(1, Math.round(sr * CROSSFADE_SECONDS));
    // Each segment fades in over its first crossfadeSamples and out over
    // its last crossfadeSamples (applyEqualPowerFade, below) - inserting a
    // new one every hopSamples means the previous segment's fade-out and
    // the new one's fade-in overlap for exactly crossfadeSamples, and
    // nothing else overlaps (only ever two segments at a time).
    this.segmentSamples = this.hopSamples + this.crossfadeSamples;
    this.nFft = nextPow2(this.segmentSamples); // padded up to a power of 2 for fft()
    this.halfLen = this.nFft / 2 + 1;
    // Corrects for only keeping segmentSamples of the nFft-long random-
    // phase reconstruction - see calibrateSegmentRmsCorrection's own
    // comment. One-time cost, not per-segment (nFft is small here, cheap
    // regardless).
    this.rmsCorrection = calibrateSegmentRmsCorrection(this.nFft, this.halfLen, this.segmentSamples);

    // Rolling buffer of recent live mic audio - only ever needs to hold one
    // segment's worth (segmentSamples, ~75ms - FIXED, not
    // sensitivity-dependent; the longer "sensitivity" memory lives in
    // emaAlpha below instead of in buffer length).
    this.texture = [];
    this.avgMagHalf = new Float64Array(this.halfLen);
    // Only ever needs to hold one segment's length ahead - at most two
    // segments (current tail, next head) ever overlap here, unlike a WOLA
    // scheme's several simultaneous frames.
    this.outBuffer = new Float32Array(this.segmentSamples);
    this._samplesUntilNextSegment = 0; // 0 => generate the very first segment immediately

    this.setSensitivitySeconds(sensitivitySeconds);
  }

  setSensitivitySeconds(seconds) {
    const clamped = Math.min(SSN_MAX_SENSITIVITY_S, Math.max(SSN_MIN_SENSITIVITY_S, seconds));
    this.sensitivitySeconds = clamped;
    // Standard EMA-from-time-constant: alpha=1 would track each segment's
    // own magnitude exactly (no memory at all); smaller alpha retains more
    // of the running average. Derived from HOP_SECONDS (how often the EMA
    // actually gets updated), not from nFft/sr. Governs both the noise's
    // spectral shape AND its overall level now (avgMagHalf feeds synthesis
    // as-is - see the module header), since level is no longer rescaled
    // to a fixed target separately.
    this.emaAlpha = 1 - Math.exp(-HOP_SECONDS / clamped);
  }

  reset() {
    this.texture = [];
    this.avgMagHalf.fill(0);
    this.outBuffer.fill(0);
    this._samplesUntilNextSegment = 0;
  }

  /** Feed one chunk of live mic audio (the same x stream.js's processChunk receives), keeping only the most recent segmentSamples. */
  feed(x) {
    for (let i = 0; i < x.length; i++) this.texture.push(x[i]);
    if (this.texture.length > this.segmentSamples) {
      this.texture.splice(0, this.texture.length - this.segmentSamples);
    }
  }

  _currentSnapshot() {
    if (this.texture.length === 0) return new Float64Array(this.segmentSamples);
    if (this.texture.length >= this.segmentSamples) {
      return Float64Array.from(this.texture.slice(this.texture.length - this.segmentSamples));
    }
    // Not enough real audio yet (startup) - tile what's there to fill the
    // snapshot rather than zero-padding, so the average magnitude isn't
    // artificially diluted by a padded-silence majority.
    const out = new Float64Array(this.segmentSamples);
    for (let i = 0; i < this.segmentSamples; i++) out[i] = this.texture[i % this.texture.length];
    return out;
  }

  _insertSegment() {
    const snapshot = this._currentSnapshot();
    const re = new Float64Array(this.nFft);
    const im = new Float64Array(this.nFft);
    re.set(snapshot); // zero-padded tail beyond segmentSamples - plain (unwindowed) analysis, matching calibrateSegmentRmsCorrection's own assumption
    fft(re, im);

    for (let k = 0; k < this.halfLen; k++) {
      const curMag = Math.hypot(re[k], im[k]);
      this.avgMagHalf[k] += this.emaAlpha * (curMag - this.avgMagHalf[k]);
    }

    const { re: outRe, im: outIm } = randomPhaseSpectrum(this.avgMagHalf, this.nFft, this.halfLen);
    ifft(outRe, outIm);

    const segment = new Float32Array(this.segmentSamples);
    for (let i = 0; i < this.segmentSamples; i++) segment[i] = outRe[i] * this.rmsCorrection;
    const faded = applyEqualPowerFade(segment, this.crossfadeSamples);
    for (let i = 0; i < faded.length; i++) this.outBuffer[i] += faded[i];
  }

  /**
   * Pull `frames` samples of continuous SSN output, generating/crossfading
   * a fresh segment internally every 50ms as needed. NOT yet scaled by the
   * "Noise" level slider or the "Concealer level" dB fader - stream.js
   * applies both when mixing this into playSum/recSum, so they stay live-
   * adjustable at any instant (unlike the noise's own spectral content,
   * which only updates once per 50ms tick).
   */
  nextBlock(frames) {
    if (this._samplesUntilNextSegment <= 0) {
      this._insertSegment();
      this._samplesUntilNextSegment += this.hopSamples;
    }
    const n = Math.min(frames, this.outBuffer.length);
    const block = new Float32Array(frames);
    for (let i = 0; i < n; i++) block[i] = this.outBuffer[i];
    this.outBuffer.copyWithin(0, frames);
    this.outBuffer.fill(0, Math.max(0, this.outBuffer.length - frames));
    this._samplesUntilNextSegment -= frames;
    return block;
  }
}
