// "Concealing noise" - two alternative techniques, both applied to
// candidate clips that already passed the memory branch's own VAD check,
// right before they're stored (see granspeechmask.js's _feedMemory) -
// never to the live query signal, same separation of concerns as the
// denoiser/de-esser. Switch between them by changing CONCEALING_NOISE_MODE
// below - nothing else needs to change.
//
// 'blur' (blurConcealer) - a 2D Gaussian blur across the STFT's time and
// frequency axes, per this project's advisor. Implemented as a SEPARABLE
// Gaussian (1D blur along frequency, then along time - equivalent to full
// 2D, far cheaper). Phase is left untouched; only magnitude is blurred.
// Sounded "vocoded" in practice - the fine harmonic structure survives,
// just smeared, so it still reads as a (blurry) voice.
//
// 'noiseShape' (noiseShapeConcealer) - closer to what's called
// "noise-vocoded speech" in psychoacoustics research: keep the clip's
// magnitude spectrogram exactly as-is (same loudness-over-time, same
// spectral coloring at each moment) but discard its phase, replacing it
// with random phase. The envelope/"shape" survives; the harmonic/tonal
// fine structure that makes something sound voiced does not - the result
// is textured, noise-like, but dynamically follows the original speech.
const CONCEALING_NOISE_MODE = 'noiseShape'; // 'blur' | 'noiseShape' - the one variable to flip to switch techniques
import { fft, hannWindow } from './mel.js';
import { istft } from './stft.js';

// blurConcealer's own STFT config.
const BLUR_N_FFT = 2048; // ~43ms at 48kHz - a longer window than the de-esser's: this effect wants to blur sustained spectral shape (formants), not track short transients
const BLUR_HOP = 512;
// Intensity 1.0 (top of the "Concealing noise" slider) maps to these sigmas
// (in bins/frames); intensity scales both linearly. NOT tuned by listening
// (no way to do that here) - reasoned from rough acoustic scale instead:
// at BLUR_N_FFT=2048/BLUR_HOP=512/48kHz, a bin is ~23Hz wide and a frame
// step is ~10.7ms, so 6 bins ~= 140Hz (in the ballpark of a typical formant
// bandwidth) and 4 frames ~= 43ms (roughly one phoneme's duration) - wide
// enough to blur formant/transient structure without erasing the whole
// spectral envelope outright. Treat these as a starting point to validate
// by ear, not a settled answer.
const MAX_FREQ_SIGMA_BINS = 128;
const MAX_TIME_SIGMA_FRAMES = 4;

// noiseShapeConcealer's own STFT config - deliberately BIGGER than the
// blur's, by explicit request: a bigger window gives finer frequency
// resolution (which ENVELOPE_SMOOTH_SIGMA_BINS's smoothing then coarsens
// back down, erasing harmonics - see that constant's comment) but ALSO
// coarser time resolution as a side effect (each frame spans a longer
// stretch, so fast-changing content within it blurs together) - normally
// a pure cost with no benefit, EXCEPT here that coarser time resolution is
// itself wanted ("a tiny bit more coarse"), not just tolerated.
const NOISE_SHAPE_N_FFT = 4096; // ~85ms at 48kHz - 2x blurConcealer's window
const NOISE_SHAPE_HOP = 1024; // keeps the same hop/window ratio (COLA) as the blur's 512/2048
// Same absolute Hz-width as before (469Hz), recomputed for this bigger
// window's finer bin resolution (~11.7Hz/bin here vs ~23Hz/bin for the
// blur's smaller window) - see the original reasoning below: needs to be
// wider than the spacing between a voice's harmonics (adjacent harmonics
// are F0 apart, and adult voices commonly run F0 ~85-300Hz) so they
// actually merge into a smooth envelope rather than surviving as a comb of
// narrow peaks, while staying narrower than typical formant separation
// (several hundred Hz to over 1kHz) so broad spectral coloring survives.
// Not tuned by listening (same caveat as MAX_FREQ_SIGMA_BINS above).
const ENVELOPE_SMOOTH_SIGMA_BINS = 40;

function gaussianKernel(sigma) {
  if (sigma <= 0) return [1];
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return kernel;
}

/** Time-axis (across frames, same bin) convolution of a 2D [frame][bin] magnitude array, edges clamped (replicate). Never crosses the DC/Nyquist mirror boundary, so no symmetry concern here (see convolveFreqAxisHalf's comment for why the frequency axis needs different handling). */
function convolveTimeAxis(mag, kernel) {
  const nFrames = mag.length;
  const nBins = mag[0].length;
  const radius = (kernel.length - 1) / 2;
  const out = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) out[t] = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    for (let t = 0; t < nFrames; t++) {
      let sum = 0;
      for (let d = -radius; d <= radius; d++) {
        const idx = Math.max(0, Math.min(nFrames - 1, t + d));
        sum += mag[idx][k] * kernel[d + radius];
      }
      out[t][k] = sum;
    }
  }
  return out;
}

/**
 * Frequency-axis blur, operating ONLY on the canonical half-spectrum
 * (bins 0..N_FFT/2, i.e. DC to Nyquist). fft()'s output is a FULL
 * nFft-length conjugate-symmetric spectrum (not an rfft) - bin k and bin
 * N-k are mirror images (equal magnitude) because the input is real. A
 * naive 1D convolution across the full 0..N-1 range with edge-clamping
 * treats that as one flat linear sequence and breaks the mirror relationship
 * right at the DC and Nyquist boundaries, which corrupts conjugate symmetry
 * and - discovered the hard way - caused a ~40x energy blowup on
 * reconstruction (a non-conjugate-symmetric spectrum's inverse FFT is
 * genuinely complex; keeping only the real part silently scrambles the
 * energy accounting). Fixed by blurring only the half-spectrum with
 * REFLECTIVE boundaries at both ends (bin -1 reflects to bin 1, not a
 * repeat of bin 0; same at Nyquist) - the caller mirrors the result back
 * onto the upper half afterward, guaranteeing symmetry by construction
 * rather than hoping the convolution preserves it.
 */
function convolveFreqAxisHalf(magHalf, kernel) {
  const nFrames = magHalf.length;
  const halfLen = magHalf[0].length; // N_FFT/2 + 1 (bins 0..Nyquist inclusive)
  const radius = (kernel.length - 1) / 2;
  const out = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) out[t] = new Float64Array(halfLen);
  for (let t = 0; t < nFrames; t++) {
    const row = magHalf[t];
    const outRow = out[t];
    for (let k = 0; k < halfLen; k++) {
      let sum = 0;
      for (let d = -radius; d <= radius; d++) {
        let idx = k + d;
        // Reflect (not clamp) at both ends - index -1 maps to 1, index
        // halfLen maps to halfLen-2, matching the true periodic/mirror
        // structure of the spectrum at DC and Nyquist.
        if (idx < 0) idx = -idx;
        if (idx >= halfLen) idx = 2 * (halfLen - 1) - idx;
        idx = Math.max(0, Math.min(halfLen - 1, idx));
        sum += row[idx] * kernel[d + radius];
      }
      outRow[k] = sum;
    }
  }
  return out;
}

/**
 * x: Float32Array of audio in [-1, 1] at sr. intensity: 0..1, 0 = no change
 * (the "Concealing noise" slider's default). Returns a new Float32Array of
 * the same length. Thin wrapper around blurConcealerWithSigmas - FREQUENCY
 * ONLY for now (time sigma pinned to 0), by explicit preference after
 * comparing both axes by ear: frequency-only blurring (vaguer vowel
 * timbre/formants, crisp timing) was clearly preferred over smearing time
 * (vaguer transients/syllable boundaries). MAX_TIME_SIGMA_FRAMES is left
 * defined and blurConcealerWithSigmas still takes both axes independently,
 * so time blurring can be re-enabled here later without reworking anything.
 */
export function blurConcealer(x, sr, intensity) {
  if (intensity <= 0) return x;
  return blurConcealerWithSigmas(x, sr, intensity * MAX_FREQ_SIGMA_BINS, 0);
}

/**
 * Same effect as blurConcealer, but with the frequency and time blur
 * widths (in FFT bins / STFT frames - see this module's header comment for
 * what a "bin"/"frame" is worth in Hz/ms at BLUR_N_FFT/BLUR_HOP above) set
 * directly and independently, rather than both scaled together by one
 * intensity value. freqSigmaBins/timeSigmaFrames <= 0 skips blurring that axis.
 */
export function blurConcealerWithSigmas(x, sr, freqSigmaBins, timeSigmaFrames) {
  if (freqSigmaBins <= 0 && timeSigmaFrames <= 0) return x;

  // Zero-pad by a full BLUR_N_FFT on each side before analysis, trimmed back
  // off after reconstruction (see this function's header comment below for
  // why: the WOLA overlap-add normalization divides by a near-zero
  // window-squared sum right at a frame's own edge, which only stays safe
  // when the reconstructed content there naturally tapers to match - true
  // for an unmodified round-trip, but broken once frequency blurring
  // redistributes energy into what used to be that taper. Padding keeps all
  // genuine content safely inside the fully-overlapped interior, where
  // normSum is stable, and confines the unstable edge region to padding we
  // discard.
  const padded = new Float32Array(x.length + 2 * BLUR_N_FFT);
  padded.set(x, BLUR_N_FFT);

  const halfLen = BLUR_N_FFT / 2 + 1; // bins 0..Nyquist inclusive - the canonical half, see convolveFreqAxisHalf
  const window = hannWindow(BLUR_N_FFT);
  const nFrames = Math.max(1, Math.floor(Math.max(0, padded.length - BLUR_N_FFT) / BLUR_HOP) + 1);
  const frames = new Array(nFrames);
  const magHalf = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    const start = t * BLUR_HOP;
    const re = new Float64Array(BLUR_N_FFT);
    const im = new Float64Array(BLUR_N_FFT);
    for (let i = 0; i < BLUR_N_FFT; i++) {
      const s = start + i < padded.length ? padded[start + i] : 0;
      re[i] = s * window[i];
    }
    fft(re, im);
    const m = new Float64Array(halfLen);
    for (let k = 0; k < halfLen; k++) m[k] = Math.hypot(re[k], im[k]);
    frames[t] = { re, im };
    magHalf[t] = m;
  }

  const freqKernel = gaussianKernel(freqSigmaBins);
  const timeKernel = gaussianKernel(timeSigmaFrames);
  let blurredHalf = convolveFreqAxisHalf(magHalf, freqKernel);
  blurredHalf = convolveTimeAxis(blurredHalf, timeKernel);

  // Re-apply the blurred magnitude, keeping each frame's original phase -
  // reconstructed directly from the phase ANGLE, not as a ratio of the
  // blurred to original magnitude: that ratio explodes near spectral nulls,
  // where the original magnitude is near-zero but the blur has pulled in
  // real energy from neighboring strong bins. Only bins 0..Nyquist are
  // touched directly; the upper (mirror) half is then set explicitly from
  // the lower half's conjugate, which is what actually guarantees the
  // reconstructed spectrum stays conjugate-symmetric (see
  // convolveFreqAxisHalf's comment for why that matters).
  for (let t = 0; t < nFrames; t++) {
    const { re, im } = frames[t];
    for (let k = 0; k < halfLen; k++) {
      const originalMag = magHalf[t][k];
      const angle = originalMag > 1e-12 ? Math.atan2(im[k], re[k]) : 0;
      const newMag = blurredHalf[t][k];
      re[k] = newMag * Math.cos(angle);
      im[k] = newMag * Math.sin(angle);
    }
    for (let k = halfLen; k < BLUR_N_FFT; k++) {
      const mirror = BLUR_N_FFT - k;
      re[k] = re[mirror];
      im[k] = -im[mirror];
    }
  }

  const out = istft(frames, BLUR_N_FFT, BLUR_HOP, padded.length);
  return Float32Array.from(out.subarray(BLUR_N_FFT, BLUR_N_FFT + x.length));
}

/**
 * x: Float32Array of audio in [-1, 1] at sr. intensity: 0..1, 0 = original
 * unchanged, 1 = fully replaced by envelope-shaped noise. Reconstructs from
 * the original's STFT magnitude with RANDOM phase per bin per frame (no
 * smoothing across frames needed - the overlap-add itself, plus the
 * shared magnitude envelope, is what keeps it sounding continuous rather
 * than choppy), then crossfades that against the original waveform in the
 * time domain (blending phase angles directly wouldn't be meaningful,
 * since phase is circular). envelopeSigmaBins overrides
 * ENVELOPE_SMOOTH_SIGMA_BINS - exposed as a parameter (not just the module
 * constant) so the width can be explored directly, same reasoning as
 * blurConcealerWithSigmas above.
 */
export function noiseShapeConcealer(x, sr, intensity, envelopeSigmaBins = ENVELOPE_SMOOTH_SIGMA_BINS) {
  if (intensity <= 0) return x;

  // Same padding reasoning as blurConcealer above: replacing phase, just
  // like redistributing magnitude, breaks the "this frame's own content
  // naturally tapers to zero at its edges" property the WOLA overlap-add
  // normalization depends on near a frame's own boundary.
  const padded = new Float32Array(x.length + 2 * NOISE_SHAPE_N_FFT);
  padded.set(x, NOISE_SHAPE_N_FFT);

  const halfLen = NOISE_SHAPE_N_FFT / 2 + 1;
  const window = hannWindow(NOISE_SHAPE_N_FFT);
  const nFrames = Math.max(1, Math.floor(Math.max(0, padded.length - NOISE_SHAPE_N_FFT) / NOISE_SHAPE_HOP) + 1);
  const frames = new Array(nFrames);
  const magHalf = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    const start = t * NOISE_SHAPE_HOP;
    const re = new Float64Array(NOISE_SHAPE_N_FFT);
    const im = new Float64Array(NOISE_SHAPE_N_FFT);
    for (let i = 0; i < NOISE_SHAPE_N_FFT; i++) {
      const s = start + i < padded.length ? padded[start + i] : 0;
      re[i] = s * window[i];
    }
    fft(re, im);
    const m = new Float64Array(halfLen);
    for (let k = 0; k < halfLen; k++) m[k] = Math.hypot(re[k], im[k]);
    magHalf[t] = m;
  }

  // Randomizing phase alone does NOT make a voiced sound stop sounding
  // pitched/vocoded - pitch perception is driven almost entirely by the
  // MAGNITUDE spectrum's sharp, narrow peaks at the voice's harmonics, not
  // by phase coherence between them (random phase leaves those peaks
  // exactly where they were, every frame, so the ear still hears a stable
  // narrowband tone there). Genuine noise-vocoded speech (the psychoacoustics
  // technique this is modeled on) avoids that by using a small number of
  // BROAD frequency bands rather than fine per-bin resolution - the coarse
  // resolution itself is what erases individual harmonics, leaving only the
  // broad formant envelope. ENVELOPE_SMOOTH_SIGMA_BINS reproduces that here:
  // a wide enough frequency-axis blur (reusing convolveFreqAxisHalf) to
  // merge adjacent harmonics together before using the result as the
  // noise's shaping envelope, rather than using the raw per-bin magnitude.
  const smoothedHalf = convolveFreqAxisHalf(magHalf, gaussianKernel(envelopeSigmaBins));

  for (let t = 0; t < nFrames; t++) {
    const re2 = new Float64Array(NOISE_SHAPE_N_FFT);
    const im2 = new Float64Array(NOISE_SHAPE_N_FFT);
    for (let k = 0; k < halfLen; k++) {
      const mag = smoothedHalf[t][k];
      const angle = Math.random() * 2 * Math.PI;
      re2[k] = mag * Math.cos(angle);
      im2[k] = mag * Math.sin(angle);
    }
    for (let k = halfLen; k < NOISE_SHAPE_N_FFT; k++) {
      const mirror = NOISE_SHAPE_N_FFT - k;
      re2[k] = re2[mirror];
      im2[k] = -im2[mirror];
    }
    frames[t] = { re: re2, im: im2 };
  }

  const noiseOut = istft(frames, NOISE_SHAPE_N_FFT, NOISE_SHAPE_HOP, padded.length);
  const noiseTrimmed = noiseOut.subarray(NOISE_SHAPE_N_FFT, NOISE_SHAPE_N_FFT + x.length);

  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (1 - intensity) * x[i] + intensity * noiseTrimmed[i];
  return out;
}

/** Entry point granspeechmask.js actually calls - dispatches to whichever technique CONCEALING_NOISE_MODE above selects. */
export function applyConcealingNoise(x, sr, intensity) {
  return CONCEALING_NOISE_MODE === 'noiseShape' ? noiseShapeConcealer(x, sr, intensity) : blurConcealer(x, sr, intensity);
}
