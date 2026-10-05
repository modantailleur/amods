// "Concealing noise" - blurs a concealer clip's spectral fine structure
// (formant sharpness, transient edges) rather than adding broadband hiss,
// per this project's advisor: a 2D Gaussian blur across the STFT's time and
// frequency axes. Applied to candidate clips that already passed the memory
// branch's own VAD check, right before they're stored (see
// granspeechmask.js's _feedMemory) - never to the live query signal, same
// separation of concerns as the denoiser/de-esser.
//
// Implemented as a SEPARABLE Gaussian: a 1D blur along the frequency axis
// (within each frame) followed by a 1D blur along the time axis (across
// frames, per bin) - mathematically equivalent to a full 2D Gaussian
// convolution, far cheaper to compute. Phase is left untouched; only the
// magnitude spectrogram is blurred, which is what actually controls
// perceived sharpness/intelligibility - re-applying the original phase on
// reconstruction avoids phase-induced ringing a blurred-phase version would
// add on top of the intended effect.
import { fft, hannWindow } from './mel.js';
import { istft } from './stft.js';

const N_FFT = 2048; // ~43ms at 48kHz - a longer window than the de-esser's: this effect wants to blur sustained spectral shape (formants), not track short transients
const HOP = 512;
// Intensity 1.0 (top of the "Concealing noise" slider) maps to these sigmas
// (in bins/frames); intensity scales both linearly. NOT tuned by listening
// (no way to do that here) - reasoned from rough acoustic scale instead:
// at N_FFT=2048/HOP=512/48kHz, a bin is ~23Hz wide and a frame step is
// ~10.7ms, so 6 bins ~= 140Hz (in the ballpark of a typical formant
// bandwidth) and 4 frames ~= 43ms (roughly one phoneme's duration) - wide
// enough to blur formant/transient structure without erasing the whole
// spectral envelope outright. Treat these as a starting point to validate
// by ear, not a settled answer.
const MAX_FREQ_SIGMA_BINS = 128;
const MAX_TIME_SIGMA_FRAMES = 4;

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
 * what a "bin"/"frame" is worth in Hz/ms at N_FFT/HOP below) set directly
 * and independently, rather than both scaled together by one intensity
 * value. freqSigmaBins/timeSigmaFrames <= 0 skips blurring that axis.
 */
export function blurConcealerWithSigmas(x, sr, freqSigmaBins, timeSigmaFrames) {
  if (freqSigmaBins <= 0 && timeSigmaFrames <= 0) return x;

  // Zero-pad by a full N_FFT on each side before analysis, trimmed back off
  // after reconstruction (see this function's header comment below for why:
  // the WOLA overlap-add normalization divides by a near-zero window-squared
  // sum right at a frame's own edge, which only stays safe when the
  // reconstructed content there naturally tapers to match - true for an
  // unmodified round-trip, but broken once frequency blurring redistributes
  // energy into what used to be that taper. Padding keeps all genuine
  // content safely inside the fully-overlapped interior, where normSum is
  // stable, and confines the unstable edge region to padding we discard.
  const padded = new Float32Array(x.length + 2 * N_FFT);
  padded.set(x, N_FFT);

  const halfLen = N_FFT / 2 + 1; // bins 0..Nyquist inclusive - the canonical half, see convolveFreqAxisHalf
  const window = hannWindow(N_FFT);
  const nFrames = Math.max(1, Math.floor(Math.max(0, padded.length - N_FFT) / HOP) + 1);
  const frames = new Array(nFrames);
  const magHalf = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP;
    const re = new Float64Array(N_FFT);
    const im = new Float64Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
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
    for (let k = halfLen; k < N_FFT; k++) {
      const mirror = N_FFT - k;
      re[k] = re[mirror];
      im[k] = -im[mirror];
    }
  }

  const out = istft(frames, N_FFT, HOP, padded.length);
  return Float32Array.from(out.subarray(N_FFT, N_FFT + x.length));
}
