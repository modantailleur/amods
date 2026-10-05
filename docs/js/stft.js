// Shared STFT/ISTFT building blocks, factored out of deesser.js once a
// second feature (concealer-noise.js) needed the same machinery. Built on
// mel.js's forward FFT rather than duplicating an FFT implementation.
import { fft, hannWindow } from './mel.js';

/** In-place inverse FFT via the conjugate trick, reusing mel.js's forward fft(). */
export function ifft(re, im) {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

/** x: Float32Array/Float64Array. Returns an array of {re, im} Float64Array(nFft) frames. */
export function stft(x, nFft, hop) {
  const window = hannWindow(nFft);
  const nFrames = Math.max(1, Math.floor(Math.max(0, x.length - nFft) / hop) + 1);
  const frames = new Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    const start = t * hop;
    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);
    for (let i = 0; i < nFft; i++) {
      const s = start + i < x.length ? x[start + i] : 0;
      re[i] = s * window[i];
    }
    fft(re, im);
    frames[t] = { re, im };
  }
  return frames;
}

/** Weighted overlap-add reconstruction - window applied at both analysis and
 * synthesis, normalized by the sum of window^2, which is self-correcting for
 * COLA rather than relying on an exact window/hop combination. */
export function istft(frames, nFft, hop, outLength) {
  const window = hannWindow(nFft);
  const out = new Float64Array(outLength);
  const normSum = new Float64Array(outLength);
  const reBuf = new Float64Array(nFft);
  const imBuf = new Float64Array(nFft);
  for (let t = 0; t < frames.length; t++) {
    reBuf.set(frames[t].re);
    imBuf.set(frames[t].im);
    ifft(reBuf, imBuf);
    const start = t * hop;
    for (let i = 0; i < nFft; i++) {
      const idx = start + i;
      if (idx >= outLength) break;
      out[idx] += reBuf[i] * window[i];
      normSum[idx] += window[i] * window[i];
    }
  }
  for (let i = 0; i < outLength; i++) out[i] = normSum[i] > 1e-10 ? out[i] / normSum[i] : 0;
  return out;
}

/** Bin k's represented frequency, folding the upper (mirror/negative-frequency) half back onto the positive range - fft()'s output is a full nFft-length conjugate-symmetric spectrum, not an rfft. */
export function binFreqHz(k, nFft, sr) {
  const kEff = k <= nFft / 2 ? k : nFft - k;
  return (kEff * sr) / nFft;
}

/** Linear-interpolated percentile, matching numpy's default. */
export function percentile(sortedArr, p) {
  const n = sortedArr.length;
  if (n === 0) return 0;
  if (n === 1) return sortedArr[0];
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
}
