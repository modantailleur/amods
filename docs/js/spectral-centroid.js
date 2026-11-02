// Magnitude-weighted spectral centroid - the standard "brightness" measure
// (same convention librosa's spectral_centroid uses): sum(f * |X(f)|) /
// sum(|X(f)|) over a real-input FFT's positive-frequency half. Used both to
// precompute water_flow.wav's own reference centroid once (see
// water-noise-generator.js's WATER_NATIVE_CENTROID_HZ - see that constant's
// own comment for how/where it was computed) and, every chunk, to track the
// live source's centroid for the "Match source" feature (see stream.js's
// processChunk).
//
// Plain radix-2 Cooley-Tukey FFT below (the Web Audio API's AnalyserNode
// isn't available in a Worker, where this runs) - cheap enough at this
// app's chunk size (zero-padded to the next power of 2, a few thousand
// samples) to run every 50ms chunk alongside the VAD/denoiser work already
// happening there.
function fft(re, im, invert) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (invert ? 1 : -1) * (2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe; curIm = nextIm;
      }
    }
  }
  if (invert) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/**
 * x: Float32Array/Float64Array, real-valued, any length (zero-padded to
 * the next power of 2 internally). Returns the magnitude-weighted spectral
 * centroid in Hz, or 0 for an all-silent/empty input (sum(|X(f)|) === 0).
 */
export function computeSpectralCentroid(x, sr) {
  const n = x.length;
  if (n === 0) return 0;
  let n2 = 1;
  while (n2 < n) n2 <<= 1;
  const re = new Float64Array(n2);
  const im = new Float64Array(n2);
  re.set(x);
  fft(re, im, false);
  const half = n2 / 2;
  let num = 0;
  let den = 0;
  for (let k = 0; k <= half; k++) {
    const mag = Math.hypot(re[k], im[k]);
    num += ((k * sr) / n2) * mag;
    den += mag;
  }
  return den > 0 ? num / den : 0;
}
