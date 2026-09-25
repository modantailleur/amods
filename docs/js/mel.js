// From-scratch mel-spectrogram feature extractor, approximating
// librosa.feature.melspectrogram(n_mels=32, fmin=100, fmax=6000) +
// librosa.power_to_db(ref=1.0).mean(axis=1), as used by
// GranSpeechMaskCM._feature_extractor (base.py) to fingerprint clips for
// cosine-distance matching. There is no browser equivalent of librosa, so
// this is a genuine reimplementation, not a mechanical port - it is NOT
// bit-identical to librosa (no center/reflect-padding, a plain radix-2
// FFT, and top_db postprocessing is skipped), but reproduces the same
// Slaney-style mel filterbank and log-power averaging, which is what this
// feature vector is actually used for: a *relative* (cosine-similarity)
// fingerprint, not an absolute physical measurement.

const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_MELS = 32;
const FMIN = 100;
const FMAX = 6000;

function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** In-place iterative radix-2 Cooley-Tukey FFT; re/im are Float64Array of length n (a power of 2). */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len / 2;
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }
}

// Slaney-style Hz<->mel conversion, matching librosa's default (htk=False).
function hzToMel(hz) {
  const fMin = 0, fSp = 200 / 3;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  const logstep = Math.log(6.4) / 27.0;
  let mel = (hz - fMin) / fSp;
  if (hz >= minLogHz) mel = minLogMel + Math.log(hz / minLogHz) / logstep;
  return mel;
}

function melToHz(mel) {
  const fMin = 0, fSp = 200 / 3;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  const logstep = Math.log(6.4) / 27.0;
  let hz = fMin + fSp * mel;
  if (mel >= minLogMel) hz = minLogHz * Math.exp(logstep * (mel - minLogMel));
  return hz;
}

function melFilterbank(sr, nFft, nMels, fmin, fmax) {
  const nFreqs = nFft / 2 + 1;
  const fftFreqs = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) fftFreqs[i] = (i * sr) / nFft;

  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melPoints = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) melPoints[i] = melMin + (i * (melMax - melMin)) / (nMels + 1);
  const hzPoints = Array.from(melPoints, melToHz);

  // Slaney-style area-normalized triangular filters.
  const weights = [];
  for (let m = 0; m < nMels; m++) {
    const fLeft = hzPoints[m], fCenter = hzPoints[m + 1], fRight = hzPoints[m + 2];
    const w = new Float64Array(nFreqs);
    const enorm = 2.0 / (fRight - fLeft);
    for (let i = 0; i < nFreqs; i++) {
      const f = fftFreqs[i];
      let val = 0;
      if (f >= fLeft && f <= fCenter) val = (f - fLeft) / (fCenter - fLeft);
      else if (f > fCenter && f <= fRight) val = (fRight - f) / (fRight - fCenter);
      w[i] = Math.max(0, val) * enorm;
    }
    weights.push(w);
  }
  return weights;
}

const filterbankCache = new Map();
function getFilterbank(sr) {
  if (!filterbankCache.has(sr)) {
    filterbankCache.set(sr, melFilterbank(sr, N_FFT, N_MELS, FMIN, FMAX));
  }
  return filterbankCache.get(sr);
}

/**
 * Reduce x (Float32Array/Float64Array) to a 32-dim log-mel spectral
 * fingerprint (mean over time). Set norm=true to peak-normalize x first,
 * so loudness differences don't affect the match - matches
 * GranSpeechMaskCM._feature_extractor's contract.
 */
export function featureExtractor(x, sr, { norm = false } = {}) {
  let sig = x;
  if (norm) {
    let peak = 0;
    for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    const scale = 1 / (peak + 1e-10);
    sig = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) sig[i] = x[i] * scale;
  }

  const window = hannWindow(N_FFT);
  const filterbank = getFilterbank(sr);
  const nFreqs = N_FFT / 2 + 1;
  const nFrames = Math.max(1, Math.floor(Math.max(0, sig.length - N_FFT) / HOP_LENGTH) + 1);
  const melSum = new Float64Array(N_MELS);

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    im.fill(0);
    for (let i = 0; i < N_FFT; i++) {
      const s = start + i < sig.length ? sig[start + i] : 0;
      re[i] = s * window[i];
    }
    fft(re, im);

    for (let m = 0; m < N_MELS; m++) {
      let e = 0;
      const w = filterbank[m];
      for (let i = 0; i < nFreqs; i++) e += w[i] * (re[i] * re[i] + im[i] * im[i]);
      // power_to_db(ref=1.0), top_db clamping skipped (see module docstring).
      melSum[m] += 10 * Math.log10(Math.max(e, 1e-10));
    }
  }

  const feat = new Float64Array(N_MELS);
  for (let m = 0; m < N_MELS; m++) feat[m] = melSum[m] / nFrames;
  return feat;
}
