// Offline, non-causal split-band de-esser, prototyped and tuned in Python
// against audios/test.wav (see conversation history - no script survives in
// this repo, the parameters below are the result of that iteration) before
// being ported here. Runs on the full ~2s pendingVoice snapshot GranSpeechMask
// feeds into memory (see granspeechmask.js's _feedMemory), NOT on live
// real-time audio - that's what makes the non-causal approach possible at
// all: a real-time de-esser can only ever look a few ms ahead (lookahead
// buffering), but this one sees the whole clip up front and can smooth its
// detection symmetrically (forward AND backward in time), which is what a
// tool like iZotope RX does and a live plugin cannot.
//
// Algorithm: take an STFT, measure energy in a "detect" band per frame
// (6-18kHz - confirmed, by listening to the isolated removed content during
// tuning, to reliably track genuine sibilant ("s"/"sh") moments), smooth that
// curve symmetrically, gate it against a threshold computed from the clip's
// OWN energy distribution (a percentile, not a hardcoded dB number - every
// attempt at a fixed threshold during tuning needed re-guessing per
// recording level), then attenuate a wider "suppress" band (3-18kHz - real
// sibilant energy extends lower than the detect band alone, confirmed
// during tuning that suppressing only 6-18kHz left an audible residual) by
// up to a fixed dB amount wherever the gate is active. Reconstructed via
// ISTFT (windowed overlap-add).
import { stft, istft, binFreqHz, percentile } from './stft.js';

const N_FFT = 512; // ~10.7ms at 48kHz - short enough to not blur a brief "s" together with the vowel/consonant next to it (a 2048-sample/~43ms window, tried first, smeared the two together)
const HOP = 128; // 75% overlap, satisfies COLA with the Hann window used for both analysis and synthesis below
const DETECT_BAND_HZ = [6000, 18000];
const SUPPRESS_BAND_HZ = [3000, 18000];
const THRESHOLD_PERCENTILE = 82;
const MAX_REDUCTION_DB = 35;
const SMOOTH_FRAMES = 2;
const STEEPNESS = 1.5;

/**
 * x: Float32Array of audio in [-1, 1] at sr. Returns a new Float32Array of
 * the same length with sibilant moments suppressed. Intended to run once,
 * on a whole known clip, before denoising (see worker-engine.js's FbdeDM
 * option) - NOT per real-time chunk.
 */
export function deEss(x, sr) {
  const frames = stft(x, N_FFT, HOP);
  const nFrames = frames.length;

  const detectIdx = new Uint8Array(N_FFT);
  const suppressIdx = new Uint8Array(N_FFT);
  for (let k = 0; k < N_FFT; k++) {
    const fHz = binFreqHz(k, N_FFT, sr);
    detectIdx[k] = fHz >= DETECT_BAND_HZ[0] && fHz <= DETECT_BAND_HZ[1] ? 1 : 0;
    suppressIdx[k] = fHz >= SUPPRESS_BAND_HZ[0] && fHz <= SUPPRESS_BAND_HZ[1] ? 1 : 0;
  }

  const bandDb = new Float64Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    const { re, im } = frames[t];
    let energy = 0;
    for (let k = 0; k < N_FFT; k++) {
      if (detectIdx[k]) energy += re[k] * re[k] + im[k] * im[k];
    }
    bandDb[t] = 10 * Math.log10(energy + 1e-15);
  }

  const thresholdDb = percentile(Float64Array.from(bandDb).sort(), THRESHOLD_PERCENTILE);

  // Symmetric (non-causal) smoothing - a centered moving average, valid
  // only because the whole clip is already known; edges clamp to the
  // nearest in-range frame rather than zero-padding.
  const half = Math.floor(SMOOTH_FRAMES / 2);
  const smoothed = new Float64Array(nFrames);
  for (let t = 0; t < nFrames; t++) {
    let sum = 0;
    let count = 0;
    for (let d = -half; d <= SMOOTH_FRAMES - 1 - half; d++) {
      const idx = Math.max(0, Math.min(nFrames - 1, t + d));
      sum += bandDb[idx];
      count++;
    }
    smoothed[t] = sum / count;
  }

  for (let t = 0; t < nFrames; t++) {
    const gate = 1 / (1 + Math.exp(-(smoothed[t] - thresholdDb) * STEEPNESS));
    const gainLin = 10 ** ((-gate * MAX_REDUCTION_DB) / 20);
    const { re, im } = frames[t];
    for (let k = 0; k < N_FFT; k++) {
      if (suppressIdx[k]) {
        re[k] *= gainLin;
        im[k] *= gainLin;
      }
    }
  }

  const out = istft(frames, N_FFT, HOP, x.length);
  return Float32Array.from(out);
}
