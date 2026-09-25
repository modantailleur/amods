// Direct port of src/amods/audio.py - keep in sync with that file.
// Small, dependency-free signal helpers shared across the pipeline.

/** Half-Hann fade-in ramp: 0 -> 1 with zero slope at both ends, unlike a linear ramp's sharp corners. */
function raisedCosineIn(fadeSize) {
  const ramp = new Float64Array(fadeSize);
  for (let i = 0; i < fadeSize; i++) {
    const t = fadeSize === 1 ? 0 : i / (fadeSize - 1);
    ramp[i] = 0.5 * (1.0 - Math.cos(Math.PI * t));
  }
  return ramp;
}

/** Apply a raised-cosine (half-Hann) fade-in and fade-out to both ends of x (Float32Array, not mutated). */
export function applyFade(x, fadeSize) {
  const n = x.length;
  if (fadeSize <= 0 || n < 2) return x;
  fadeSize = Math.min(fadeSize, Math.floor(n / 2));
  if (fadeSize <= 0) return x;
  const y = Float32Array.from(x);
  const ramp = raisedCosineIn(fadeSize);
  for (let i = 0; i < fadeSize; i++) {
    y[i] *= ramp[i];
    y[n - 1 - i] *= ramp[i];
  }
  return y;
}

/** Apply a raised-cosine (half-Hann) fade-in to the start of x (Float32Array, not mutated). */
export function applyFadeIn(x, fadeSize) {
  const n = x.length;
  if (fadeSize <= 0 || n < 2) return x;
  fadeSize = Math.min(fadeSize, Math.floor(n / 2));
  if (fadeSize <= 0) return x;
  const y = Float32Array.from(x);
  const ramp = raisedCosineIn(fadeSize);
  for (let i = 0; i < fadeSize; i++) y[i] *= ramp[i];
  return y;
}

/** Apply a raised-cosine (half-Hann) fade-out to the end of x (Float32Array, not mutated). */
export function applyFadeOut(x, fadeSize) {
  const n = x.length;
  if (fadeSize <= 0 || n < 2) return x;
  fadeSize = Math.min(fadeSize, Math.floor(n / 2));
  if (fadeSize <= 0) return x;
  const y = Float32Array.from(x);
  const ramp = raisedCosineIn(fadeSize);
  for (let i = 0; i < fadeSize; i++) y[n - 1 - i] *= ramp[i];
  return y;
}

/**
 * Scale x down so its peak absolute value never exceeds `limit`, using a
 * real attack/release envelope (like a hardware/plugin limiter) rather than
 * one flat scale factor per call. Unlike hard-clipping, this preserves the
 * waveform's shape - it scales rather than flattens - so it never actually
 * saturates/distorts, only ever gets quieter when it would have.
 *
 * Returns {y, newGain}; pass newGain back in as prevGain on the next call
 * over the same signal to keep the envelope continuous across an entire
 * stream (a fresh prevGain=1.0 starts it cleanly). `sr` must match the
 * signal's actual sample rate for attack/release times to mean what they say.
 *
 * See amods.audio.limit_peak (Python) for the full rationale - this is a
 * direct port, kept numerically equivalent.
 */
export function limitPeak(x, { limit = 0.95, prevGain = 1.0, sr = 48000, attackS = 0.002, releaseS = 0.15 } = {}) {
  const n = x.length;
  if (n === 0) return { y: x, newGain: prevGain };

  const attackCoeff = 1.0 - Math.exp(-1.0 / (Math.max(attackS, 1e-6) * sr));
  const releaseCoeff = 1.0 - Math.exp(-1.0 / (Math.max(releaseS, 1e-6) * sr));

  const gain = new Float64Array(n);
  let g = prevGain;
  let yPeak = 0;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const absXi = Math.abs(x[i]);
    const target = absXi > limit ? limit / Math.max(absXi, 1e-12) : 1.0;
    const coeff = target < g ? attackCoeff : releaseCoeff;
    g += coeff * (target - g);
    gain[i] = g;
    const yi = x[i] * g;
    y[i] = yi;
    const absYi = Math.abs(yi);
    if (absYi > yPeak) yPeak = absYi;
  }

  if (yPeak > limit) {
    const safety = limit / yPeak;
    for (let i = 0; i < n; i++) {
      y[i] *= safety;
      gain[i] *= safety;
    }
  }

  return { y, newGain: gain[n - 1] };
}
