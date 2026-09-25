// Linear-interpolation resampler, used everywhere amods.py calls
// librosa.resample. Not bit-identical to librosa (which uses a
// higher-quality bandlimited resampler) - a known, deliberate
// simplification for this browser port. Good enough for VAD (which only
// needs a coarse speech/non-speech signal) and adequate, if not pristine,
// for the denoiser; revisit if denoised audio quality suffers.
export function resampleLinear(x, srcRate, dstRate) {
  if (srcRate === dstRate) return Float32Array.from(x);
  const srcLen = x.length;
  const dstLen = Math.max(1, Math.round((srcLen * dstRate) / srcRate));
  const y = new Float32Array(dstLen);
  const scale = (srcLen - 1) / Math.max(1, dstLen - 1);
  for (let i = 0; i < dstLen; i++) {
    const srcPos = i * scale;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, srcLen - 1);
    const frac = srcPos - i0;
    y[i] = x[i0] * (1 - frac) + x[i1] * frac;
  }
  return y;
}
