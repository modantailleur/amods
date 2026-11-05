// Shared machinery for looping a real recorded clip as a "Noise controls"
// color - factored out once a SECOND real recording (fx_water.wav)
// needed the exact same WAV-parsing/crossfade-loop/calibration/variable-
// rate-playback logic water_flow.wav already had (see
// water-noise-generator.js/fx-water-noise-generator.js, both now thin
// wrappers around RecordedNoiseGenerator below, and docs/audios/CREDITS.md
// for both clips' source/license).
//
// Shares WhiteNoiseGenerator/PinkNoiseGenerator's { reset(), feed(),
// nextBlock(frames) } interface so stream.js can hold/swap every color
// uniformly - feed() is a no-op here too, same reasoning (no relationship
// to the live mic). nextBlock takes an extra optional pitchRatio argument
// white/pink don't have (see its own comment) - stream.js always passes
// it, white/pink's nextBlock(frames) signatures just ignore the extra
// argument, same as any JS function call with more arguments than
// declared parameters.
//
// Unlike the synthesized colors, this needs to load real audio - fetch()
// works fine in a Worker (where this runs, same as the rest of
// stream.js's world), but AudioContext.decodeAudioData does not exist
// there, so the WAV is parsed by hand below. That's only reasonable
// because every audioPath this is given is a file this project controls
// the exact shape of (plain 16-bit PCM, see CREDITS.md's ffmpeg commands)
// - this is not a general-purpose WAV decoder.
//
// Loading is async (fetch + parse) but nextBlock() is called synchronously
// from stream.js's processChunk, same as white/pink - so nextBlock just
// returns silence until the shared load finishes, then starts looping.
// The load is cached at module scope (by audioPath + sample rate) so
// switching the noise-type dropdown back and forth, or disabling/
// re-enabling Noise, never re-fetches or re-parses a given file more than
// once per sample rate.
import { resampleLinear } from './resample.js';
import { NOISE_COLOR_REFERENCE_RMS } from './noise-generators.js';

const sharedBufferPromises = new Map(); // `${audioPath}@${sr}` -> Promise<Float32Array>

function parseWavMono16(arrayBuffer, audioPath) {
  const view = new DataView(arrayBuffer);
  if (view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) {
    throw new Error(`${audioPath}: not a RIFF/WAVE file`);
  }
  let offset = 12;
  let sampleRate = null, numChannels = null, bitsPerSample = null, dataOffset = null, dataLength = null;
  while (offset + 8 <= view.byteLength) {
    const chunkId = view.getUint32(offset, false);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkId === 0x666d7420) { // 'fmt '
      numChannels = view.getUint16(chunkStart + 2, true);
      sampleRate = view.getUint32(chunkStart + 4, true);
      bitsPerSample = view.getUint16(chunkStart + 14, true);
    } else if (chunkId === 0x64617461) { // 'data'
      dataOffset = chunkStart;
      dataLength = chunkSize;
    }
    offset = chunkStart + chunkSize + (chunkSize % 2); // chunks are word-aligned
  }
  if (dataOffset === null || sampleRate === null) throw new Error(`${audioPath}: missing fmt/data chunk`);
  if (bitsPerSample !== 16) throw new Error(`${audioPath}: unsupported bit depth ${bitsPerSample} (expected 16)`);
  const bytesPerSample = 2 * numChannels;
  const nFrames = Math.floor(dataLength / bytesPerSample);
  const mono = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) sum += view.getInt16(dataOffset + i * bytesPerSample + ch * 2, true);
    mono[i] = sum / numChannels / 32768;
  }
  return { samples: mono, sampleRate };
}

// Same equal-power ramp shape as speech-shaped-noise.js's own
// equalPowerFadeRamp - duplicated rather than imported since that one
// isn't exported (small enough not to be worth changing just for this).
function equalPowerFadeRamp(fadeSize) {
  const ramp = new Float64Array(fadeSize);
  for (let i = 0; i < fadeSize; i++) {
    const t = fadeSize === 1 ? 0 : i / (fadeSize - 1);
    ramp[i] = Math.sin((Math.PI / 2) * t);
  }
  return ramp;
}

// Crossfades the clip's own tail into its own head so the result can be
// read as a circular buffer with no audible seam at the wrap point.
function buildLoopBuffer(samples, crossfadeSamples) {
  const n = samples.length;
  const fade = Math.min(crossfadeSamples, Math.floor(n / 2));
  const loopLen = n - fade;
  const out = Float32Array.from(samples.subarray(0, loopLen));
  const ramp = equalPowerFadeRamp(fade);
  for (let i = 0; i < fade; i++) {
    out[i] = samples[i] * ramp[i] + samples[loopLen + i] * ramp[fade - 1 - i];
  }
  return out;
}

function calibrate(buffer) {
  let sumSq = 0;
  for (let i = 0; i < buffer.length; i++) sumSq += buffer[i] * buffer[i];
  const rms = Math.sqrt(sumSq / buffer.length) || 1;
  const scale = NOISE_COLOR_REFERENCE_RMS / rms;
  const out = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) out[i] = buffer[i] * scale;
  return out;
}

async function loadSharedBuffer(audioPath, targetSr, loopCrossfadeSeconds) {
  const response = await fetch(audioPath);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText} (${audioPath})`);
  const arrayBuffer = await response.arrayBuffer();
  const { samples, sampleRate } = parseWavMono16(arrayBuffer, audioPath);
  const resampled = sampleRate === targetSr ? samples : resampleLinear(samples, sampleRate, targetSr);
  const looped = buildLoopBuffer(resampled, Math.round(loopCrossfadeSeconds * targetSr));
  return calibrate(looped);
}

export class RecordedNoiseGenerator {
  /** @param {object} opts - { audioPath, loopCrossfadeSeconds } */
  constructor(sr, { audioPath, loopCrossfadeSeconds }) {
    this.sr = sr;
    this.audioPath = audioPath;
    this.readPos = 0; // fractional (float) - see nextBlock's own comment for why
    this.buffer = null; // stays null (nextBlock returns silence) until the shared load resolves
    const key = `${audioPath}@${sr}`;
    if (!sharedBufferPromises.has(key)) sharedBufferPromises.set(key, loadSharedBuffer(audioPath, sr, loopCrossfadeSeconds));
    sharedBufferPromises.get(key)
      .then((buf) => { this.buffer = buf; })
      .catch((e) => console.error(`[RecordedNoiseGenerator] failed to load ${audioPath}`, e));
  }

  reset() {
    this.readPos = 0;
  }

  /** No-op - like white/pink noise, this has no relationship to the live mic. */
  feed() {}

  /**
   * pitchRatio (default 1, normal speed/pitch): advances the read position
   * by this many source samples per output sample instead of exactly 1,
   * with linear interpolation between the two samples straddled by the
   * resulting fractional position - the standard "variable-rate
   * resampling" way to pitch-shift (every frequency in the buffer scales
   * by pitchRatio; a playback-rate change, not a frequency-domain shift).
   * See stream.js's "Match source" comment for where this ratio comes
   * from and why shifting rate - not EQ/filtering - is what actually
   * moves the spectral centroid by a known, invertible amount.
   */
  nextBlock(frames, pitchRatio = 1) {
    const out = new Float32Array(frames);
    const buf = this.buffer;
    if (!buf) return out; // silence until loaded
    const n = buf.length;
    let pos = this.readPos;
    for (let i = 0; i < frames; i++) {
      const i0 = Math.floor(pos);
      const i1 = (i0 + 1) % n;
      const frac = pos - i0;
      out[i] = buf[i0] * (1 - frac) + buf[i1] * frac;
      pos += pitchRatio;
      if (pos >= n) pos -= n * Math.floor(pos / n);
      else if (pos < 0) pos += n * Math.ceil(-pos / n);
    }
    this.readPos = pos;
    return out;
  }
}
