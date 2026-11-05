// Convolves the dry bubbles (synth_bubbles.mjs's output) with a REAL
// recorded impulse response (not an algorithmic/synthetic one) -
// "St Nicolaes Church.wav" from Aleksey Vaneev's (Voxengo) freely-licensed
// impulse pack. See this directory's own README.md for where that file
// needs to live locally (not committed to this repo - see the license
// note there) and for how to run this end to end.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));

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

// Generic WAV reader - handles mono or stereo, any bit depth among
// 16/24/32-bit PCM, any sample rate. Downmixes multi-channel to mono.
function readWav(filePath) {
  const buf = fs.readFileSync(filePath);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let offset = 12, sampleRate = null, numChannels = null, bitsPerSample = null, dataOff = null, dataLen = null;
  while (offset + 8 <= view.byteLength) {
    const id = view.getUint32(offset, false);
    const size = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (id === 0x666d7420) {
      numChannels = view.getUint16(start + 2, true);
      sampleRate = view.getUint32(start + 4, true);
      bitsPerSample = view.getUint16(start + 14, true);
    } else if (id === 0x64617461) { dataOff = start; dataLen = size; }
    offset = start + size + (size % 2);
  }
  const bytesPerSample = bitsPerSample / 8;
  const nFrames = Math.floor(dataLen / (bytesPerSample * numChannels));
  const mono = new Float64Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const sampleOff = dataOff + i * bytesPerSample * numChannels + ch * bytesPerSample;
      let v;
      if (bitsPerSample === 16) v = view.getInt16(sampleOff, true) / 32768;
      else if (bitsPerSample === 24) {
        const b0 = view.getUint8(sampleOff), b1 = view.getUint8(sampleOff + 1), b2 = view.getUint8(sampleOff + 2);
        let raw = b0 | (b1 << 8) | (b2 << 16);
        if (raw & 0x800000) raw -= 0x1000000;
        v = raw / 8388608;
      } else if (bitsPerSample === 32) v = view.getInt32(sampleOff, true) / 2147483648;
      else throw new Error(`${filePath}: unsupported bit depth ${bitsPerSample}`);
      sum += v;
    }
    mono[i] = sum / numChannels;
  }
  return { samples: mono, sampleRate };
}

function resampleLinear(x, srcRate, dstRate) {
  if (srcRate === dstRate) return Float64Array.from(x);
  const srcLen = x.length;
  const dstLen = Math.max(1, Math.round((srcLen * dstRate) / srcRate));
  const y = new Float64Array(dstLen);
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

function writeWavMono(filePath, mono, sr) {
  const n = mono.length;
  const dataSize = n * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0, 'ascii'); buf.writeUInt32LE(36 + dataSize, 4); buf.write('WAVE', 8, 'ascii');
  buf.write('fmt ', 12, 'ascii'); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36, 'ascii'); buf.writeUInt32LE(dataSize, 40);
  let off = 44;
  for (let i = 0; i < n; i++) { buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, mono[i])) * 32767), off); off += 2; }
  fs.writeFileSync(filePath, buf);
}

function fftConvolve(dry, ir) {
  const outLen = dry.length + ir.length - 1;
  let n2 = 1;
  while (n2 < outLen) n2 <<= 1;
  const dryRe = new Float64Array(n2), dryIm = new Float64Array(n2);
  const irRe = new Float64Array(n2), irIm = new Float64Array(n2);
  dryRe.set(dry);
  irRe.set(ir);
  fft(dryRe, dryIm, false);
  fft(irRe, irIm, false);
  const outRe = new Float64Array(n2), outIm = new Float64Array(n2);
  for (let i = 0; i < n2; i++) {
    outRe[i] = dryRe[i] * irRe[i] - dryIm[i] * irIm[i];
    outIm[i] = dryRe[i] * irIm[i] + dryIm[i] * irRe[i];
  }
  fft(outRe, outIm, true);
  return outRe.subarray(0, outLen);
}

// Path to the IR file - NOT committed to this repo (see README.md's own
// license note). Override with IR_PATH=... if it lives somewhere else.
const irPath = process.env.IR_PATH || '/home/user/Documents/post-doc/Code/lofigen/samples/impulses/St Nicolaes Church.wav';
const dryPath = path.join(here, 'out', 'bubbles_dry.wav');
const outPath = path.join(here, 'out', 'bubbles_cathedral_full.wav');

const { samples: dry, sampleRate: sr } = readWav(dryPath);
const { samples: irRawAtSrcRate, sampleRate: irSr } = readWav(irPath);
const irRaw = resampleLinear(irRawAtSrcRate, irSr, sr);
console.log('dry:', (dry.length / sr).toFixed(1), 's  IR:', (irRaw.length / sr).toFixed(1), 's at', sr, 'Hz (resampled from', irSr, 'Hz)');

// Normalize the IR's own energy so the convolution's gain is predictable
// regardless of how the original recording was leveled.
let sumSq = 0;
for (let i = 0; i < irRaw.length; i++) sumSq += irRaw[i] * irRaw[i];
const irRms = Math.sqrt(sumSq / irRaw.length) || 1;
const ir = new Float64Array(irRaw.length);
for (let i = 0; i < irRaw.length; i++) ir[i] = irRaw[i] / (irRms * Math.sqrt(irRaw.length));

const wet = fftConvolve(dry, ir);

// Dry/wet mix - a real convolution with an 8s church tail is EXTREMELY
// wet if used anywhere near 1:1, so dry stays dominant with just enough
// wet blended in to hear the cathedral space around the blops rather
// than drowning them in it. Tuned by ear against this exact bubble patch.
const dryLevel = 0.55;
const wetLevel = 0.45;
const outLen = wet.length;
const mixed = new Float64Array(outLen);
for (let i = 0; i < outLen; i++) {
  mixed[i] = (i < dry.length ? dry[i] * dryLevel : 0) + wet[i] * wetLevel;
}

let peak = 0;
for (let i = 0; i < outLen; i++) peak = Math.max(peak, Math.abs(mixed[i]));
const scale = peak > 0 ? 0.9 / peak : 1;
const out16 = new Float32Array(outLen);
for (let i = 0; i < outLen; i++) out16[i] = mixed[i] * scale;

fs.mkdirSync(path.dirname(outPath), { recursive: true });
writeWavMono(outPath, out16, sr);
console.log('wrote', outPath, '-', (outLen / sr).toFixed(1), 's');
console.log('Next: trim the 5s-45s window into docs/audios/, e.g.:');
console.log(`  ffmpeg -i "${outPath}" -ss 5 -to 45 -ac 1 -ar 48000 -acodec pcm_s16le -map_metadata -1 ../../docs/audios/<name>.wav`);
