// Physical-model bubble synthesis - same "excite + resonate" principle as
// the SuperCollider \bpfsaw marimba (impulse -> narrow bandpass -> ringing
// tone at the filter's center frequency), but here each "bubble" IS its
// own resonator (a damped sinusoid), with its frequency set by the
// Minnaert resonance formula instead of a chosen musical note:
//   f0 (Hz) = 6520 / diameter_mm   (standard STP air-bubble-in-water rule)
// and amplitude envelope tau(t) = A0 * sin(2*pi*f(t)*t) * exp(-beta*t),
// matching the well-established damped-harmonic-oscillator bubble model
// (see Sound Design Toolkit's bubbles module, Cornell's bubble synthesis
// paper). A brief exponential frequency "chirp" is layered on top of f0
// at onset (the bubble's effective radius settling right after it forms/
// detaches), same mechanism real bubble-acoustics work models separately.
//
// Each bubble is scattered in time the same way the SC ~marimba pattern's
// \dur, Pexprand(0.1, 1, inf) worked - exponentially-distributed random
// inter-arrival times (several size "populations" layered, for a mix of
// tiny fizz and big glugs).
import fs from 'fs';

const sr = 48000;
const dur = 50; // render 50s so the final file can be trimmed to a clean 5s-45s (40s) window, same convention as water_flow.wav
const N = sr * dur;

function randGauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// One bubble's own damped-sinusoid impulse response, written into `out`
// starting at sample `start`. variant controls which extra ideas are
// layered on top of the bare Minnaert model, for A/B comparison.
function renderBubble(out, start, { diameterMm, amp, numCycles, chirpDepth, chirpCycles, clickAmt, attackMs = 50 }, sr) { // 50ms confirmed - a much softer fade-in than a percussive "pop" would suggest, but that's what sounded right
  const f0 = 6520 / diameterMm; // Minnaert resonance, Hz
  const beta = (Math.PI * f0) / numCycles; // decay rate - tuned so the envelope rings for ~numCycles audible periods regardless of f0
  // BUG (fixed): this used to cap duration at a flat 8/f0 seconds regardless
  // of numCycles/beta, which for anything above ~8 cycles of ring truncated
  // the envelope mid-decay (e.g. numCycles=18 left it at ~25% amplitude,
  // not silence) - an abrupt click on EVERY single bubble that dominated
  // the sound and masked whatever the chirp/click variants were actually
  // doing, which is why all three variants sounded identical. Now derived
  // from beta itself: render until the envelope decays to 0.1% amplitude.
  const durSeconds = Math.min(2, Math.log(1000) / beta);
  const durSamples = Math.min(out.length - start, Math.round(sr * durSeconds));
  if (durSamples <= 0) return;
  // Attack ramp: every bubble used to jump from 0 to full amplitude in a
  // single sample (both the sine and the click) - an instantaneous
  // discontinuity that reads as a harsh digital "tick" rather than a soft
  // "pop", especially with many overlapping. Fading in over attackMs
  // (default 2ms - short enough not to blur the percussive character,
  // long enough to remove the hard edge) softens it.
  const attackTau = attackMs / 1000 / 3; // ~3 time constants to reach ~95% over attackMs
  let phase = 0;
  for (let i = 0; i < durSamples; i++) {
    const t = i / sr;
    // Chirp: frequency starts elevated by chirpDepth (fraction of f0) and
    // decays exponentially back down to f0 over chirpCycles periods of f0.
    const chirpEnv = chirpDepth > 0 ? Math.exp(-t * f0 / Math.max(0.5, chirpCycles)) : 0;
    const fInst = f0 * (1 + chirpDepth * chirpEnv);
    phase += 2 * Math.PI * fInst / sr;
    const attackRamp = 1 - Math.exp(-t / attackTau);
    const envelope = Math.exp(-beta * t);
    let s = Math.sin(phase) * envelope;
    if (clickAmt > 0 && i < sr * 0.002) {
      // Tiny (<2ms) broadband click blended in at onset, like a mallet's
      // own transient riding on top of a marimba bar's resonant tone.
      s += clickAmt * (Math.random() * 2 - 1) * Math.exp(-i / (sr * 0.0008));
    }
    out[start + i] += amp * s * attackRamp;
  }
}

// clusterSize/clusterGapMs: real bubbles-from-a-straw don't arrive as
// independent, evenly-scattered single events (that reads as isolated
// "drips") - they come in rapid "glug-glug-glug" bursts as a stream
// briefly runs. Each Poisson-timed event here can now spawn a short run
// of 1-N bubbles in quick succession instead of just one.
function renderPopulation(out, sr, {
  ratePerSec, diameterRangeMm, ampMean, numCycles, chirpDepth, chirpCycles, clickAmt,
  clusterSize = [1, 1], clusterGapMs = [0, 0], riseFactor = 0,
}) {
  let t = 0;
  const n = out.length;
  // Was a pure Poisson process (-log(rand)/rate): mathematically bursty by
  // construction (its gap distribution's std equals its mean), which is
  // exactly why clusters piled up right next to each other while 3s gaps
  // opened up elsewhere. Jittered-uniform instead (0.5x-1.5x the mean
  // gap) - still randomized, not metronomic, but far less variance, so
  // density stays roughly even over time instead of bursty.
  const meanGapSamples = sr / ratePerSec;
  while (t < n) {
    const gap = Math.max(1, Math.round(meanGapSamples * (0.5 + Math.random())));
    t += gap;
    if (t >= n) break;
    const burstCount = clusterSize[0] + Math.floor(Math.random() * (clusterSize[1] - clusterSize[0] + 1));
    let tt = t;
    for (let k = 0; k < burstCount && tt < n; k++) {
      const baseDiameterMm = diameterRangeMm[0] + Math.random() * (diameterRangeMm[1] - diameterRangeMm[0]);
      // "Blooping" (Sound Design Toolkit's term) - a stream of bubbles
      // rising toward the surface rings progressively HIGHER as it goes
      // (shallower bubbles have less effective mass around them), not a
      // glide WITHIN any one bubble's own tone (physics says a single
      // bubble's own resonant frequency stays ~constant through its
      // decay). riseFactor=0.1 (the toolkit's documented typical value
      // for water) shrinks the effective diameter by up to 10% across a
      // cluster, progressively raising pitch bubble-to-bubble as the
      // "stream" progresses.
      const riseAmount = burstCount > 1 ? k / (burstCount - 1) : 0;
      const diameterMm = baseDiameterMm * (1 - riseFactor * riseAmount);
      // Was exponential (-log(rand)*ampMean) - that distribution has a long
      // tail (occasionally several times ampMean), giving far more
      // loudness variation between bubbles than wanted. Narrow uniform
      // range instead: +-25% around ampMean.
      const amp = ampMean * (0.75 + 0.5 * Math.random());
      renderBubble(out, tt, { diameterMm, amp, numCycles, chirpDepth, chirpCycles, clickAmt }, sr);
      const gapMs = clusterGapMs[0] + Math.random() * (clusterGapMs[1] - clusterGapMs[0]);
      tt += Math.round((sr * gapMs) / 1000);
    }
  }
}

function buildVariant(name, populations) {
  const out = new Float64Array(N);
  for (const pop of populations) renderPopulation(out, sr, pop);
  // peak-normalize
  let peak = 0;
  for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(out[i]));
  const scale = peak > 0 ? 0.85 / peak : 1;
  const out16 = new Float32Array(N);
  for (let i = 0; i < N; i++) out16[i] = out[i] * scale;
  return out16;
}

function writeWavMono(path, mono, sr) {
  const n = mono.length;
  const dataSize = n * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0, 'ascii'); buf.writeUInt32LE(36 + dataSize, 4); buf.write('WAVE', 8, 'ascii');
  buf.write('fmt ', 12, 'ascii'); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36, 'ascii'); buf.writeUInt32LE(dataSize, 40);
  let off = 44;
  for (let i = 0; i < n; i++) { buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, mono[i])) * 32767), off); off += 2; }
  fs.writeFileSync(path, buf);
}

// Redesigned again per feedback: the mid-size 7-16mm "blop" population
// didn't sound good and is dropped; the trace fizz population (0.5-1.3mm)
// read as an unwanted "crispy" background hiss between blops rather than
// realistic texture, and is dropped too - just the one big-blop
// population now, exactly as asked ("they should all be big blops").
function populations({ chirpDepth, chirpCycles, clickAmt }) {
  return [
    // Big blops - denser than before (higher cluster-start rate, bigger
    // clusters) per "more blops in density".
    // diameterRangeMm narrowed from [14,25] (f0 261-466Hz, a wide ~1.8x
    // spread) to [17,21] (f0 310-383Hz, ~1.24x) so every blop sits in
    // roughly the same pitch register instead of varying so much bubble
    // to bubble.
    // Higher cluster-start rate (more overall density) but SMALLER
    // clusters (1-3, down from 2-5) - density now comes from more
    // frequent small clusters rather than occasional big piles, combined
    // with the jittered-uniform (not Poisson) gap timing above.
    { ratePerSec: 6, diameterRangeMm: [17, 21], ampMean: 0.9, numCycles: 13, chirpDepth, chirpCycles, clickAmt, clusterSize: [1, 3], clusterGapMs: [50, 110], riseFactor: 0.3 },
  ];
}

const outDir = new URL('./out/', import.meta.url).pathname;
fs.mkdirSync(outDir, { recursive: true });
const outPath = `${outDir}bubbles_dry.wav`;

writeWavMono(outPath, buildVariant('D', populations({ chirpDepth: 0.5, chirpCycles: 3, clickAmt: 0.4 })), sr);

console.log('Wrote', outPath);
