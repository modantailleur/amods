// K-weighting (ITU-R BS.1770-4, Annex 1 / EBU R128) - the perceptual
// pre-filter real loudness meters apply before their own gated-RMS step
// (the same gating technique granspeechmask.js's gatedMeanSquare/
// liveGatedMeanSquare already borrow from those standards - see their own
// comments). A cascade of two 2nd-order IIR (biquad) filters:
//   1. a high-shelf ("head effects" simulation - very roughly, the
//      acoustic effect of measuring at the ear rather than in free air),
//      which boosts roughly the 2-4kHz range;
//   2. a high-pass (RLB - "Revised Low-frequency B" - weighting), which
//      rolls off bass;
// together approximating how the ear perceives loudness across frequency,
// rather than treating all frequencies as equally loud the way a flat
// broadband RMS does.
//
// Coefficients are derived here from the filters' own analog-domain
// specification (center frequency/Q/gain) via the standard bilinear
// transform, for whatever the real device sample rate turns out to be -
// NOT hardcoded for one fixed rate, since this app runs at whatever rate
// the browser/device actually reports (see main.js's own CHUNK_SIZE_AT_48K
// comment for why that can't be assumed). This is the same derivation
// reference implementations of the standard use to support arbitrary
// sample rates - cross-checked against pyloudnorm
// (https://github.com/csteinmetz1/pyloudnorm, MIT) and ffmpeg's
// libavfilter ebur128 filter, both open-source implementations of BS.1770 -
// and verified directly (not just assumed) to reproduce the standard's own
// published Table 1 coefficients at 48kHz exactly (see this repo's test
// scripts from the session that added this).

const filterCache = new Map();

function designFilters(sr) {
  // Stage 1: high-shelf ("head effects").
  const f0a = 1681.9744509555319;
  const Ga = 3.99984385397342; // dB
  const Qa = 0.7071752369554196;
  const Ka = Math.tan((Math.PI * f0a) / sr);
  const Vh = 10 ** (Ga / 20);
  const Vb = Vh ** 0.4996667741545416;
  const a0a = 1 + Ka / Qa + Ka * Ka;
  const stage1 = {
    b0: (Vh + (Vb * Ka) / Qa + Ka * Ka) / a0a,
    b1: (2 * (Ka * Ka - Vh)) / a0a,
    b2: (Vh - (Vb * Ka) / Qa + Ka * Ka) / a0a,
    a1: (2 * (Ka * Ka - 1)) / a0a,
    a2: (1 - Ka / Qa + Ka * Ka) / a0a,
  };

  // Stage 2: high-pass (RLB weighting).
  const f0b = 38.13547087613982;
  const Qb = 0.5003270373238773;
  const Kb = Math.tan((Math.PI * f0b) / sr);
  const a0b = 1 + Kb / Qb + Kb * Kb;
  const stage2 = {
    b0: 1,
    b1: -2,
    b2: 1,
    a1: (2 * (Kb * Kb - 1)) / a0b,
    a2: (1 - Kb / Qb + Kb * Kb) / a0b,
  };

  return [stage1, stage2];
}

function getFilters(sr) {
  if (!filterCache.has(sr)) filterCache.set(sr, designFilters(sr));
  return filterCache.get(sr);
}

/**
 * Direct-form-I biquad, one stage, using/updating the given state object
 * (x1,x2,y1,y2) in place - lets a caller filter a long signal
 * incrementally, chunk by chunk, with the filter's own history correctly
 * carried across calls (see StreamingKWeighting below), rather than
 * restarting - and re-triggering the startup transient - every call the
 * way a fresh {x1:0,x2:0,y1:0,y2:0} state would.
 */
function applyBiquadStateful(x, coeffs, state) {
  const { b0, b1, b2, a1, a2 } = coeffs;
  const y = new Float64Array(x.length);
  let { x1, x2, y1, y2 } = state;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    y[i] = yi;
    x2 = x1;
    x1 = xi;
    y2 = y1;
    y1 = yi;
  }
  state.x1 = x1;
  state.x2 = x2;
  state.y1 = y1;
  state.y2 = y2;
  return y;
}

/**
 * K-weights x (see this module's own header), returning a new Float64Array
 * the same length as x. Fresh (zero) filter state every call - correct for
 * filtering a standalone buffer with no accessible prior history (a
 * stored memory clip), and still correct for a live buffer slice as long
 * as the WHOLE slice is filtered in one call before any further chunking
 * happens (see granspeechmask.js's liveGatedMeanSquare) - the startup
 * transient only ever happens once, at the true start of whatever's being
 * measured, not re-introduced at internal chunk boundaries. For a
 * long-running real-time stream fed in many successive small chunks
 * instead (continuous mic input, not a single already-complete buffer),
 * use StreamingKWeighting below, which carries filter state correctly
 * across calls instead of restarting it fresh (and re-triggering the
 * transient) every time.
 */
export function applyKWeighting(x, sr) {
  const [stage1, stage2] = getFilters(sr);
  return applyBiquadStateful(applyBiquadStateful(x, stage1, { x1: 0, x2: 0, y1: 0, y2: 0 }), stage2, { x1: 0, x2: 0, y1: 0, y2: 0 });
}

/**
 * Continuous, stateful K-weighting for a long-running real-time stream fed
 * in small successive chunks (e.g. this app's 50ms chunks) - see
 * level-tracker.js for what this is actually used for. Unlike
 * applyKWeighting (fresh/zero filter state every call), this instance
 * carries the two biquad stages' own history correctly across calls, so
 * the startup transient only ever happens once, right when the instance
 * is first created, not re-triggered on every chunk.
 */
export class StreamingKWeighting {
  constructor(sr) {
    const [stage1, stage2] = getFilters(sr);
    this.stage1 = stage1;
    this.stage2 = stage2;
    this.state1 = { x1: 0, x2: 0, y1: 0, y2: 0 };
    this.state2 = { x1: 0, x2: 0, y1: 0, y2: 0 };
  }

  /** x: Float32Array/Float64Array chunk. Returns a new Float64Array, same length. */
  process(x) {
    return applyBiquadStateful(applyBiquadStateful(x, this.stage1, this.state1), this.stage2, this.state2);
  }
}
