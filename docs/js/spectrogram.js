// Live log-spectrogram visualization for debug.html's "Visualization"
// panel (Microphone / Concealer / Noise tracks). Two halves, deliberately
// in one file since they're tightly coupled by the data format they
// share, but used in two different realms:
//   - SpectrogramAnalyzer runs in worker-engine.js (the dedicated Worker
//     doing real audio processing) - pure number-crunching, no DOM/canvas
//     access, since Workers don't have it (well, OffscreenCanvas exists,
//     but rendering stays main-thread-side here to keep this Worker
//     focused on audio, not drawing).
//   - SpectrogramView runs in main.js (the main/UI thread) - owns a
//     <canvas>, receives the Analyzer's already-computed columns over
//     postMessage, and only ever draws - no FFT/analysis logic here.
import { fft, hannWindow, melFilterbank } from './mel.js';

const FFT_SIZE = 1024; // ~21ms at 48kHz - fine time/frequency tradeoff for a debug visualization, not a DSP-critical computation
// Most speech/noise content of interest lives well under 8kHz - devoting
// display resolution to the upper half of a 48kHz signal's range
// (12-24kHz) would mostly show near-silence. Bins above this are
// discarded rather than compressed in, so the DISPLAY_BINS that remain
// cover the range that actually varies.
const MAX_DISPLAY_HZ = 8000;
// Reasoned starting point for what counts as "silent" vs "loud" on the
// color scale, not calibrated by ear (no way to do that here). DB_CEIL in
// particular needs real headroom: the "Concealer level"/"Noise level"
// faders go up to +36dB (a 63x multiplier) on top of whatever a track's
// own unboosted content already measures, so a ceiling set from
// unboosted-only observations clips solid at the top the moment either
// fader is pushed up - confirmed directly (a live test with "Noise
// level" at just +10dB came out as a single uniform saturated block, no
// visible texture at all). Adjust these two further if everything still
// looks uniformly dark or uniformly saturated in practice.
const DB_FLOOR = -70;
const DB_CEIL = 15;

/**
 * Worker-side: maintains a rolling FFT_SIZE-sample buffer per track,
 * producing one log-mel column (Float32Array of length displayBins,
 * values pre-normalized to [0, 1], index 0 = lowest frequency) each time
 * computeColumn() is called. No DOM dependency - safe to import and run
 * inside a Worker.
 *
 * Mel-scaled (not linear-frequency) so the limited displayBins rows are
 * spent where speech energy actually concentrates (low frequencies) -
 * reuses the same Slaney-style filterbank builder mel.js already has for
 * VAD/denoiser feature extraction (see melFilterbank there), just with
 * its own fmin/fmax/nMels tuned for this 0-8kHz display instead.
 */
export class SpectrogramAnalyzer {
  constructor(sr, displayBins) {
    this.sr = sr;
    this.displayBins = displayBins;
    this.window = hannWindow(FFT_SIZE);
    this.buffer = new Float64Array(FFT_SIZE); // rolling - holds the most recent FFT_SIZE samples seen so far
    this.filterbank = melFilterbank(sr, FFT_SIZE, displayBins, 0, MAX_DISPLAY_HZ);
  }

  /** Append new samples (any length) to the rolling buffer, dropping the oldest to make room. */
  push(x) {
    const n = x.length;
    if (n >= FFT_SIZE) {
      this.buffer.set(x.subarray(n - FFT_SIZE));
    } else {
      this.buffer.copyWithin(0, n);
      for (let i = 0; i < n; i++) this.buffer[FFT_SIZE - n + i] = x[i];
    }
  }

  computeColumn() {
    const re = new Float64Array(FFT_SIZE);
    const im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) re[i] = this.buffer[i] * this.window[i];
    fft(re, im);

    const nFreqs = FFT_SIZE / 2 + 1;
    const out = new Float32Array(this.displayBins);
    for (let m = 0; m < this.displayBins; m++) {
      const w = this.filterbank[m];
      let e = 0;
      for (let i = 0; i < nFreqs; i++) e += w[i] * (re[i] * re[i] + im[i] * im[i]);
      const db = 10 * Math.log10(Math.max(e, 1e-12));
      out[m] = Math.max(0, Math.min(1, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)));
    }
    return out;
  }
}

// Matplotlib's "magma" colormap - perceptually uniform (equal steps in
// the data read as equal steps in perceived brightness, unlike the
// earlier black/blue/cyan/yellow/red gradient this replaced, which had
// uneven perceptual jumps between its stops) and still high-contrast/
// readable at a glance. These 9 points are standard magma reference
// values (black -> purple -> magenta -> orange -> pale yellow),
// interpolated linearly between them the same way the old gradient was.
const COLOR_STOPS = [
  [0, 0, 4],
  [28, 16, 68],
  [79, 18, 123],
  [129, 37, 129],
  [181, 54, 122],
  [229, 80, 100],
  [251, 135, 97],
  [254, 194, 135],
  [252, 253, 191],
];

function magmaColorRGB(t) {
  t = Math.max(0, Math.min(1, t));
  const segs = COLOR_STOPS.length - 1;
  const pos = t * segs;
  const idx = Math.min(segs - 1, Math.floor(pos));
  const frac = pos - idx;
  const [r0, g0, b0] = COLOR_STOPS[idx];
  const [r1, g1, b1] = COLOR_STOPS[idx + 1];
  return [Math.round(r0 + (r1 - r0) * frac), Math.round(g0 + (g1 - g0) * frac), Math.round(b0 + (b1 - b0) * frac)];
}

/**
 * Main-thread: owns a <canvas> and renders incoming columns (see
 * SpectrogramAnalyzer.computeColumn) as a scrolling spectrogram - time
 * flows left to right (oldest data scrolls off the left edge, each new
 * column is drawn on the right), low frequency at the bottom, high
 * frequency at the top (standard spectrogram convention). The canvas's
 * own pixel height must equal the column length it's fed (see main.js,
 * where SpectrogramAnalyzer's displayBins is constructed to match).
 *
 * Each incoming column is drawn columnWidth pixels wide (default 1) -
 * raising it shows LESS total time across the same canvas width (each
 * chunk takes up more of it, so older content scrolls off sooner), i.e.
 * "zooms in" on the time axis without changing how often new data
 * actually arrives. columnWidth can be fractional (set via
 * setColumnWidth, e.g. to hit an exact target number of seconds across
 * the canvas's fixed pixel width) - handled via a carried-over fractional
 * accumulator (classic Bresenham-style technique) so the long-run average
 * pixels-per-column stays exactly right instead of drifting from rounding
 * every single push the same way.
 */
export class SpectrogramView {
  constructor(canvas, columnWidth = 1) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = canvas.width;
    this.height = canvas.height;
    this.columnWidth = columnWidth;
    this._shiftAccumulator = 0;
    this.clear();
  }

  /** Sets how many pixels wide each new column is drawn, in one step, from a desired total-seconds-visible target and how many columns arrive per second. */
  setTimeScale(columnsPerSecond, targetSeconds) {
    this.columnWidth = this.width / (targetSeconds * columnsPerSecond);
  }

  clear() {
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.width, this.height);
    this._shiftAccumulator = 0;
  }

  /** column: Float32Array of normalized [0,1] magnitudes, index 0 = lowest frequency, length === this.height. */
  pushColumn(column) {
    this._shiftAccumulator += this.columnWidth;
    const shiftPx = Math.min(this.width, Math.max(1, Math.round(this._shiftAccumulator)));
    this._shiftAccumulator -= shiftPx;

    // Scroll existing content shiftPx left (self-copy via drawImage - the
    // standard, widely-used pattern for a scrolling canvas; verified
    // working directly rather than assumed, since drawImage with
    // overlapping source/destination regions on the SAME canvas isn't
    // something every engine is required to special-case consistently).
    if (shiftPx < this.width) {
      this.ctx.drawImage(this.canvas, shiftPx, 0, this.width - shiftPx, this.height, 0, 0, this.width - shiftPx, this.height);
    } else {
      this.ctx.fillStyle = '#000';
      this.ctx.fillRect(0, 0, this.width, this.height);
    }

    const nBins = column.length;
    const imgData = this.ctx.createImageData(shiftPx, this.height);
    const data = imgData.data;
    for (let y = 0; y < this.height; y++) {
      const binIdx = Math.min(nBins - 1, nBins - 1 - Math.floor((y / this.height) * nBins));
      const [r, g, b] = magmaColorRGB(column[binIdx]);
      for (let px = 0; px < shiftPx; px++) {
        const off = (y * shiftPx + px) * 4;
        data[off] = r;
        data[off + 1] = g;
        data[off + 2] = b;
        data[off + 3] = 255;
      }
    }
    this.ctx.putImageData(imgData, this.width - shiftPx, 0);
  }
}
