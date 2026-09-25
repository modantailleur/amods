// Port of Stream._process_chunk (stream.py). The forecaster is always
// "identity" in amods today (the only implementation that exists), so it's
// inlined here rather than ported as its own abstraction.
import { limitPeak } from './audio-utils.js';

export class ConcealerStream {
  /**
   * @param {object} streamConfig - { sr, channels_out, monitor_gain, record_mic_gain, conc_multiplier }
   * @param {SileroVAD} sourceVad - the real-time "is speech happening now" VAD (distinct from concealer.vad).
   * @param {GranSpeechMask} concealer
   */
  constructor(streamConfig, sourceVad, concealer) {
    this.streamConfig = streamConfig;
    this.sourceVad = sourceVad;
    this.concealer = concealer;

    this.pendingConcMaxSize = concealer.pendingConcMaxSize;
    this.pendingConc = new Float32Array(this.pendingConcMaxSize);
    this._playMixGain = 1.0;
    this._recMixGain = 1.0;

    // Timing stats, mirrors Stream._record_callback_timing/pop_callback_timing.
    this._callbackMsSum = 0;
    this._callbackMsMax = 0;
    this._callbackMsCount = 0;
  }

  resetState() {
    this.pendingConc = new Float32Array(this.pendingConcMaxSize);
    this._playMixGain = 1.0;
    this._recMixGain = 1.0;
  }

  /**
   * x: Float32Array, one audio block (mono) at streamConfig.sr.
   * Returns { playMix: Float32Array (mono), recMix: Float32Array (mono), voiceName: string, concealerBlock: Float32Array }.
   * The caller (worker-engine.js) is responsible for recording/tiling to
   * channels_out and for anything Stream.py's record_mode handled - this
   * port has no on-disk recording (no filesystem in a browser tab the way
   * amods.stream has); if recording is wanted later, capture playMix/recMix
   * here and encode client-side (e.g. via MediaRecorder or a WAV writer).
   */
  async processChunk(x) {
    const t0 = performance.now();
    const frames = x.length;

    const voiceActivity = await this.sourceVad.predict(x); // forecast === x (identity forecaster)
    let voiceName = '';
    if (voiceActivity) {
      const { audio: concealerAudio, voiceName: vn } = await this.concealer.getConcealer(x);
      voiceName = vn;
      if (concealerAudio) {
        const mult = this.streamConfig.conc_multiplier;
        for (let i = 0; i < concealerAudio.length && i < this.pendingConc.length; i++) {
          this.pendingConc[i] += concealerAudio[i] * mult;
        }
      }
    } else {
      this.concealer.refresh(x);
    }

    const concBlock = this.pendingConc.slice(0, frames);
    // Shift the ring buffer left by `frames`, zero-filling the tail.
    this.pendingConc.copyWithin(0, frames);
    this.pendingConc.fill(0, this.pendingConc.length - frames);

    const playMic = new Float32Array(frames);
    const recMic = new Float32Array(frames);
    const monitorGain = this.streamConfig.monitor_gain;
    const recordMicGain = this.streamConfig.record_mic_gain;
    for (let i = 0; i < frames; i++) {
      playMic[i] = monitorGain * x[i];
      recMic[i] = recordMicGain * x[i];
    }

    const playSum = new Float32Array(frames);
    const recSum = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      playSum[i] = playMic[i] + concBlock[i];
      recSum[i] = recMic[i] + concBlock[i];
    }

    const sr = this.streamConfig.sr;
    const { y: playMix, newGain: playGain } = limitPeak(playSum, { limit: 0.95, prevGain: this._playMixGain, sr });
    this._playMixGain = playGain;
    const { y: recMix, newGain: recGain } = limitPeak(recSum, { limit: 0.95, prevGain: this._recMixGain, sr });
    this._recMixGain = recGain;

    const elapsedMs = performance.now() - t0;
    this._callbackMsSum += elapsedMs;
    this._callbackMsMax = Math.max(this._callbackMsMax, elapsedMs);
    this._callbackMsCount += 1;
    const budgetMs = (frames / sr) * 1000;
    if (elapsedMs > budgetMs) {
      console.warn(`[SLOW CALLBACK] took ${elapsedMs.toFixed(1)}ms, budget was ${budgetMs.toFixed(1)}ms`);
    }

    return { playMix, recMix, voiceName, concealerBlock: concBlock };
  }

  /** Pop and reset the average/max callback-time stats (mirrors Stream.pop_callback_timing). */
  popCallbackTiming() {
    const avg = this._callbackMsCount > 0 ? this._callbackMsSum / this._callbackMsCount : 0;
    const max = this._callbackMsMax;
    this._callbackMsSum = 0;
    this._callbackMsMax = 0;
    this._callbackMsCount = 0;
    return { avgMs: avg, maxMs: max };
  }
}
