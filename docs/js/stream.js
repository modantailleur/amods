// Port of Stream._process_chunk (stream.py). The forecaster is always
// "identity" in amods today (the only implementation that exists), so it's
// inlined here rather than ported as its own abstraction.
import { limitPeak } from './audio-utils.js';

export class ConcealerStream {
  /**
   * @param {object} streamConfig - { sr, channels_out, monitor_gain, record_mic_gain, conc_multiplier }
   * @param {SileroVAD} sourceVad - the real-time "is speech happening now" VAD (distinct from concealer.vad).
   * @param {GranSpeechMask} concealer
   * @param {boolean} debug - debug.html's debugTelemetry flag; gates [STREAMDIAG] logging below (see worker-engine.js)
   */
  constructor(streamConfig, sourceVad, concealer, debug = false) {
    this.streamConfig = streamConfig;
    this.sourceVad = sourceVad;
    this.concealer = concealer;
    this.debug = debug; // DIAGNOSTIC gate

    this.pendingConcMaxSize = concealer.pendingConcMaxSize;
    this.pendingConc = new Float32Array(this.pendingConcMaxSize);
    this._playMixGain = 1.0;
    this._recMixGain = 1.0;

    // Timing stats, mirrors Stream._record_callback_timing/pop_callback_timing.
    this._callbackMsSum = 0;
    this._callbackMsMax = 0;
    this._callbackMsCount = 0;
    // Count (not log) of chunks that ran over their real-time budget - a
    // per-chunk [SLOW CALLBACK] warning used to fire here directly, but that
    // spams the console under any sustained slowdown (one line per ~20ms
    // chunk) right when it's least useful to be scrolling past duplicate
    // lines. worker-engine.js's periodic [HEALTH] log reports this count
    // instead, alongside avgMs/maxMs from popCallbackTiming() below - a
    // single "N/M chunks over budget, max Xms" line already says everything
    // the per-chunk spam did.
    this._callbackSlowCount = 0;

    // Debug-visualization only (see docs/debug.html) - the concealing
    // branch's most recent real-time VAD decision, for display alongside
    // the memory branch's own VAD. Harmless to always maintain (one bool).
    // Sent to the main thread on its own dedicated per-chunk message (see
    // worker-engine.js's drainQueue) rather than via the ~500ms status
    // poll, specifically so the on-screen dot reacts at the same ~50ms
    // cadence real detection happens at, not in up-to-500ms-late steps.
    this.lastVoiceActivity = false;
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

    // DIAGNOSTIC (debug.html only) - proves processChunk itself is still
    // being called at a steady cadence (rules out the whole pipeline having
    // silently hung, as opposed to a specific piece like the VAD misbehaving
    // while everything around it keeps running fine).
    if (this.debug) {
      this.__chunkCount = (this.__chunkCount || 0) + 1;
      if (this.__chunkCount % 40 === 0) console.error(`[STREAMDIAG] processChunk call#${this.__chunkCount}`);
    }

    const voiceActivity = await this.sourceVad.predict(x); // forecast === x (identity forecaster)
    this.lastVoiceActivity = voiceActivity;
    let voiceName = '';
    // updateMemory runs every chunk regardless of voice activity - not just
    // while getConcealer's own voice-active branch runs - so that once a
    // pending-speech window has started (see GranSpeechMask.update), it can
    // still complete and fire while you've gone quiet again, rather than
    // only being checked on the next chunk that happens to have voice in it
    // (see granspeechmask.js's pendingSpeechActive/pendingSpeechSamples for
    // why this JS port deliberately diverges from upstream amods here).
    await this.concealer.updateMemory(x);
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

    // Debug-only toggles (see docs/debug.html) - independent of monitorGain/
    // conc_multiplier above, which control each track's own VOLUME when
    // included. These instead gate whether a track is in the played mix AT
    // ALL, so you can A/B "what does the concealer alone sound like" vs "how
    // does it sound blended with my real voice" without having to fiddle
    // with monitorGain (which stays 0 by default outside of this, matching
    // production's original mic-not-monitored behavior). recSum deliberately
    // ignores these - it's the "full" mix regardless of what you chose to
    // listen to, for whenever recording is wired up.
    const listenOriginal = this.streamConfig.listen_original ?? false;
    const listenConcealer = this.streamConfig.listen_concealer ?? true;

    const playSum = new Float32Array(frames);
    const recSum = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      playSum[i] = (listenOriginal ? playMic[i] : 0) + (listenConcealer ? concBlock[i] : 0);
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
    if (elapsedMs > budgetMs) this._callbackSlowCount += 1;

    return { playMix, recMix, voiceName, concealerBlock: concBlock };
  }

  /** Pop and reset the average/max/slow-count callback-time stats (mirrors Stream.pop_callback_timing). */
  popCallbackTiming() {
    const avg = this._callbackMsCount > 0 ? this._callbackMsSum / this._callbackMsCount : 0;
    const max = this._callbackMsMax;
    const count = this._callbackMsCount;
    const slowCount = this._callbackSlowCount;
    this._callbackMsSum = 0;
    this._callbackMsMax = 0;
    this._callbackMsCount = 0;
    this._callbackSlowCount = 0;
    return { avgMs: avg, maxMs: max, count, slowCount };
  }
}
