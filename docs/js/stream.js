// Port of Stream._process_chunk (stream.py). The forecaster is always
// "identity" in amods today (the only implementation that exists), so it's
// inlined here rather than ported as its own abstraction.
import { limitPeak } from './audio-utils.js';
import { SpeechShapedNoise } from './speech-shaped-noise.js';
import { WhiteNoiseGenerator, PinkNoiseGenerator } from './noise-generators.js';

export class ConcealerStream {
  /**
   * @param {object} streamConfig - { sr, channels_out, monitor_gain, record_mic_gain, conc_multiplier, concealing_noise_level, concealing_noise_sensitivity, noise_type, noise_shaped_purity }
   * @param {SileroVAD} sourceVad - the real-time "is speech happening now" VAD (distinct from concealer.vad).
   * @param {GranSpeechMask} concealer
   * @param {SileroVAD|TenVAD} noiseVad - dedicated VAD (own model instance, see worker-engine.js's init()) used only to gate SpeechShapedNoise's calibration input behind "Speech-shaped purity" below; constructed with logitThreshold: null so predict() returns the raw speech ratio, not a bool.
   * @param {EchoCanceller|null} echoCanceller - "Speaker mode" (see echo-canceller.js's own header) - null while off, the exact original/default behavior.
   * @param {boolean} debug - debug.html's debugTelemetry flag; gates [STREAMDIAG] logging below (see worker-engine.js)
   */
  constructor(streamConfig, sourceVad, concealer, noiseVad, echoCanceller, debug = false) {
    this.streamConfig = streamConfig;
    this.sourceVad = sourceVad;
    this.concealer = concealer;
    this.noiseVad = noiseVad;
    this.echoCanceller = echoCanceller;
    this.debug = debug; // DIAGNOSTIC gate

    this.pendingConcMaxSize = concealer.pendingConcMaxSize;
    this.pendingConc = new Float32Array(this.pendingConcMaxSize);
    this._playMixGain = 1.0;
    this._recMixGain = 1.0;
    // Master on/off for the whole Concealer section - see
    // setConcealerEnabled below for why this is a real disconnect (no VAD/
    // denoiser calls, no memory growth, freed RAM) rather than just a
    // muted output.
    this.concealerEnabled = streamConfig.concealer_enabled ?? true;

    // "Concealing noise" background bed - see speech-shaped-noise.js's own
    // header for why this is a separate, always-running additive layer
    // rather than anything living inside GranSpeechMask. Only ONE
    // generator instance ever exists at a time (this._noiseGen, lazily
    // created) - NOT one of each color held simultaneously - and none at
    // all while the "Noise" toggle is off, so a disabled/inactive color
    // holds no buffers in memory and costs no per-chunk CPU (see
    // _createNoiseGen/setNoiseEnabled/setNoiseType/processChunk below).
    // An earlier version kept all three alive always, for instant
    // dropdown switching at the cost of memory/CPU that's wasted whenever
    // that color isn't the active one, or Noise is off entirely - that
    // tradeoff was explicitly asked to be undone ("real disconnection").
    this.noiseType = streamConfig.noise_type ?? 'speechShaped';
    this.noiseEnabled = streamConfig.noise_enabled ?? false;
    this._noiseGen = this.noiseEnabled ? this._createNoiseGen(this.noiseType) : null;

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
    if (this._noiseGen) this._noiseGen.reset();
  }

  _createNoiseGen(type) {
    if (type === 'white') return new WhiteNoiseGenerator(this.streamConfig.sr);
    if (type === 'pink') return new PinkNoiseGenerator(this.streamConfig.sr);
    return new SpeechShapedNoise(this.streamConfig.sr, this.streamConfig.concealing_noise_sensitivity ?? 2);
  }

  /** Switches the active noise color. Only reconstructs the generator (dropping the old instance for GC) if Noise is currently enabled - while disabled, this.noiseType is just remembered for whenever it's turned back on (see setNoiseEnabled). */
  setNoiseType(type) {
    this.noiseType = type;
    if (this.noiseEnabled) this._noiseGen = this._createNoiseGen(type);
  }

  /** Master on/off for the whole Noise section - see this constructor's own comment for why this actually constructs/drops the generator instance rather than just gating its output. */
  setNoiseEnabled(enabled) {
    this.noiseEnabled = enabled;
    this._noiseGen = enabled ? this._createNoiseGen(this.noiseType) : null;
  }

  /** Live-updates the speech-shaped color's own EMA time constant - no-ops (just remembers the value for next activation) unless that color is both selected and currently instantiated. */
  setNoiseSensitivity(seconds) {
    this.streamConfig.concealing_noise_sensitivity = seconds;
    if (this.noiseType === 'speechShaped' && this._noiseGen) this._noiseGen.setSensitivitySeconds(seconds);
  }

  /**
   * Master on/off for the whole Concealer section. A real disconnect, not
   * just a muted output: while disabled, processChunk below skips calling
   * into this.concealer/this.sourceVad entirely (no VAD inference, no
   * denoiser calls, no memory growth - the actual CPU cost of this
   * feature), and disabling also flushes any in-flight concealer audio
   * still draining out of pendingConc and frees GranSpeechMask's own two
   * large allocations (see its releaseMemory()'s own comment for what
   * that actually frees and why this.vad/this.denoiser are deliberately
   * left alone). Re-enabling resumes from empty, same as a fresh session.
   */
  setConcealerEnabled(enabled) {
    this.concealerEnabled = enabled;
    if (!enabled) {
      this.pendingConc.fill(0);
      this.concealer.releaseMemory();
    }
  }

  /**
   * x: Float32Array, one audio block (mono) at streamConfig.sr.
   * Returns { playMix, recMix: Float32Array (mono), voiceName: string, micBlock, micAecBlock (null unless Speaker mode is on), concealerBlock, noiseBlock: Float32Array (mono, unmixed per-track signals) }.
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

    // Acoustic echo cancellation ("Speaker mode" - see echo-canceller.js's
    // own header for the algorithm/citation/why). Replaces the raw mic
    // signal with its own best estimate of "x with our own speaker output
    // subtracted out" BEFORE anything else below (noise calibration, VAD,
    // concealer selection) ever sees it. Without this, over real speakers,
    // the concealer's own synthesized output - picked back up by the mic -
    // would look just like fresh human speech to everything downstream:
    // VAD fires on our own echo, the concealer might store it into memory
    // or trigger MORE output in response to hearing itself - a recursive
    // content feedback loop, a bigger problem than simple audible howling
    // alone. null (the default, Speaker mode off) means this is a no-op -
    // cancelledX === x, the exact original behavior from before this existed.
    const cancelledX = this.echoCanceller ? await this.echoCanceller.processCapture(x) : x;

    // Fed (every chunk, regardless of concealing state) ONLY while Noise is
    // actually enabled - this._noiseGen is null otherwise (see the
    // constructor/setNoiseEnabled), so there's nothing to feed and no
    // per-chunk cost. For SpeechShapedNoise this is the "texture" the
    // background noise bed is shaped from, a completely separate concern
    // from GranSpeechMask's own pendingVoice/VAD/memory below. White/pink
    // noise ignore it (see noise-generators.js) but share the same call.
    //
    // "Speech-shaped purity" (0-0.9) can further gate this for the
    // speechShaped color specifically: only audio that this.noiseVad - a
    // DEDICATED VAD, independent of sourceVad/concealer.vad, see its own
    // constructor param comment - rates above the live purity threshold is
    // actually fed into the calibration, so a silent/non-voice chunk
    // doesn't get baked into the noise's spectral envelope. 0 (the
    // default) means this gate is off entirely - no VAD call made, fed
    // unconditionally, the original behavior from before this existed.
    // White/pink ignore feed()'s content outright, so the gate is a no-op
    // for them regardless of this setting.
    if (this._noiseGen) {
      const purity = this.streamConfig.noise_shaped_purity ?? 0;
      if (this.noiseType === 'speechShaped' && purity > 0 && this.noiseVad) {
        const speechRatio = await this.noiseVad.predict(cancelledX);
        if (speechRatio > purity) this._noiseGen.feed(cancelledX);
      } else {
        this._noiseGen.feed(cancelledX);
      }
    }

    let voiceName = '';
    // Everything in this block - sourceVad inference, GranSpeechMask's own
    // memory-growth/VAD/denoiser work, the concealer-selection search -
    // is skipped entirely while Concealer is disabled (see
    // setConcealerEnabled's own comment): a real disconnect, not just a
    // muted output. lastVoiceActivity (debug viz only) just stays false.
    if (this.concealerEnabled) {
      const voiceActivity = await this.sourceVad.predict(cancelledX); // forecast === cancelledX (identity forecaster)
      this.lastVoiceActivity = voiceActivity;
      // updateMemory runs every chunk regardless of voice activity - not just
      // while getConcealer's own voice-active branch runs - so that once a
      // pending-speech window has started (see GranSpeechMask.update), it can
      // still complete and fire while you've gone quiet again, rather than
      // only being checked on the next chunk that happens to have voice in it
      // (see granspeechmask.js's pendingSpeechActive/pendingSpeechSamples for
      // why this JS port deliberately diverges from upstream amods here).
      await this.concealer.updateMemory(cancelledX);
      if (voiceActivity) {
        const { audio: concealerAudio, voiceName: vn } = await this.concealer.getConcealer(cancelledX);
        voiceName = vn;
        if (concealerAudio) {
          const mult = this.streamConfig.conc_multiplier;
          for (let i = 0; i < concealerAudio.length && i < this.pendingConc.length; i++) {
            this.pendingConc[i] += concealerAudio[i] * mult;
          }
        }
      } else {
        this.concealer.refresh(cancelledX);
      }
    } else {
      this.lastVoiceActivity = false;
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
    // Same gate as listenOriginal/listenConcealer, same reasoning - lets
    // the "Noise track" debug toggle A/B whether the noise bed is
    // audible without touching the "Noise" level slider itself. Unlike
    // those two, defaults to true (production always has the noise bed
    // audible whenever its own level is above zero - this is a debug-only
    // A/B control, not a feature production needs an opinion on).
    const listenNoise = this.streamConfig.listen_noise ?? true;

    // The noise bed is its own third layer, neither "the concealer" nor
    // "the original mic" - always included in recSum (when its own level
    // is above zero and enabled), same as concBlock is, rather than
    // gated the way playSum's copy is by listenNoise above.
    // concealing_noise_level is the "Noise level" dB fader's own
    // multiplier, independent of conc_multiplier ("Concealer level") -
    // the two used to be tied together, but that's been explicitly
    // undone; each level fader now only affects its own section.
    const noiseBlock = this._noiseGen ? this._noiseGen.nextBlock(frames) : null;
    const noiseGain = this.streamConfig.concealing_noise_level ?? 0;

    const playSum = new Float32Array(frames);
    const recSum = new Float32Array(frames);
    // Gain-applied noise contribution, per sample - built as its own array
    // (not just an inline scalar in the loop below) so it can be returned
    // for debug.html's "Visualization" spectrograms (see worker-engine.js's
    // drainQueue), the same unmixed-per-track role concBlock already plays.
    const noiseContributionBlock = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      // concBlock is already guaranteed all-zero while concealerEnabled is
      // false (see setConcealerEnabled - nothing adds to pendingConc while
      // disabled, and it's flushed the moment it's turned off), so this
      // gate is a defensive no-op rather than load-bearing, but keeps the
      // intent explicit here too.
      const concContribution = this.concealerEnabled ? concBlock[i] : 0;
      const noiseContribution = noiseBlock ? noiseBlock[i] * noiseGain : 0;
      noiseContributionBlock[i] = noiseContribution;
      playSum[i] = (listenOriginal ? playMic[i] : 0) + (listenConcealer ? concContribution : 0) + (listenNoise ? noiseContribution : 0);
      recSum[i] = recMic[i] + concContribution + noiseContribution;
    }

    const sr = this.streamConfig.sr;
    const { y: playMix, newGain: playGain } = limitPeak(playSum, { limit: 0.95, prevGain: this._playMixGain, sr });
    this._playMixGain = playGain;
    const { y: recMix, newGain: recGain } = limitPeak(recSum, { limit: 0.95, prevGain: this._recMixGain, sr });
    this._recMixGain = recGain;

    // Registers THIS chunk's final mixed output as "render" data (see
    // echo-canceller.js's own header) - its echo, once it actually travels
    // speaker->air->mic, is what a LATER chunk's processCapture call above
    // will be matched against and subtracted out.
    if (this.echoCanceller) await this.echoCanceller.analyzeRender(playMix);

    const elapsedMs = performance.now() - t0;
    this._callbackMsSum += elapsedMs;
    this._callbackMsMax = Math.max(this._callbackMsMax, elapsedMs);
    this._callbackMsCount += 1;
    const budgetMs = (frames / sr) * 1000;
    if (elapsedMs > budgetMs) this._callbackSlowCount += 1;

    // micBlock/concealerBlock/noiseBlock: the three unmixed per-track
    // signals (debug.html's "Visualization" spectrograms read these - see
    // worker-engine.js's drainQueue). Each reflects its own section's
    // master enabled/disabled state (silent when off) but NOT the debug-
    // only listen toggles - the point is to show what each track is
    // actually producing, not just whatever you currently have audible.
    // micBlock is deliberately the RAW x (not cancelledX) - a genuine
    // "before" reference for debug.html's "Source" vs "Source (echo-
    // canceled)" comparison. micAecBlock is cancelledX only while Speaker
    // mode is on, null otherwise (nothing to compare against when it's off).
    return {
      playMix,
      recMix,
      voiceName,
      micBlock: x,
      micAecBlock: this.echoCanceller ? cancelledX : null,
      concealerBlock: concBlock,
      noiseBlock: noiseContributionBlock,
    };
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
