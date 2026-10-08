// Port of Stream._process_chunk (stream.py). The forecaster is always
// "identity" in amods today (the only implementation that exists), so it's
// inlined here rather than ported as its own abstraction.
import { limitPeak } from './audio-utils.js';
import { SpeechShapedNoise } from './speech-shaped-noise.js';
import { WhiteNoiseGenerator, PinkNoiseGenerator } from './noise-generators.js';
import { LevelTracker } from './level-tracker.js';

// "Soft start" for the Concealer specifically while Speaker mode's
// EchoCanceller is still converging (see echo-canceller.js's own header -
// AEC3 typically needs several seconds of real exposure to a non-trivial
// render signal before it meaningfully cancels an echo; observed directly
// in practice to be on the order of ~10s). Rather than reduce the VOLUME
// of early concealer clips (which would mean AEC3 only ever sees a
// systematically quieter echo, not a realistic one - and the point is for
// it to learn the real room), this instead reduces how OFTEN a selected
// clip actually gets mixed in at all during the ramp window - fewer
// echo events for you to hear while things are still catastrophic, each
// one at its normal, unmodified volume once it does play. The allowed
// fraction ramps linearly from CONCEALER_RAMP_START_FRACTION up to 1.0
// (every opportunity let through, the original/default behavior) over
// CONCEALER_RAMP_SECONDS, using the same fractional-accumulator technique
// as SpectrogramView.pushColumn (see spectrogram.js) so the long-run
// average exactly matches the target fraction with no drift, rather than
// random skipping or round-off bias toward always/never letting a given
// moment through.
//
// The ramp clock starts from the first moment there's actually a clip to
// maybe-play, not from when Speaker mode/the session itself started - that
// distinction matters: GranSpeechMask's own memory-warm-up delay
// (min_memory_to_conceal) already means no concealer output exists for a
// while in a typical session regardless, so a clock that only starts once
// there's real output to gate is what actually lines up with when the
// "catastrophic" window begins, rather than potentially finishing the
// ramp before concealing has even started.
//
// Deliberately NOT applied to the Noise bed (concealing_noise_level stays
// exactly as set, at its normal rate) - unlike concealer clips (short,
// intermittent bursts), Noise is continuous, which makes it a much
// better/faster training signal for AEC3 to converge against in the first
// place; thinning it out would only slow convergence down further,
// working against this whole mechanism's own purpose.
// Set to false to remove this whole mechanism - every opportunity is then
// always let through immediately, same as if Speaker mode had no ramp at
// all (the exact behavior from before CONCEALER_RAMP_SECONDS existed).
// Nothing else needs changing - both _concealerShouldAllowOutput and
// getConcealerRampStatus (and so the "Calibrating echo cancellation…" UI
// card it feeds in main.js) check this first.
const CONCEALER_RAMP_ENABLED = false;
const CONCEALER_RAMP_SECONDS = 15;
const CONCEALER_RAMP_START_FRACTION = 0.1; // only 1 in 10 opportunities let through at the very start

// One-sided power-law expansion applied to noiseLevelScale (see its own
// comment below) - a textbook dynamics "expander": raising a ratio to a
// power widens its dB-domain distance from 1.0 by that same factor. Left
// at 1 (ratio ** 1 = ratio, a no-op) above 1.0 so a louder-than-noise
// source isn't touched at all, but above 1 for a quieter-than-noise one
// so the quiet/silent end gets pushed down much further than the plain
// ratio alone would - directly widens the gap between "loud" and "quiet"
// noise-bed loudness without changing what "loud" sounds like. Purely a
// tuning dial - raise it for a steeper drop-off during quiet, lower
// (toward 1) to soften it back toward the plain ratio.
const NOISE_LEVEL_EXPANSION_EXPONENT = 3;

export class ConcealerStream {
  /**
   * @param {object} streamConfig - { sr, channels_out, monitor_gain, record_mic_gain, conc_multiplier, concealing_noise_level, concealing_noise_sensitivity, noise_level_sensitivity_seconds, noise_level_purity, noise_type, noise_shaped_purity }
   * @param {SileroVAD} sourceVad - the real-time "is speech happening now" VAD (distinct from concealer.vad).
   * @param {GranSpeechMask} concealer
   * @param {SileroVAD|TenVAD} noiseVad - dedicated VAD (own model instance, see worker-engine.js's init()) used only to gate SpeechShapedNoise's calibration input behind "Envelope purity" below; constructed with logitThreshold: null so predict() returns the raw speech ratio, not a bool.
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

    // "Sensitivity" (generic, common to every noise color - unlike
    // "Envelope sensitivity"/concealing_noise_sensitivity above, which
    // only affects SpeechShapedNoise's own spectral envelope) - how
    // reactively the Noise bed's overall LEVEL follows the live source
    // signal's loudness over time. See level-tracker.js's own header for
    // the full mechanism; tied to the SAME lifecycle as _noiseGen itself
    // (lazy, real disconnection while Noise is off) since tracking a
    // level nobody's listening to would be pure waste - see
    // setNoiseEnabled below.
    this._sourceLevelTracker = this.noiseEnabled ? new LevelTracker(this.streamConfig.sr) : null;
    this._noiseLevelTracker = this.noiseEnabled ? new LevelTracker(this.streamConfig.sr) : null;
    if (this._sourceLevelTracker) {
      this._sourceLevelTracker.setTimeConstant(streamConfig.noise_level_sensitivity_seconds ?? 30);
      this._noiseLevelTracker.setTimeConstant(streamConfig.noise_level_sensitivity_seconds ?? 30);
    }

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

    // See this file's own CONCEALER_RAMP_SECONDS comment. _concealerFirstActiveAt:
    // performance.now() timestamp of the first opportunity to maybe-play a
    // concealer clip since Speaker mode turned on (or since the last time
    // Concealer was disabled - see setConcealerEnabled), null until then.
    // _concealerAllowAccumulator: the fractional accumulator itself (see
    // _concealerShouldAllowOutput), carried between calls the same way
    // SpectrogramView's own _shiftAccumulator is.
    this._concealerFirstActiveAt = null;
    this._concealerAllowAccumulator = 0;
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

  /** Master on/off for the whole Noise section - see this constructor's own comment for why this actually constructs/drops the generator instance (and the level trackers alongside it) rather than just gating its output. */
  setNoiseEnabled(enabled) {
    this.noiseEnabled = enabled;
    this._noiseGen = enabled ? this._createNoiseGen(this.noiseType) : null;
    this._sourceLevelTracker = enabled ? new LevelTracker(this.streamConfig.sr) : null;
    this._noiseLevelTracker = enabled ? new LevelTracker(this.streamConfig.sr) : null;
    if (enabled) {
      const seconds = this.streamConfig.noise_level_sensitivity_seconds ?? 30;
      this._sourceLevelTracker.setTimeConstant(seconds);
      this._noiseLevelTracker.setTimeConstant(seconds);
    }
  }

  /** Live-updates the speech-shaped color's own EMA time constant - no-ops (just remembers the value for next activation) unless that color is both selected and currently instantiated. */
  setNoiseSensitivity(seconds) {
    this.streamConfig.concealing_noise_sensitivity = seconds;
    if (this.noiseType === 'speechShaped' && this._noiseGen) this._noiseGen.setSensitivitySeconds(seconds);
  }

  /** Live-updates the generic level-tracking "Sensitivity" slider's EMA time constant - applies to both trackers, regardless of noise color (unlike setNoiseSensitivity above). No-ops (just remembers the value) while Noise is disabled, same reasoning as setNoiseSensitivity. */
  setNoiseLevelSensitivity(seconds) {
    this.streamConfig.noise_level_sensitivity_seconds = seconds;
    if (this._sourceLevelTracker) {
      this._sourceLevelTracker.setTimeConstant(seconds);
      this._noiseLevelTracker.setTimeConstant(seconds);
    }
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
      // Re-enabling later starts a fresh ramp (see CONCEALER_RAMP_SECONDS'
      // own comment) - AEC3's convergence state may well have drifted
      // during however long Concealer was off, so treating the next
      // activation as a new "first output" moment is the conservative,
      // consistent choice (same "resumes from empty" reasoning as the rest
      // of this method).
      this._concealerFirstActiveAt = null;
      this._concealerAllowAccumulator = 0;
    }
  }

  /**
   * true (always let it through) unless Speaker mode is on - see this
   * file's own CONCEALER_RAMP_SECONDS comment for the full reasoning.
   * Records its own start time on first call (the first opportunity to
   * maybe-play a concealer clip), then ramps the allowed fraction linearly
   * from CONCEALER_RAMP_START_FRACTION up to 1.0 (every opportunity) over
   * CONCEALER_RAMP_SECONDS, via a carried-over fractional accumulator
   * (same technique as SpectrogramView.pushColumn in spectrogram.js) so
   * the long-run average exactly matches the target fraction with no
   * drift. Call this exactly once per opportunity (it advances state).
   */
  _concealerShouldAllowOutput() {
    if (!this.echoCanceller || !CONCEALER_RAMP_ENABLED) return true;
    const now = performance.now();
    if (this._concealerFirstActiveAt === null) this._concealerFirstActiveAt = now;
    const elapsedS = (now - this._concealerFirstActiveAt) / 1000;
    const t = Math.min(1, elapsedS / CONCEALER_RAMP_SECONDS);
    const allowFraction = CONCEALER_RAMP_START_FRACTION + (1 - CONCEALER_RAMP_START_FRACTION) * t;
    this._concealerAllowAccumulator += allowFraction;
    if (this._concealerAllowAccumulator >= 1) {
      this._concealerAllowAccumulator -= 1;
      return true;
    }
    return false;
  }

  /**
   * Read-only status of the Concealer ramp (see CONCEALER_RAMP_SECONDS'
   * own comment) for UI display - unlike _concealerShouldAllowOutput,
   * never advances the accumulator or starts the clock itself, so polling
   * this for a status display has no effect on the ramp's own behavior.
   * Returns null whenever there's nothing worth showing: Speaker mode is
   * off, the ramp hasn't started yet (Concealer has had no output to gate
   * at all so far), or it already finished. Otherwise
   * { remainingS, progressFraction } - progressFraction 0..1, how far
   * through the ramp window elapsed time is.
   */
  getConcealerRampStatus() {
    if (!this.echoCanceller || !CONCEALER_RAMP_ENABLED || this._concealerFirstActiveAt === null) return null;
    const elapsedS = (performance.now() - this._concealerFirstActiveAt) / 1000;
    if (elapsedS >= CONCEALER_RAMP_SECONDS) return null;
    return { remainingS: CONCEALER_RAMP_SECONDS - elapsedS, progressFraction: elapsedS / CONCEALER_RAMP_SECONDS };
  }

  /**
   * x: Float32Array, one audio block (mono) at streamConfig.sr - the RAW
   * mic signal, used only for playMic/recMic and the micBlock return value
   * (see their own comments below).
   * cancelledX: Float32Array, same length as x - the signal everything
   * else below (noise calibration, VAD, concealer selection) actually
   * uses. While Speaker mode is off, the caller always passes cancelledX
   * === x (see worker-engine.js's handleChunk), the exact original
   * behavior from before "Speaker mode" existed. While it's on, the
   * caller (worker-engine.js's handleAecChunk) has ALREADY run x through
   * EchoCanceller.processCapture before processChunk is even called -
   * see that function's own comment for why (processing smaller slices
   * of x as they arrive, rather than waiting for this whole chunk to
   * accumulate, shortens the real render->capture latency Speaker mode's
   * AEC3 instance has to estimate and track). Without cancellation here
   * one way or another, over real speakers, the concealer's own
   * synthesized output - picked back up by the mic - would look just like
   * fresh human speech to everything downstream: VAD fires on our own
   * echo, the concealer might store it into memory or trigger MORE output
   * in response to hearing itself - a recursive content feedback loop, a
   * bigger problem than simple audible howling alone.
   * Returns { playMix, recMix: Float32Array (mono), voiceName: string, micBlock, micAecBlock (null unless Speaker mode is on), concealerBlock, noiseBlock: Float32Array (mono, unmixed per-track signals) }.
   * The caller (worker-engine.js) is responsible for recording/tiling to
   * channels_out and for anything Stream.py's record_mode handled - this
   * port has no on-disk recording (no filesystem in a browser tab the way
   * amods.stream has); if recording is wanted later, capture playMix/recMix
   * here and encode client-side (e.g. via MediaRecorder or a WAV writer).
   */
  async processChunk(x, cancelledX) {
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

    // Fed (every chunk, regardless of concealing state) ONLY while Noise is
    // actually enabled - this._noiseGen is null otherwise (see the
    // constructor/setNoiseEnabled), so there's nothing to feed and no
    // per-chunk cost. For SpeechShapedNoise this is the "texture" the
    // background noise bed is shaped from, a completely separate concern
    // from GranSpeechMask's own pendingVoice/VAD/memory below. White/pink
    // noise ignore it (see noise-generators.js) but share the same call.
    //
    // "Envelope purity" (0-0.9) can further gate this for the
    // speechShaped color specifically: only audio that this.noiseVad - a
    // DEDICATED VAD, independent of sourceVad/concealer.vad, see its own
    // constructor param comment - rates above the live purity threshold is
    // actually fed into the calibration, so a silent/non-voice chunk
    // doesn't get baked into the noise's spectral envelope. 0 (the
    // default) means this gate is off entirely - no VAD call made, fed
    // unconditionally, the original behavior from before this existed.
    // White/pink ignore feed()'s content outright, so the gate is a no-op
    // for them regardless of this setting.
    //
    // "Voice focus" (0-0.9, generic - see noise_level_purity below) shares this
    // same noiseVad call when it needs one too, so a chunk that both gates
    // care about only costs one inference, not two.
    const noiseShapedPurity = this.streamConfig.noise_shaped_purity ?? 0;
    const noiseLevelPurity = this.streamConfig.noise_level_purity ?? 0;
    const needsSpeechShapedGate = Boolean(this._noiseGen) && this.noiseType === 'speechShaped' && noiseShapedPurity > 0;
    const needsLevelGate = Boolean(this._sourceLevelTracker) && noiseLevelPurity > 0;
    let noiseVadSpeechRatio = null;
    if ((needsSpeechShapedGate || needsLevelGate) && this.noiseVad) {
      noiseVadSpeechRatio = await this.noiseVad.predict(cancelledX);
    }
    if (this._noiseGen) {
      if (needsSpeechShapedGate) {
        if (noiseVadSpeechRatio > noiseShapedPurity) this._noiseGen.feed(cancelledX);
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
        // _concealerShouldAllowOutput() is always true unless Speaker mode
        // is on - see this file's own CONCEALER_RAMP_SECONDS comment. The
        // clip was still selected/consumed above either way (getConcealer's
        // own countdown/selection state advances the same regardless) -
        // this only gates whether it's actually audible this time, at its
        // normal, unmodified volume whenever it is.
        if (concealerAudio && this._concealerShouldAllowOutput()) {
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

    // "Sensitivity" (see level-tracker.js's own header) - scales noiseGain
    // dynamically so the noise bed's overall level follows the live source
    // signal's loudness over time, multiplying ON TOP of the manual "Noise
    // level" fader above rather than replacing it (that fader still sets
    // the baseline/target level; this tracks around it), using cancelledX
    // (the same post-AEC signal the rest of the pipeline already treats as
    // "the real source" - see this method's own param comment) and the
    // noise generator's own raw (pre-fader) output. The small epsilon
    // avoids a divide-by-zero/huge-spike ratio before the noise generator
    // has produced any real level yet (right at startup, both EMAs begin
    // at 0).
    //
    // this._noiseLevelTracker is always fed plainly (it's just establishing
    // the noise generator's own baseline output level, nothing to do with
    // voice activity). this._sourceLevelTracker is fed plainly too UNLESS
    // "Voice focus" (0-0.9, noiseVadSpeechRatio/noiseLevelPurity above) is
    // raised above 0: gatedMeanSquare's own gate is only relative to
    // whatever's loudest WITHIN each short window it's handed (see its own
    // comment in granspeechmask.js), so room/mic ambient noise alone - with
    // no absolute silence concept - keeps sourceRms from ever really
    // approaching the true floor and the noise bed never gets pulled down
    // with it. Raising "Voice focus" fixes that via updateGated (see level-
    // tracker.js's own comment there): a chunk that would pull the level
    // DOWN is always let through (so it still decays on real silence), one
    // that would push it UP needs noiseVad's speechRatio above the live
    // threshold first (so ambient noise/false triggers can't inflate it).
    // 0 (the default) keeps the original always-on update(), no VAD call.
    let noiseLevelScale = 1;
    if (this._sourceLevelTracker) {
      if (noiseLevelPurity > 0) {
        const speechDetected = noiseVadSpeechRatio !== null && noiseVadSpeechRatio > noiseLevelPurity;
        this._sourceLevelTracker.updateGated(cancelledX, speechDetected);
      } else {
        this._sourceLevelTracker.update(cancelledX);
      }
      this._noiseLevelTracker.update(noiseBlock);
      const rawScale = this._sourceLevelTracker.getRms() / (this._noiseLevelTracker.getRms() + 1e-9);
      // NOISE_LEVEL_EXPANSION_EXPONENT (see its own comment above) - only
      // below 1.0, so a source louder than the noise's own level is passed
      // through unchanged.
      noiseLevelScale = rawScale >= 1 ? rawScale : Math.pow(rawScale, NOISE_LEVEL_EXPANSION_EXPONENT);
    }

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
      const noiseContribution = noiseBlock ? noiseBlock[i] * noiseGain * noiseLevelScale : 0;
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
