// Port of GranSpeechMask (granspeechmask.py) + the memory/matching parts of
// GranSpeechMaskCM (base.py) it inherits. Runs inside a dedicated Worker
// (see worker-engine.js), not the audio thread, so async ONNX calls here
// (VAD, denoiser) are fine even though they briefly block that worker.
//
// Deliberately NOT ported, since the GUI this is modeled on never enables
// them (concealer_config always sets random_reverse: false, and
// max_len_strat/distance_type are never set in configs/concealer/default.yaml,
// so they stay at their "queue"/"cosine" defaults):
//   - random_reverse / ReverseAM (playing a clip time-reversed)
//   - max_len_strat="diversity_marginal_gain" and distance_type="p-correlation"
// If either is ever wanted for the web build, port them from
// src/amods/models/adversary.py and base.py's
// _compute_swap_marginal_gain/_compute_correlation_matrix at that point.
import { applyFade } from './audio-utils.js';
import { featureExtractor } from './mel.js';

export class GranSpeechMask {
  /**
   * @param {number} sr - audio sample rate this instance operates at.
   * @param {object} config - see each field below for the Python config key it matches.
   * @param {object} deps - { denoiser: DnsDenoiser|null, vad: SileroVAD } - the concealer's *own* VAD (config's concealer_vad_config), not the source VAD.
   */
  constructor(sr, config, deps) {
    this.sr = sr;
    this.config = config;
    this.denoiser = deps.denoiser; // null if config.denoise is false
    this.vad = deps.vad;

    this.memoryMaxlen = config.memory_maxlen;
    this.minMemoryToConceal = config.min_memory_to_conceal ?? 20;
    this.maxCountdownReuse = config.max_countdown_reuse;
    this.maxConcealerDistanceToBuffer = config.max_concealer_distance_to_buffer;
    this.concealerDuration = config.concealer_duration;
    this.concealingMinTimeoutRatio = config.concealing_min_timeout_ratio;
    this.concealingMaxTimeoutRatio = config.concealing_max_timeout_ratio;
    this.fadeDuration = config.fade_duration;
    this.isStream = config.is_stream ?? true;
    this.pendingConcMaxSize = config.pending_conc_max_size;
    this.freezeLearning = config.freeze_learning ?? false;
    this.decisionWin = config.decision_win ?? 0.3;
    this.denoise = config.denoise ?? true;

    this.concealerSize = Math.trunc(sr * this.concealerDuration);
    this.fadeSize = Math.trunc(sr * this.fadeDuration);

    this.memory = []; // [{clip: Float32Array, feat: Float64Array, cooldown: number, voiceName: string}]
    this.pendingVoice = []; // rolling window, plain array of samples (see module docstring in the .py for why streaming mode never clears this)
    this.pendingVoiceMaxDuration = 2; // seconds
    this.nConcealerBeforeDenoise = Math.ceil(this.pendingVoiceMaxDuration / this.concealerDuration);

    this.curSizeBeforeUpdateMemory = 0;
    this.stopProcessing = false;
    this.curConcealing = false;
    this.concealingCountdown = 0;
    // JS-port-only divergence from upstream amods (see the .py's own
    // new_voice_since_feed, which instead requires pendingVoiceMaxDuration
    // worth of CUMULATIVE voice-active audio - possibly spread across
    // arbitrarily long silent gaps - before ever feeding memory). That
    // design lets pendingVoice (a plain rolling last-N-seconds window) drift
    // out of sync with what the counter is actually counting: audio spoken
    // long enough ago can scroll out of pendingVoice before the cumulative
    // total finally crosses the threshold, so the clips that actually get
    // fed may contain none of the speech that triggered the feed.
    //
    // Instead: the moment source VAD detects voice after being idle, a
    // pending-speech window starts and always completes exactly
    // pendingVoiceMaxDuration later, counting every sample from then on
    // (voice or silence - see update()/refresh()) rather than only
    // voice-active ones. Since that's the same duration pendingVoice itself
    // spans, the fed clip is guaranteed to actually contain the voice that
    // started the window. It also bounds worst-case denoiser load to "at
    // most once per pendingVoiceMaxDuration of audio that contains any
    // speech at all", rather than running arbitrarily rarely (as above) or,
    // if instead triggered unconditionally on a timer, arbitrarily often
    // across long stretches of pure silence/noise.
    this.pendingSpeechActive = false;
    this.pendingSpeechSamples = 0;

    // Debug-visualization only (see docs/debug.html and debugSnapshot()
    // below) - which memory index getConcealer most recently chose, and
    // the pass/fail VAD verdict for each candidate clip _feedMemory's last
    // cycle produced (not just the ones that made it into memory - the
    // rejected ones matter for the visualization too).
    this.lastSelectedIndex = null;
    this.lastFeedCandidates = [];
    this.denoiseRunning = false;
    // memVadSeq starts at 0 ("no cycle has run yet") and increments once per
    // completed cycle - lets main.js's display tell "a new cycle just
    // finished" apart from "still the same cycle as last poll" without
    // relying on array-reference equality across the postMessage boundary.
    // Purely a display counter, read nowhere else in this file.
    this.memVadSeq = 0;
    this.lastMemVadDurationMs = 0;
    this.feedCycleSeq = 0; // same idea, but incremented when a cycle STARTS rather than when it finishes - see debugSnapshot()
  }

  status() {
    return {
      ready: this.memory.length >= this.minMemoryToConceal,
      progress: this.memory.length,
      target: this.memoryMaxlen,
      threshold: this.minMemoryToConceal,
      label: 'Memory',
    };
  }

  /**
   * Debug-visualization only (see docs/debug.html) - a snapshot of internal
   * state that status() deliberately doesn't expose (it's UI-facing, this
   * is purely for the concealer diagram). Not used for anything the
   * concealer itself depends on.
   */
  debugSnapshot() {
    const lengthCondition = Math.max(0, this.memory.length - this.maxConcealerDistanceToBuffer);
    return {
      memory: this.memory.map((entry, i) => ({
        cooldown: entry.cooldown,
        tooCloseToBuffer: i >= lengthCondition,
      })),
      memoryMaxlen: this.memoryMaxlen,
      selectedIndex: this.curConcealing ? this.lastSelectedIndex : null,
      lastCandidates: this.lastFeedCandidates,
      denoiseRunning: this.denoiseRunning,
      memVadSeq: this.memVadSeq,
      memVadDurationMs: this.lastMemVadDurationMs,
      feedCycleSeq: this.feedCycleSeq,
      candidateSlots: this.nConcealerBeforeDenoise, // fixed constant, known before any cycle actually runs
    };
  }

  /** While speech is active: extend the pending-voice window with x, start (or keep advancing) the pending-speech window, and count down any active concealing cooldown. */
  update(x) {
    for (let i = 0; i < x.length; i++) this.pendingVoice.push(x[i]);
    const maxLen = Math.trunc(this.pendingVoiceMaxDuration * this.sr);
    if (this.pendingVoice.length > maxLen) this.pendingVoice.splice(0, this.pendingVoice.length - maxLen);

    if (!this.pendingSpeechActive) {
      this.pendingSpeechActive = true; // voice just started after being idle - the fixed window begins now
      this.pendingSpeechSamples = 0;
    }
    this.pendingSpeechSamples += x.length;

    if (this.curConcealing) {
      this.concealingCountdown -= x.length;
      if (this.concealingCountdown <= 0) {
        this.curConcealing = false;
        this.concealingCountdown = 0;
      }
    }
  }

  /** During silence: keep extending the pending-voice window with x, keep advancing an already-started pending-speech window, but drop any in-progress concealing state. */
  refresh(x) {
    for (let i = 0; i < x.length; i++) this.pendingVoice.push(x[i]);
    const maxLen = Math.trunc(this.pendingVoiceMaxDuration * this.sr);
    if (this.pendingVoice.length > maxLen) this.pendingVoice.splice(0, this.pendingVoice.length - maxLen);
    // Once a pending-speech window has started, silence doesn't pause or
    // cancel it - it keeps counting toward the fixed pendingVoiceMaxDuration
    // deadline exactly like a voice-active chunk would (see updateMemory).
    if (this.pendingSpeechActive) this.pendingSpeechSamples += x.length;
    this.curConcealing = false;
    this.concealingCountdown = 0;
  }

  /**
   * Age existing memory entries' cooldowns every concealer_duration worth
   * of audio, and, once a pending-speech window (started by update(), see
   * its comment) has run for a full pendingVoiceMaxDuration, feed the
   * current pendingVoice snapshot into memory. In streaming mode this is
   * fire-and-forget (like Python's daemon background thread - the caller
   * doesn't wait for it to finish); otherwise it's awaited, matching
   * Python's synchronous (non-streaming) call to _background_task. Must be
   * called every chunk regardless of voice activity (see stream.js) - a
   * window that started on a voice-active chunk still needs to be checked
   * (and can still complete/fire) on the silent chunks that follow it.
   */
  async updateMemory(x) {
    this.curSizeBeforeUpdateMemory += x.length;
    if (this.curSizeBeforeUpdateMemory >= this.concealerSize) {
      for (const entry of this.memory) entry.cooldown = Math.max(0, entry.cooldown - 1);
      this.curSizeBeforeUpdateMemory = 0;
    }

    if (this.freezeLearning) return;
    const maxLen = Math.trunc(this.pendingVoiceMaxDuration * this.sr);
    const pendingVoiceFull = this.pendingVoice.length >= maxLen;
    const pendingSpeechWindowComplete = this.pendingSpeechActive && this.pendingSpeechSamples >= maxLen;
    if (!this.stopProcessing && pendingVoiceFull && pendingSpeechWindowComplete) {
      this.stopProcessing = true;
      const snapshot = Float32Array.from(this.pendingVoice);
      if (this.isStream) {
        this._feedMemory(snapshot).catch((e) => console.error('GranSpeechMask._feedMemory failed:', e));
      } else {
        await this._feedMemory(snapshot);
      }
    }
  }

  /**
   * Split y into concealer_duration-sized clips, denoise them (if enabled),
   * fade their edges, extract each clip's features from its non-denoised
   * original, keep only the clips the concealer's own VAD still calls
   * speech, and merge the result into memory (queue strategy: drop oldest
   * past memoryMaxlen).
   */
  async _feedMemory(y, voiceName = '') {
    // newVoiceSinceFeed/stopProcessing must reset even if this throws (e.g.
    // the remote denoiser worker failed to load its model) - otherwise a
    // single failure would permanently wedge memory refresh, since
    // stopProcessing is only ever cleared below. Worth guarding explicitly
    // now that the denoiser call crosses a Worker boundary (see
    // denoiser-proxy.js) and so has real, independent failure modes that
    // didn't exist when it ran in-process.
    this.feedCycleSeq += 1; // debug-visualization only - see debugSnapshot()
    try {
      const newMemory = [];
      let denoised = y;
      if (this.denoise && this.denoiser) {
        this.denoiseRunning = true; // debug-visualization only - see debugSnapshot()
        try {
          denoised = await this.denoiser.predict(y);
          if (denoised.length > y.length) denoised = denoised.subarray(0, y.length);
          else if (denoised.length < y.length) {
            const padded = new Float32Array(y.length);
            padded.set(denoised);
            denoised = padded;
          }
        } finally {
          this.denoiseRunning = false;
        }
      }

      // Matches numpy.array_split(y, n): the first `remainder` chunks get
      // one extra sample, not just the last one.
      const n = this.nConcealerBeforeDenoise;
      const base = Math.floor(y.length / n);
      const remainder = y.length % n;
      const fadeSize = Math.max(Math.trunc(this.sr * this.fadeDuration), Math.trunc(this.sr / 100)); // min 10ms

      const candidateResults = [];
      const memVadStartedAt = performance.now(); // debug-visualization only - see debugSnapshot()
      let start = 0;
      for (let i = 0; i < n; i++) {
        const size = base + (i < remainder ? 1 : 0);
        const end = start + size;
        if (start >= end) { start = end; continue; }
        let clip = applyFade(denoised.subarray(start, end), fadeSize);
        const clipOriginal = y.subarray(start, end);
        // Features come from the original (non-denoised) segment - the live
        // query side (pendingVoice) is raw mic audio, never denoised, so
        // matching against denoised candidate features would compare across
        // two different audio domains.
        const feat = featureExtractor(clipOriginal, this.sr, { norm: true });
        // eslint-disable-next-line no-await-in-loop
        const passed = await this.vad.predict(clip);
        candidateResults.push(passed);
        if (passed) {
          newMemory.push({ clip: Float32Array.from(clip), feat, cooldown: 0, voiceName });
        }
        start = end;
      }
      // debug-visualization only, all three - see debugSnapshot()
      this.lastFeedCandidates = candidateResults;
      this.lastMemVadDurationMs = performance.now() - memVadStartedAt;
      this.memVadSeq += 1;

      while (this.memory.length < this.memoryMaxlen && newMemory.length > 0) {
        this.memory.push(newMemory.shift());
      }
      this.memory = this.memory.concat(newMemory);
      while (this.memory.length > this.memoryMaxlen) this.memory.shift();

      if (!this.isStream) this.pendingVoice = [];
    } finally {
      this.pendingSpeechActive = false;
      this.pendingSpeechSamples = 0;
      this.stopProcessing = false;
    }
  }

  /**
   * Extend pendingVoice/the pending-speech window with x (updateMemory - the
   * check that can actually trigger a memory feed from that window - is now
   * the caller's job, run unconditionally every chunk; see stream.js), then,
   * if not already mid-concealment and the memory is large enough, pick the
   * clip whose features best match the recent live buffer (cosine distance,
   * falling back to a random eligible clip if none are eligible), put it on
   * cooldown, and return it energy-normalized to match the live buffer.
   * Otherwise returns null (silence). Returns { audio: Float32Array|null, voiceName: string }.
   */
  async getConcealer(x) {
    this.update(x); // extends pendingVoice with x, so it's included below

    const shouldConceal = !this.curConcealing && this.memory.length >= this.minMemoryToConceal;
    if (!shouldConceal) return { audio: null, voiceName: '' };

    this.curConcealing = true;

    const decisionWinSamples = Math.trunc(this.sr * this.decisionWin);
    const pv = this.pendingVoice;
    const decisionAudio = Float32Array.from(pv.slice(Math.max(0, pv.length - decisionWinSamples)));
    const xFeat = featureExtractor(decisionAudio, this.sr, { norm: true });
    const xFeatNorm = Math.hypot(...xFeat);

    const lengthCondition = Math.max(0, this.memory.length - this.maxConcealerDistanceToBuffer);

    let minDistance = Infinity;
    let bestIdx = 0;
    for (let i = 0; i < this.memory.length; i++) {
      const entry = this.memory[i];
      let distance = Infinity;
      if (entry.cooldown === 0 && i < lengthCondition) {
        let dot = 0, clipNorm = 0;
        for (let k = 0; k < xFeat.length; k++) {
          dot += xFeat[k] * entry.feat[k];
          clipNorm += entry.feat[k] * entry.feat[k];
        }
        clipNorm = Math.sqrt(clipNorm);
        distance = 1 - dot / (xFeatNorm * (clipNorm + 1e-10));
      }
      if (distance < minDistance) {
        minDistance = distance;
        bestIdx = i;
      }
    }
    if (minDistance === Infinity) {
      const randomIdxRange =
        this.memory.length - this.maxConcealerDistanceToBuffer >= 0
          ? this.memory.length - this.maxConcealerDistanceToBuffer
          : this.memory.length;
      bestIdx = randomIdxRange > 0 ? Math.floor(Math.random() * randomIdxRange) : 0;
    }

    const chosen = this.memory[bestIdx];
    chosen.cooldown = this.maxCountdownReuse;
    this.lastSelectedIndex = bestIdx; // debug-visualization only - see debugSnapshot()

    // Normalize energy to match the live buffer.
    let liveMeanSq = 0;
    for (let i = 0; i < pv.length; i++) liveMeanSq += pv[i] * pv[i];
    liveMeanSq /= pv.length;
    let clipMeanSq = 0;
    for (let i = 0; i < chosen.clip.length; i++) clipMeanSq += chosen.clip[i] * chosen.clip[i];
    clipMeanSq /= chosen.clip.length;
    const energyScale = Math.sqrt(liveMeanSq) / Math.sqrt(clipMeanSq);
    const concealer = new Float32Array(chosen.clip.length);
    for (let i = 0; i < concealer.length; i++) concealer[i] = chosen.clip[i] * energyScale;

    const minTimeout = Math.trunc(concealer.length * this.concealingMinTimeoutRatio);
    const maxTimeout = Math.trunc(concealer.length * this.concealingMaxTimeoutRatio);
    this.concealingCountdown =
      maxTimeout > minTimeout ? minTimeout + Math.floor(Math.random() * (maxTimeout - minTimeout)) : minTimeout;

    return { audio: concealer, voiceName: chosen.voiceName };
  }
}
