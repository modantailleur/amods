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
    this.newVoiceSinceFeed = 0;
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

  /** While speech is active: extend the pending-voice window with x and count down any active concealing cooldown. */
  update(x) {
    for (let i = 0; i < x.length; i++) this.pendingVoice.push(x[i]);
    const maxLen = Math.trunc(this.pendingVoiceMaxDuration * this.sr);
    if (this.pendingVoice.length > maxLen) this.pendingVoice.splice(0, this.pendingVoice.length - maxLen);
    this.newVoiceSinceFeed += x.length;

    if (this.curConcealing) {
      this.concealingCountdown -= x.length;
      if (this.concealingCountdown <= 0) {
        this.curConcealing = false;
        this.concealingCountdown = 0;
      }
    }
  }

  /** During silence: keep extending the pending-voice window with x, but drop any in-progress concealing state. */
  refresh(x) {
    for (let i = 0; i < x.length; i++) this.pendingVoice.push(x[i]);
    const maxLen = Math.trunc(this.pendingVoiceMaxDuration * this.sr);
    if (this.pendingVoice.length > maxLen) this.pendingVoice.splice(0, this.pendingVoice.length - maxLen);
    this.curConcealing = false;
    this.concealingCountdown = 0;
  }

  /**
   * Age existing memory entries' cooldowns every concealer_duration worth
   * of audio, and, once enough fresh pending voice has accumulated, feed
   * it into memory. In streaming mode this is fire-and-forget (like
   * Python's daemon background thread - the caller doesn't wait for it to
   * finish); otherwise it's awaited, matching Python's synchronous
   * (non-streaming) call to _background_task.
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
    const hasFreshMaterial = !this.isStream || this.newVoiceSinceFeed >= maxLen;
    if (!this.stopProcessing && pendingVoiceFull && hasFreshMaterial) {
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
    try {
      const newMemory = [];
      let denoised = y;
      if (this.denoise && this.denoiser) {
        denoised = await this.denoiser.predict(y);
        if (denoised.length > y.length) denoised = denoised.subarray(0, y.length);
        else if (denoised.length < y.length) {
          const padded = new Float32Array(y.length);
          padded.set(denoised);
          denoised = padded;
        }
      }

      // Matches numpy.array_split(y, n): the first `remainder` chunks get
      // one extra sample, not just the last one.
      const n = this.nConcealerBeforeDenoise;
      const base = Math.floor(y.length / n);
      const remainder = y.length % n;
      const fadeSize = Math.max(Math.trunc(this.sr * this.fadeDuration), Math.trunc(this.sr / 100)); // min 10ms

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
        if (await this.vad.predict(clip)) {
          newMemory.push({ clip: Float32Array.from(clip), feat, cooldown: 0, voiceName });
        }
        start = end;
      }

      while (this.memory.length < this.memoryMaxlen && newMemory.length > 0) {
        this.memory.push(newMemory.shift());
      }
      this.memory = this.memory.concat(newMemory);
      while (this.memory.length > this.memoryMaxlen) this.memory.shift();

      if (!this.isStream) this.pendingVoice = [];
    } finally {
      this.newVoiceSinceFeed = 0;
      this.stopProcessing = false;
    }
  }

  /**
   * Grow the memory with x, then, if not already mid-concealment and the
   * memory is large enough, pick the clip whose features best match the
   * recent live buffer (cosine distance, falling back to a random eligible
   * clip if none are eligible), put it on cooldown, and return it energy-
   * normalized to match the live buffer. Otherwise returns null (silence).
   * Returns { audio: Float32Array|null, voiceName: string }.
   */
  async getConcealer(x) {
    await this.updateMemory(x);
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
