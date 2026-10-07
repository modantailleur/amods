// Dedicated Worker (not the audio thread, not the main/UI thread) running
// the actual amods.Stream port: VAD, GranSpeechMask, limiter. Talks to the
// main thread via postMessage; the main thread relays raw mic chunks in
// (from the AudioWorkletNode's port) and processed audio back out.
//
// The denoiser itself deliberately does NOT live here, even though
// GranSpeechMask's memory-refresh calls it directly - see
// denoiser-proxy.js's header comment: a single Worker has only one thread,
// so a ~1s denoiser inference call running on this same thread would block
// real-time chunk processing for its whole duration, however "background"
// it looks in the JS source. RemoteDenoiser instead hands that call off to
// docs/js/denoiser-worker.js, a second dedicated Worker, via the main
// thread relay - see main.js's start().
//
// Only TenVAD is a static top-level import - it's the default/recommended
// backend and its own WASM module is tiny (~283KB). onnxruntime-web and
// SileroVAD are dynamically imported instead, ONLY when the "VAD" dropdown
// (see index.html/debug.html, above "Concealing likelihood") is set to
// Silero, so choosing TEN VAD never pays to load onnxruntime-web or the
// ~1.3MB silero_vad.onnx model at all.
import { TenVAD } from './vad-ten.js';
import { GranSpeechMask } from './granspeechmask.js';
import { ConcealerStream } from './stream.js';
import { RemoteDenoiser } from './denoiser-proxy.js';
import { applyFadeIn } from './audio-utils.js';
import { installDebugLogRelay } from './debug-log.js';
import { SpectrogramAnalyzer } from './spectrogram.js';
import { EchoCanceller } from './echo-canceller.js';

let stream = null;
let concealer = null;
let sourceVad = null;
let remoteDenoiser = null;
let sr = 48000;
let debugTelemetry = false; // set from cfg.debugTelemetry - see docs/debug.html; gates the extra debugSnapshot() work in startStatusLoop below so plain index.html never pays for it
// debug.html's "Visualization" panel's three spectrograms (Source/
// Concealer/Noise) - only ever constructed while BOTH debugTelemetry is
// true AND the panel's own "Visualization" toggle is on (vizEnabled, see
// setVizEnabled below) - so a plain index.html session, or a debug.html
// session with the toggle off, never pays for the extra per-chunk FFT
// work (3x SpectrogramAnalyzer.computeColumn calls) or holds the three
// analyzers' rolling buffers/mel filterbanks (~1.1MB total) in memory. A
// real disconnect when toggled off, same reasoning as setConcealerEnabled/
// setNoiseEnabled in stream.js - re-enabling starts the three analyzers
// fresh (empty rolling buffers), same as a fresh session. 90 matches the
// <canvas height="90"> set in debug.html - see SpectrogramView's own
// comment for why the two must agree.
const SPECTROGRAM_DISPLAY_BINS = 90;
let vizEnabled = false;
let speakerModeEnabled = false; // set from cfg.speakerModeEnabled - see "Speaker mode"/EchoCanceller below; read here only to decide whether createSpecAnalyzers' micAec is worth constructing at all
let specAnalyzers = null; // { mic, micAec, concealer, noise } once constructed

function createSpecAnalyzers() {
  return {
    mic: new SpectrogramAnalyzer(sr, SPECTROGRAM_DISPLAY_BINS),
    // Only while Speaker mode is actually on - otherwise stream.js's
    // processChunk always returns micAecBlock: null (nothing to show), so
    // there's no point spending a 4th analyzer's FFT work/buffers on it.
    micAec: speakerModeEnabled ? new SpectrogramAnalyzer(sr, SPECTROGRAM_DISPLAY_BINS) : null,
    concealer: new SpectrogramAnalyzer(sr, SPECTROGRAM_DISPLAY_BINS),
    noise: new SpectrogramAnalyzer(sr, SPECTROGRAM_DISPLAY_BINS),
  };
}

async function init(cfg) {
  sr = cfg.sr;
  debugTelemetry = Boolean(cfg.debugTelemetry);
  // This Worker has its own separate console from the main thread's, so
  // main.js's own installDebugLogRelay('main') call never sees anything
  // logged in here - see debug-log.js for why both realms need their own.
  if (debugTelemetry) installDebugLogRelay('worker');
  speakerModeEnabled = Boolean(cfg.speakerModeEnabled);
  vizEnabled = debugTelemetry && Boolean(cfg.vizEnabled);
  specAnalyzers = vizEnabled ? createSpecAnalyzers() : null;
  // "Speaker mode" - see echo-canceller.js's own header for the algorithm/
  // citation/why. null while off, the exact original/default behavior (see
  // stream.js's processChunk - it no-ops entirely when this is null).
  const echoCanceller = speakerModeEnabled ? new EchoCanceller(cfg.sr) : null;

  // Three SEPARATE VAD instances, NEVER sharing one underlying model
  // session/module - sourceVad and concealerVad are called from genuinely
  // independent, interleavable call chains (sourceVad every ~50ms from the
  // real-time chunk loop; concealerVad in a burst from _feedMemory, which
  // runs fire-and-forget alongside that same loop - see updateMemory), and
  // noiseVad (the "Noise controls" panel's own "VAD" dropdown) gates
  // SpeechShapedNoise's calibration input behind "Speech-shaped purity" -
  // see stream.js's processChunk. Sharing one onnxruntime-web
  // InferenceSession between two of these once silently corrupted sourceVad
  // the first time concealerVad ever ran - not worth risking again for any
  // backend. cfg.vadType (Concealer's VAD) and cfg.noiseVadType (Noise's)
  // are intentionally independent dropdowns - either, both, or neither may
  // be 'silero', so onnxruntime-web/the Silero model are only loaded at all
  // if at least one of them actually needs it.
  let concealerVad;
  let noiseVad;
  const needsSilero = cfg.vadType === 'silero' || cfg.noiseVadType === 'silero';
  if (needsSilero) {
    const [{ default: ort }, { SileroVAD }] = await Promise.all([
      import('../vendor/ort.min.mjs'),
      import('./vad-silero.js'),
    ]);
    globalThis.ort = ort;
    // Single-threaded WASM: works on plain GitHub Pages, no cross-origin-
    // isolation (COOP/COEP) headers required - at some performance cost vs.
    // multi-threaded WASM, which needs SharedArrayBuffer and those headers.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = '../vendor/';
    const makeVad = async (type, opts) => {
      if (type === 'silero') {
        const session = await ort.InferenceSession.create('../models/silero_vad.onnx');
        return new SileroVAD(session, opts);
      }
      return new TenVAD(opts);
    };
    [sourceVad, concealerVad, noiseVad] = await Promise.all([
      makeVad(cfg.vadType === 'silero' ? 'silero' : 'ten', { logitThreshold: cfg.concealingThreshold, sr: cfg.sr, name: 'source', debug: debugTelemetry }),
      makeVad(cfg.vadType === 'silero' ? 'silero' : 'ten', { logitThreshold: cfg.concealerMemoryThreshold, sr: cfg.sr, name: 'concealer', debug: debugTelemetry }),
      // logitThreshold: null - predict() then returns the raw speech ratio
      // instead of a bool (see vad-ten.js's own contract comment).
      // "Speech-shaped purity" is applied live from streamConfig in
      // stream.js's processChunk instead of being baked in here, so the
      // live slider stays in sync with no separate setter needed to also
      // push the value into this instance.
      makeVad(cfg.noiseVadType === 'silero' ? 'silero' : 'ten', { logitThreshold: null, sr: cfg.sr, name: 'noise', debug: debugTelemetry }),
    ]);
  } else {
    sourceVad = new TenVAD({ logitThreshold: cfg.concealingThreshold, sr: cfg.sr, name: 'source', debug: debugTelemetry });
    concealerVad = new TenVAD({ logitThreshold: cfg.concealerMemoryThreshold, sr: cfg.sr, name: 'concealer', debug: debugTelemetry });
    noiseVad = new TenVAD({ logitThreshold: null, sr: cfg.sr, name: 'noise', debug: debugTelemetry });
  }
  remoteDenoiser = cfg.denoiserEnabled ? new RemoteDenoiser() : null;

  const concealerConfig = {
    memory_maxlen: 100,
    min_memory_to_conceal: 20,
    max_countdown_reuse: 10,
    max_concealer_distance_to_buffer: 5,
    concealer_duration: 0.3,
    // From the "Concealing overlap" slider's 3 discrete stops (see main.js's
    // CONCEALING_OVERLAP_STOPS) - live-tunable via 'setConcealingOverlap'
    // below, same as concealingThreshold/concealerMemoryThreshold.
    concealing_min_timeout_ratio: cfg.concealingOverlapMinRatio ?? 0.6,
    concealing_max_timeout_ratio: cfg.concealingOverlapMaxRatio ?? 1.0,
    // From the "Concealing smoothness" slider (see main.js's
    // smoothnessToFadeDuration) - live-tunable via 'setFadeDuration' below,
    // same as concealingThreshold/concealerMemoryThreshold.
    fade_duration: cfg.fadeDuration ?? 0.05,
    is_stream: true,
    pending_conc_max_size: 3 * cfg.sr,
    freeze_learning: false,
    decision_win: 0.3,
    denoise: cfg.denoiserEnabled,
    // "FbdeDM" - see deesser.js; applied to the pendingVoice snapshot
    // BEFORE denoising, inside _feedMemory.
    de_ess: Boolean(cfg.deEsserEnabled),
    // concealing_noise (concealer-noise.js's per-clip blur/noiseShape) is
    // deliberately left unset here (defaults to 0 inside GranSpeechMask) -
    // that was a misreading of what "concealing noise" was meant to be; the
    // real feature is the ConcealerStream-level background bed below
    // (concealing_noise_level/_sensitivity in streamConfig, see
    // speech-shaped-noise.js). The old code stays in the repo, unreachable.
  };
  concealer = new GranSpeechMask(cfg.sr, concealerConfig, { denoiser: remoteDenoiser, vad: concealerVad });

  // Always-on (not debugTelemetry-gated - see the [HEALTH] log below for why)
  // - a single line establishing the session's config up front, so a
  // supervisor's pasted-in console log is diagnosable on its own without
  // needing to ask them what settings they were running with.
  console.log('[SESSION START]', JSON.stringify({
    sr: cfg.sr,
    vadType: cfg.vadType === 'silero' ? 'silero' : 'ten',
    noiseVadType: cfg.noiseVadType === 'silero' ? 'silero' : 'ten',
    concealingThreshold: cfg.concealingThreshold,
    concealerMemoryThreshold: cfg.concealerMemoryThreshold,
    denoiserEnabled: Boolean(cfg.denoiserEnabled),
    speakerModeEnabled,
    memoryMaxlen: concealerConfig.memory_maxlen,
    minMemoryToConceal: concealerConfig.min_memory_to_conceal,
  }));

  const streamConfig = {
    sr: cfg.sr,
    channels_out: 1,
    monitor_gain: cfg.monitorGain ?? 0.0,
    record_mic_gain: 1.0,
    conc_multiplier: cfg.concMultiplier ?? 1.0,
    // Master on/off for each section's whole feature, from the toggle
    // next to the "Concealer"/"Noise" sur-titles (see stream.js's own
    // comment on concealerEnabled/noiseEnabled) - a real disable
    // (zeroes the final audio contribution entirely), not a debug-only
    // monitor toggle like listen_concealer/listen_noise below.
    concealer_enabled: cfg.concealerEnabled ?? true,
    // Defaults false (unlike concealer_enabled above) - noise is opt-in.
    noise_enabled: cfg.noiseEnabled ?? false,
    // "Noise controls" panel's background bed - level (the "Noise level"
    // dB fader, see main.js's dbToMultiplier), the EMA time constant in
    // seconds that the "Sensitivity" slider sets for the speech-shaped
    // color specifically (how far back its spectral envelope remembers -
    // NOT a buffer size, and unused by white/pink), and which of the
    // three colors (see noise-generators.js and speech-shaped-noise.js)
    // is active - see ConcealerStream's own _noiseGen/noiseType. All
    // three live-tunable via 'setConcealingNoiseLevel'/
    // 'setConcealingNoiseSensitivity'/'setConcealingNoiseType' below.
    concealing_noise_level: cfg.concealingNoiseLevel ?? 0,
    concealing_noise_sensitivity: cfg.concealingNoiseSensitivity ?? 2,
    noise_type: cfg.noiseType ?? 'speechShaped',
    // "Speech-shaped purity" slider (0-0.9) - 0 means the noiseVad gate in
    // stream.js's processChunk is off entirely (feed the speechShaped
    // generator unconditionally, same as before this existed); above 0,
    // only audio whose noiseVad speech ratio exceeds this value is fed in.
    // Live-tunable via 'setNoiseShapedPurity' below, read fresh every chunk.
    noise_shaped_purity: cfg.noiseShapedPurity ?? 0,
    // Debug-only listen toggles (see docs/debug.html) - index.html has no UI
    // for these, so they just stay at these defaults (concealer-only, same
    // as production always sounded before this existed) for every real
    // session. listen_noise defaults to true (not false, unlike the other
    // two) - production always has the noise bed audible when its own
    // level is above zero; "Noise track" is a debug-only A/B toggle, not
    // something production itself needs an opinion on muting by default.
    listen_original: cfg.listenOriginal ?? false,
    listen_concealer: cfg.listenConcealer ?? true,
    listen_noise: cfg.listenNoise ?? true,
  };
  stream = new ConcealerStream(streamConfig, sourceVad, concealer, noiseVad, echoCanceller, debugTelemetry);

  postMessage({ type: 'ready' });
  startStatusLoop();
}

let statusTimer = null;
let healthLogTicks = 0;
// Every 10th 500ms status tick = ~5s - frequent enough to catch a problem
// shortly after it starts, infrequent enough that a copy-pasted console
// excerpt covering a whole session stays a manageable size.
const HEALTH_LOG_EVERY_N_TICKS = 10;
// stream.popCallbackTiming() is popped every 500ms for the UI-facing
// 'status' message below - the [HEALTH] log fires only every 10th tick, so
// each 500ms slice's numbers are accumulated here in between, rather than
// the log only ever reflecting the most recent 500ms and silently dropping
// the other ~4.5s of chunks.
let healthMsSum = 0;
let healthMsMax = 0;
let healthMsCount = 0;
let healthSlowCount = 0;

function startStatusLoop() {
  if (statusTimer) return;
  statusTimer = setInterval(() => {
    if (!concealer || !stream) return;
    // DIAGNOSTIC (debug.html only) - kept in place, same reasoning as
    // vad-silero.js's [VADDIAG]: this is what showed stopProcessing/
    // pendingSpeechActive/the queue were all behaving correctly, isolating
    // the shared-InferenceSession bug down to the VAD calls themselves.
    if (debugTelemetry) {
      console.error(`[ENGINEDIAG] processing=${processing} stopProcessing=${concealer.stopProcessing} pendingQueue.length=${pendingQueue.length} pendingSpeechActive=${concealer.pendingSpeechActive}`);
    }
    const status = concealer.status();
    const timing = stream.popCallbackTiming();
    // voiceActive is NOT included here anymore - see drainQueue's dedicated
    // per-chunk 'rtVad' message below, which is what the dot actually reacts
    // to now (~50ms cadence instead of this 500ms poll).
    const debug = debugTelemetry ? concealer.debugSnapshot() : undefined;
    // null whenever there's nothing worth showing (Speaker mode off, ramp
    // not started yet, or already finished) - see stream.js's own comment
    // on getConcealerRampStatus/CONCEALER_RAMP_SECONDS. Read-only (polling
    // this every ~500ms for the UI has no effect on the ramp itself).
    const echoCalibration = stream.getConcealerRampStatus();
    postMessage({ type: 'status', status, timing, debug, echoCalibration });

    // Accumulate this 500ms slice's timing into the ~5s [HEALTH] window
    // (see the accumulator declarations above) - timing was already popped
    // for the 'status' message above, so this is the only place it's read.
    healthMsSum += timing.avgMs * timing.count;
    healthMsMax = Math.max(healthMsMax, timing.maxMs);
    healthMsCount += timing.count;
    healthSlowCount += timing.slowCount;

    // Always-on health summary (NOT gated behind debugTelemetry, unlike
    // [ENGINEDIAG]/[VADDIAG]/[STREAMDIAG] above/elsewhere - those only ever
    // run on debug.html, but a supervisor reporting "the memory isn't
    // filling" is running plain index.html with no debug panel at all, so
    // this is the only signal that will ever reach their DevTools console.
    // One JSON-structured line every ~5s: low enough volume that a
    // copy-pasted excerpt of a whole session is still a reasonable size, but
    // frequent enough to catch a stall shortly after it starts. Kept
    // separate from the periodic 'status'/postMessage above since that's for
    // the UI, not for reading directly - this is a console.log, for humans
    // (and for pasting back into a conversation to debug from).
    healthLogTicks += 1;
    if (healthLogTicks >= HEALTH_LOG_EVERY_N_TICKS) {
      healthLogTicks = 0;
      const sourceVadStats = sourceVad ? sourceVad.popStats() : null;
      const concealerVadStats = concealer.vad ? concealer.vad.popStats() : null;
      const memoryStats = concealer.popMemoryHealthStats();
      console.log('[HEALTH]', JSON.stringify({
        queueLength: pendingQueue.length,
        chunksSkipped: skipCount,
        callbackTiming: {
          avgMs: healthMsCount > 0 ? Math.round((healthMsSum / healthMsCount) * 10) / 10 : 0,
          maxMs: Math.round(healthMsMax * 10) / 10,
          count: healthMsCount,
          slowCount: healthSlowCount,
        },
        sourceVad: sourceVadStats,
        concealerVad: concealerVadStats,
        memory: memoryStats,
      }));
      skipCount = 0;
      healthMsSum = 0;
      healthMsMax = 0;
      healthMsCount = 0;
      healthSlowCount = 0;
    }
  }, 500);
}

// On a slow device (weak mobile CPU, a background-tab GC pause, or just the
// concealer's own memory-refresh cycle briefly adding VAD calls on this
// same thread - see granspeechmask.js's _feedMemory clip loop), occasionally
// exceeding the 50ms per-chunk budget is a transient hiccup. Two very
// different things depend on every chunk being processed in order with no
// gaps: real-time playback (which also has a latency budget - stale audio
// played back late is worse than briefly skipping it) and GranSpeechMask's
// pendingVoice/memory recording (which has NO latency budget - it can run
// behind without the user noticing, but a genuine temporal gap in it gets
// baked permanently into a stored memory clip, replayed every time that
// clip is later selected - not just heard once, so it must never happen,
// full stop, no matter how overloaded the device is).
//
// So every chunk always goes through stream.processChunk() in full,
// eventually, in order, with nothing skipped - that's what makes
// pendingVoice's recording genuinely gap-free rather than just gap-free-
// with-the-edit-points-faded-over (which was this file's previous
// approach, and doesn't actually restore the missing audio, just its
// amplitude continuity). The ONLY thing that gets sacrificed under real
// overload is whether a given chunk's result is still worth sending to the
// speaker once computed - see MAX_STALE_MS below.
let processing = false;
const pendingQueue = [];
// Purely a safety valve against unbounded memory growth if a device is so
// overloaded it can never catch up (each item is ~9.6KB, so even this many
// queued is only ~2MB) - NOT a latency bound (see MAX_STALE_MS for that).
// Should never actually trigger in practice; if it does, the device
// genuinely cannot keep up in real time and something has to give.
const MAX_QUEUE = 200;
// A chunk's playMix is still computed - and its effect on pendingVoice/
// concealer state already took effect - regardless of staleness; this only
// decides whether it's still worth sending to the speaker. Older than this
// and playing it back would feel more like a confusing delayed echo than
// real-time conversation, so it's skipped instead.
const MAX_STALE_MS = 150;
let hadSkip = false;
// Always-on count of chunks skipped for staleness since the last [HEALTH]
// log - read-and-reset there, same pattern as stream.js's _callbackSlowCount.
// A high rate here (relative to how many chunks arrived) is the direct
// signature of the device falling behind in real time.
let skipCount = 0;
const FADE_AFTER_SKIP_S = 0.01; // smooths the deliberate join back to fresh audio after a stale run was skipped - an intentional real-time trade-off, not a fix for lost data

async function handleChunk(chunk) {
  if (!stream) return;
  if (pendingQueue.length >= MAX_QUEUE) pendingQueue.shift();
  pendingQueue.push({ chunk, arrivedAt: performance.now() });
  drainQueue();
}

async function drainQueue() {
  if (processing) return;
  const item = pendingQueue.shift();
  if (!item) return;
  processing = true;
  try {
    let { playMix, micBlock, micAecBlock, concealerBlock, noiseBlock } = await stream.processChunk(item.chunk);
    // Debug-visualization only - sent every chunk (~50ms), unconditionally
    // (not gated on staleMs below - that only decides whether THIS chunk's
    // audio is still worth playing, not whether the dot's state is still
    // worth showing), so the on-screen dot reacts at real detection speed
    // instead of only refreshing once per ~500ms status poll.
    if (debugTelemetry) postMessage({ type: 'rtVad', voiceActive: stream.lastVoiceActivity });
    // debug.html's "Visualization" spectrograms - same unconditional-every-
    // chunk cadence as rtVad above, for smooth scrolling. specAnalyzers is
    // only non-null when debugTelemetry is true (see init()), so this is a
    // single cheap null-check on a plain index.html session, not a branch
    // that still does the FFT work and discards it.
    if (specAnalyzers) {
      specAnalyzers.mic.push(micBlock);
      specAnalyzers.concealer.push(concealerBlock);
      specAnalyzers.noise.push(noiseBlock);
      const mic = specAnalyzers.mic.computeColumn();
      const concealerCol = specAnalyzers.concealer.computeColumn();
      const noise = specAnalyzers.noise.computeColumn();
      const transfer = [mic.buffer, concealerCol.buffer, noise.buffer];
      const msg = { type: 'specFrame', mic, concealer: concealerCol, noise };
      // micAec: genuinely distinct data from mic above (the actual AEC3-
      // processed signal, not a duplicate) - only exists while Speaker mode
      // is on (see createSpecAnalyzers' own comment).
      if (specAnalyzers.micAec && micAecBlock) {
        specAnalyzers.micAec.push(micAecBlock);
        msg.micAec = specAnalyzers.micAec.computeColumn();
        transfer.push(msg.micAec.buffer);
      }
      postMessage(msg, transfer);
    }
    const staleMs = performance.now() - item.arrivedAt;
    if (staleMs <= MAX_STALE_MS) {
      if (hadSkip) {
        playMix = applyFadeIn(playMix, Math.round(FADE_AFTER_SKIP_S * sr));
        hadSkip = false;
      }
      postMessage({ type: 'play', playMix }, [playMix.buffer]);
    } else {
      hadSkip = true; // too stale to play, but pendingVoice/concealer state above already saw it in full - memory continuity is never affected by this
      skipCount += 1;
    }
  } catch (e) {
    console.error('[ERROR] worker-engine processChunk failed (this chunk is dropped entirely - no audio played, no memory/VAD state updated for it):', e);
  } finally {
    processing = false;
    drainQueue();
  }
}

self.onmessage = (event) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      init(msg.config).catch((e) => {
        console.error('[ERROR] worker-engine init failed (session never started - check vadType/denoiser model loading above):', e);
        postMessage({ type: 'error', message: String(e) });
      });
      break;
    case 'chunk':
      handleChunk(msg.chunk);
      break;
    case 'setConcMultiplier':
      if (stream) stream.streamConfig.conc_multiplier = msg.value;
      break;
    case 'setConcealerEnabled':
      // Master on/off for the whole "Concealer" section - routed through
      // ConcealerStream's own method (see its comment), not just poking
      // streamConfig: it skips calling into GranSpeechMask/sourceVad
      // entirely while off (no VAD/denoiser CPU cost) and frees
      // GranSpeechMask's own memory/pendingVoice allocations - a real
      // disconnect, not just a muted output.
      if (stream) stream.setConcealerEnabled(msg.value);
      break;
    case 'setNoiseEnabled':
      // Master on/off for the whole "Noise" section - unlike
      // setConcealerEnabled above, this is routed through ConcealerStream's
      // own method rather than just poking streamConfig, because it
      // actually constructs/drops the active noise generator instance
      // (see stream.js's own comment) for a real disconnect - less RAM
      // and CPU while off, not just a muted output.
      if (stream) stream.setNoiseEnabled(msg.value);
      break;
    case 'setVizEnabled':
      // Master on/off for the "Visualization" panel - see specAnalyzers'
      // own comment above for why this actually constructs/drops the
      // three SpectrogramAnalyzer instances (a real disconnect, less RAM
      // and CPU while off) rather than just hiding the canvases. Only
      // meaningful while debugTelemetry is true in the first place - a
      // plain index.html session never sends this.
      vizEnabled = debugTelemetry && Boolean(msg.value);
      specAnalyzers = vizEnabled ? createSpecAnalyzers() : null;
      break;
    case 'setListenOriginal':
      if (stream) stream.streamConfig.listen_original = msg.value;
      break;
    case 'setListenConcealer':
      if (stream) stream.streamConfig.listen_concealer = msg.value;
      break;
    case 'setListenNoise':
      if (stream) stream.streamConfig.listen_noise = msg.value;
      break;
    case 'setConcealingThreshold':
      if (sourceVad) sourceVad.logitThreshold = msg.value;
      break;
    case 'setConcealerMemoryThreshold':
      if (concealer) concealer.vad.logitThreshold = msg.value;
      break;
    case 'setFadeDuration':
      // Only affects clips split off into memory AFTER this point (see
      // _feedMemory's own fadeSize computation, read fresh from
      // this.fadeDuration every cycle) - clips already stored keep whatever
      // fade was baked into them when they were created, same as how
      // concealerMemoryThreshold only affects future candidate filtering.
      if (concealer) concealer.fadeDuration = msg.value;
      break;
    case 'setConcealingOverlap':
      // Read live inside getConcealer() every time a clip is chosen (see
      // its own concealingCountdown computation) - only affects the NEXT
      // countdown roll, same "future only" caveat as the other live setters.
      if (concealer) {
        concealer.concealingMinTimeoutRatio = msg.minRatio;
        concealer.concealingMaxTimeoutRatio = msg.maxRatio;
      }
      break;
    case 'setConcealingNoise':
      // Dead code - the per-clip blur/noiseShape feature this drove
      // (concealer-noise.js) was a misreading of what "concealing noise"
      // was meant to be. No longer sent by main.js; left in place rather
      // than removed (see concealerConfig's own comment in init() above).
      if (concealer) concealer.concealingNoise = msg.value;
      break;
    case 'setConcealingNoiseLevel':
      // The REAL "Concealing noise" slider, now driving the
      // ConcealerStream-level background bed (see speech-shaped-noise.js) -
      // applied live, every chunk, in stream.js's processChunk, so this
      // takes effect immediately (unlike most of the other live setters
      // here, which only affect future clips/cycles).
      if (stream) stream.streamConfig.concealing_noise_level = msg.value;
      break;
    case 'setConcealingNoiseSensitivity':
      // The "Sensitivity" slider - the EMA time constant (seconds, 0.2-10,
      // clamped inside SpeechShapedNoise.setSensitivitySeconds) that
      // controls how far back the noise's spectral envelope remembers, NOT
      // a buffer size (see speech-shaped-noise.js's header for why) - takes
      // effect immediately, with no glitch, since it only changes a scalar
      // smoothing factor rather than resizing/resetting anything. Only
      // the speech-shaped generator has this - unused by white/pink.
      // Routed through ConcealerStream.setNoiseSensitivity (not applied
      // directly here) since the generator instance may not exist right
      // now (Noise off, or a different color active) - that method
      // remembers the value regardless and applies it if/when relevant.
      if (stream) stream.setNoiseSensitivity(msg.value);
      break;
    case 'setNoiseShapedPurity':
      // "Speech-shaped purity" slider - read fresh every chunk from
      // streamConfig inside stream.js's processChunk (see its own
      // comment), so a direct field update here is enough; no secondary
      // object needs to be told (unlike setConcealingNoiseSensitivity,
      // which also has to push into the live generator instance).
      if (stream) stream.streamConfig.noise_shaped_purity = msg.value;
      break;
    case 'setConcealingNoiseType':
      // "Noise controls" panel's color dropdown (speechShaped/white/pink
      // - see noise-generators.js and speech-shaped-noise.js). While
      // Noise is enabled, constructs a fresh instance of the newly-chosen
      // color right away (see ConcealerStream.setNoiseType) - a clean
      // start, not stale/minutes-old state from whenever it was last
      // active. While Noise is disabled, just remembers the choice for
      // whenever it's turned back on.
      if (stream) stream.setNoiseType(msg.value);
      break;
    case 'reset':
      if (stream) stream.resetState();
      break;
    case 'denoiseResult':
    case 'denoiseError':
      if (remoteDenoiser) remoteDenoiser.handleMessage(msg);
      break;
    case 'denoiserInitError':
      if (remoteDenoiser) remoteDenoiser.rejectAll(msg.message);
      break;
    case 'debugPlayMemoryClip': {
      // Debug-visualization only (see docs/debug.html) - lets clicking a
      // memory-queue dot hear what's actually stored there. Only sent by
      // debug.html's click handler, never by index.html, but harmless
      // either way: a stale/out-of-range index (e.g. the entry got evicted
      // between the click and this message arriving) just finds nothing and
      // silently does nothing.
      const entry = concealer && concealer.memory[msg.index];
      if (entry) {
        const clipCopy = Float32Array.from(entry.clip); // defensive copy - postMessage's transfer list would otherwise detach the actual stored clip's buffer
        postMessage({ type: 'debugMemoryClip', clip: clipCopy, sr }, [clipCopy.buffer]);
      }
      break;
    }
    case 'stop':
      if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
      break;
    default:
      break;
  }
};
