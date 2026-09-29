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
import * as ort from '../vendor/ort.min.mjs';

globalThis.ort = ort;
// Single-threaded WASM: works on plain GitHub Pages, no cross-origin-
// isolation (COOP/COEP) headers required - at some performance cost vs.
// multi-threaded WASM, which needs SharedArrayBuffer and those headers.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = '../vendor/';

import { SileroVAD } from './vad-silero.js';
import { GranSpeechMask } from './granspeechmask.js';
import { ConcealerStream } from './stream.js';
import { RemoteDenoiser } from './denoiser-proxy.js';
import { applyFadeIn } from './audio-utils.js';
import { installDebugLogRelay } from './debug-log.js';

let stream = null;
let concealer = null;
let sourceVad = null;
let remoteDenoiser = null;
let sr = 48000;
let debugTelemetry = false; // set from cfg.debugTelemetry - see docs/debug.html; gates the extra debugSnapshot() work in startStatusLoop below so plain index.html never pays for it

async function init(cfg) {
  sr = cfg.sr;
  debugTelemetry = Boolean(cfg.debugTelemetry);
  // This Worker has its own separate console from the main thread's, so
  // main.js's own installDebugLogRelay('main') call never sees anything
  // logged in here - see debug-log.js for why both realms need their own.
  if (debugTelemetry) installDebugLogRelay('worker');
  // Two SEPARATE sessions (same model file, loaded twice), NOT one shared
  // between sourceVad and concealerVad - the two are called from genuinely
  // independent, interleavable call chains (sourceVad every ~50ms from the
  // real-time chunk loop; concealerVad in a burst from _feedMemory, which
  // runs fire-and-forget alongside that same loop - see updateMemory), and
  // sharing one InferenceSession's internal execution buffers/state across
  // two logically-unrelated streams of calls is not a safe pattern for a
  // stateful model. Confirmed the hard way: sourceVad detected speech
  // correctly right up until the very first time concealerVad ever ran, then
  // never detected anything again for the rest of the session, however long
  // silence followed - a single shared session getting corrupted the first
  // time both were in play at once explains that exactly.
  const sourceVadSession = await ort.InferenceSession.create('../models/silero_vad.onnx');
  const concealerVadSession = await ort.InferenceSession.create('../models/silero_vad.onnx');

  sourceVad = new SileroVAD(sourceVadSession, { logitThreshold: cfg.concealingThreshold, sr: cfg.sr, name: 'source', debug: debugTelemetry });
  const concealerVad = new SileroVAD(concealerVadSession, { logitThreshold: cfg.concealerMemoryThreshold, sr: cfg.sr, name: 'concealer', debug: debugTelemetry });
  remoteDenoiser = cfg.denoiserEnabled ? new RemoteDenoiser() : null;

  const concealerConfig = {
    memory_maxlen: 100,
    min_memory_to_conceal: 20,
    max_countdown_reuse: 10,
    max_concealer_distance_to_buffer: 5,
    concealer_duration: 0.3,
    concealing_min_timeout_ratio: 0.6,
    concealing_max_timeout_ratio: 1.0,
    fade_duration: 0.05,
    is_stream: true,
    pending_conc_max_size: 3 * cfg.sr,
    freeze_learning: false,
    decision_win: 0.3,
    denoise: cfg.denoiserEnabled,
  };
  concealer = new GranSpeechMask(cfg.sr, concealerConfig, { denoiser: remoteDenoiser, vad: concealerVad });

  const streamConfig = {
    sr: cfg.sr,
    channels_out: 1,
    monitor_gain: cfg.monitorGain ?? 0.0,
    record_mic_gain: 1.0,
    conc_multiplier: cfg.concMultiplier ?? 1.0,
    // Debug-only listen toggles (see docs/debug.html) - index.html has no UI
    // for these, so they just stay at these defaults (concealer-only, same
    // as production always sounded before this existed) for every real
    // session.
    listen_original: cfg.listenOriginal ?? false,
    listen_concealer: cfg.listenConcealer ?? true,
  };
  stream = new ConcealerStream(streamConfig, sourceVad, concealer, debugTelemetry);

  postMessage({ type: 'ready' });
  startStatusLoop();
}

let statusTimer = null;
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
    const debug = debugTelemetry
      ? { voiceActive: stream.popVoiceActiveSincePoll(), ...concealer.debugSnapshot() }
      : undefined;
    postMessage({ type: 'status', status, timing, debug });
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
    let { playMix } = await stream.processChunk(item.chunk);
    const staleMs = performance.now() - item.arrivedAt;
    if (staleMs <= MAX_STALE_MS) {
      if (hadSkip) {
        playMix = applyFadeIn(playMix, Math.round(FADE_AFTER_SKIP_S * sr));
        hadSkip = false;
      }
      postMessage({ type: 'play', playMix }, [playMix.buffer]);
    } else {
      hadSkip = true; // too stale to play, but pendingVoice/concealer state above already saw it in full - memory continuity is never affected by this
    }
  } catch (e) {
    console.error('worker-engine processChunk failed:', e);
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
        console.error('worker-engine init failed:', e);
        postMessage({ type: 'error', message: String(e) });
      });
      break;
    case 'chunk':
      handleChunk(msg.chunk);
      break;
    case 'setConcMultiplier':
      if (stream) stream.streamConfig.conc_multiplier = msg.value;
      break;
    case 'setListenOriginal':
      if (stream) stream.streamConfig.listen_original = msg.value;
      break;
    case 'setListenConcealer':
      if (stream) stream.streamConfig.listen_concealer = msg.value;
      break;
    case 'setConcealingThreshold':
      if (sourceVad) sourceVad.logitThreshold = msg.value;
      break;
    case 'setConcealerMemoryThreshold':
      if (concealer) concealer.vad.logitThreshold = msg.value;
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
