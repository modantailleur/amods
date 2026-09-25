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

let stream = null;
let concealer = null;
let sourceVad = null;
let remoteDenoiser = null;

async function init(cfg) {
  const vadSession = await ort.InferenceSession.create('../models/silero_vad.onnx');

  sourceVad = new SileroVAD(vadSession, { logitThreshold: cfg.concealingThreshold, sr: cfg.sr });
  const concealerVad = new SileroVAD(vadSession, { logitThreshold: cfg.concealerMemoryThreshold, sr: cfg.sr });
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
  };
  stream = new ConcealerStream(streamConfig, sourceVad, concealer);

  postMessage({ type: 'ready' });
  startStatusLoop();
}

let statusTimer = null;
function startStatusLoop() {
  if (statusTimer) return;
  statusTimer = setInterval(() => {
    if (!concealer || !stream) return;
    const status = concealer.status();
    const timing = stream.popCallbackTiming();
    postMessage({ type: 'status', status, timing });
  }, 500);
}

let processing = false;
async function handleChunk(chunk) {
  if (!stream || processing) return; // drop this chunk rather than queue up and fall behind
  processing = true;
  try {
    const { playMix } = await stream.processChunk(chunk);
    postMessage({ type: 'play', playMix }, [playMix.buffer]);
  } catch (e) {
    console.error('worker-engine processChunk failed:', e);
  } finally {
    processing = false;
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
    case 'stop':
      if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
      break;
    default:
      break;
  }
};
