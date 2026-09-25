// Dedicated Worker (not the audio thread, not the main/UI thread) running
// the actual amods.Stream port: VAD, denoiser, GranSpeechMask, limiter.
// Talks to the main thread via postMessage; the main thread relays raw mic
// chunks in (from the AudioWorkletNode's port) and processed audio back out.
import * as ort from '../vendor/ort.min.mjs';

globalThis.ort = ort;
// Single-threaded WASM: works on plain GitHub Pages, no cross-origin-
// isolation (COOP/COEP) headers required - at some performance cost vs.
// multi-threaded WASM, which needs SharedArrayBuffer and those headers.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = '../vendor/';

import { SileroVAD } from './vad-silero.js';
import { DnsDenoiser } from './denoiser-dns64.js';
import { GranSpeechMask } from './granspeechmask.js';
import { ConcealerStream } from './stream.js';

let stream = null;
let concealer = null;
let sourceVad = null;

async function init(cfg) {
  const [vadSession, dnsSession] = await Promise.all([
    ort.InferenceSession.create('../models/silero_vad.onnx'),
    cfg.denoise ? ort.InferenceSession.create('../models/dns64.onnx') : Promise.resolve(null),
  ]);

  sourceVad = new SileroVAD(vadSession, { logitThreshold: cfg.concealingThreshold, sr: cfg.sr });
  const concealerVad = new SileroVAD(vadSession, { logitThreshold: cfg.concealerMemoryThreshold, sr: cfg.sr });
  const denoiser = dnsSession ? new DnsDenoiser(dnsSession, { sr: cfg.sr }) : null;

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
    denoise: cfg.denoise,
  };
  concealer = new GranSpeechMask(cfg.sr, concealerConfig, { denoiser, vad: concealerVad });

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
    case 'stop':
      if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
      break;
    default:
      break;
  }
};
