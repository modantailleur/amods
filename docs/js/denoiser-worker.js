// Dedicated Worker whose only job is running dns64 inference - split out
// from worker-engine.js so a ~800ms-1s denoiser call can never block the
// real-time per-chunk pipeline (see denoiser-proxy.js's header comment for
// why that split is required, not optional, in a single-threaded Worker).
// Talks only to the main thread (main.js), which relays 'denoiseRequest'
// from worker-engine.js in and 'denoiseResult'/'denoiseError' back out -
// see main.js's onPing-adjacent worker wiring in start().
import * as ort from '../vendor/ort.min.mjs';

globalThis.ort = ort;
ort.env.wasm.numThreads = 1;

import { DnsDenoiser } from './denoiser-dns64.js';

let denoiser = null;
let modelLoadError = null;
// Requests that arrive before the model finishes loading (plausible for the
// original ~134MB model on a slow connection, since this worker's 'init'
// and worker-engine's own init race independently) are queued rather than
// rejected outright.
const queue = [];

async function runDenoise(msg) {
  try {
    const result = await denoiser.predict(msg.y);
    // Defensive copy before transferring: result may be a view backed
    // directly by onnxruntime-web's WASM heap rather than a plain, safely
    // transferable/detachable ArrayBuffer.
    const safeResult = Float32Array.from(result);
    postMessage({ type: 'denoiseResult', id: msg.id, result: safeResult }, [safeResult.buffer]);
  } catch (e) {
    postMessage({ type: 'denoiseError', id: msg.id, message: String(e) });
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === 'init') {
    try {
      const session = await ort.InferenceSession.create(msg.path);
      denoiser = new DnsDenoiser(session, { sr: msg.sr });
      postMessage({ type: 'ready' });
      const pending = queue.splice(0);
      for (const m of pending) runDenoise(m);
    } catch (e) {
      modelLoadError = String(e);
      postMessage({ type: 'error', message: modelLoadError });
      const pending = queue.splice(0);
      for (const m of pending) postMessage({ type: 'denoiseError', id: m.id, message: modelLoadError });
    }
  } else if (msg.type === 'denoiseRequest') {
    if (modelLoadError) {
      postMessage({ type: 'denoiseError', id: msg.id, message: modelLoadError });
    } else if (!denoiser) {
      queue.push(msg);
    } else {
      runDenoise(msg);
    }
  }
};
