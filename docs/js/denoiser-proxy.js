// A drop-in stand-in for DnsDenoiser's {predict(y)} interface (see
// denoiser-dns64.js), except the actual ONNX inference happens in a
// completely separate Worker (docs/js/denoiser-worker.js) instead of here.
//
// Why this exists: GranSpeechMask's _feedMemory calls the denoiser roughly
// every 2 seconds, in what's meant to be a background/parallel branch that
// never affects real-time playback (matching amods.gui's own design, where
// that call runs on a background Python thread and native onnxruntime
// releases the GIL so the real-time audio callback thread keeps running).
// A Worker has only one thread, though - wrapping a call in an unawaited
// promise ("fire and forget") does NOT make it run in parallel, it just
// defers when the *caller* proceeds; the actual WASM computation still
// monopolizes that one thread for its whole duration (measured: ~800ms for
// the quantized model), during which incoming mic chunks queue up
// undelivered and arrive that late once it's done. The only real fix is a
// second Worker (a second OS thread) - this class is what lets
// worker-engine.js hand off to it without GranSpeechMask itself knowing or
// caring that its "denoiser" isn't local anymore.
//
// worker-engine.js is responsible for relaying 'denoiseResult'/'denoiseError'
// messages (received from the main thread, itself relaying from
// denoiser-worker.js - see main.js) into handleMessage() below.
export class RemoteDenoiser {
  constructor() {
    this._nextId = 1;
    this._pending = new Map();
  }

  handleMessage(msg) {
    const pending = this._pending.get(msg.id);
    if (!pending) return;
    this._pending.delete(msg.id);
    if (msg.type === 'denoiseResult') pending.resolve(msg.result);
    else pending.reject(new Error(msg.message || 'denoise failed'));
  }

  /** Reject every in-flight request (e.g. the denoiser worker failed to load its model) rather than leaving them hanging forever. */
  rejectAll(message) {
    for (const pending of this._pending.values()) pending.reject(new Error(message));
    this._pending.clear();
  }

  /**
   * y: Float32Array. Returns a Promise<Float32Array>, resolved once
   * denoiser-worker.js replies. Matches DnsDenoiser.predict's contract.
   */
  predict(y) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      // A defensive copy: y (GranSpeechMask's `snapshot`) is still read
      // from after this call returns (_feedMemory slices clips out of it),
      // so it must not be the same buffer handed to postMessage's transfer
      // list - transferring detaches the original.
      const copy = Float32Array.from(y);
      postMessage({ type: 'denoiseRequest', id, y: copy }, [copy.buffer]);
    });
  }
}
