// Runs on the browser's real-time audio rendering thread. Deliberately does
// almost nothing: the AudioWorkletGlobalScope can't load WASM/ONNX
// Runtime reliably, and process() is called every ~128-sample render
// quantum (~2.9ms @44.1kHz) - far too fine-grained for VAD/denoiser
// inference anyway. All of that (the actual amods.Stream port) runs in a
// separate dedicated Worker; this processor's only job is buffering mic
// input into fixed-size chunks (matching stream_config.buffer_duration,
// same granularity Stream._process_chunk uses) and playing back whatever
// processed output the worker has produced so far, via its port - this
// mirrors Stream's own input_callback/output_callback + ring buffer split
// (see stream.py), just relayed through the main thread once instead of
// talking to the worker directly (an AudioWorkletProcessor's port can only
// reach its owning AudioWorkletNode on the main thread). While "Speaker
// mode" is on, mic input is instead buffered into smaller aecChunkSize
// pieces (see the constructor's own comment) so its EchoCanceller can
// start working on a given slice of audio sooner than waiting for a whole
// chunkSize block to accumulate - VAD/concealer selection/etc. still only
// ever see reassembled chunkSize blocks either way (see worker-engine.js's
// handleAecChunk), unaffected by this.
// Matches amods.gui's LEVEL_METER_DB_RANGE / LEVEL_METER_MIN_UPDATE_INTERVAL
// exactly: RMS-in-dBFS mapped onto this range gives a 0-100 bar value,
// chosen so normal speech (roughly -25 to -15 dBFS) sits in the upper half
// of the bar; updates throttled so the meter doesn't redraw far more often
// than it's actually useful to look at.
const LEVEL_METER_DB_MIN = -55.0;
const LEVEL_METER_DB_MAX = -5.0;
const LEVEL_METER_MIN_UPDATE_INTERVAL_S = 0.05;

function levelFromSumSq(sumSq, count) {
  if (count === 0) return 0;
  const rms = Math.sqrt(sumSq / count);
  const db = 20 * Math.log10(rms + 1e-9);
  const level = ((db - LEVEL_METER_DB_MIN) / (LEVEL_METER_DB_MAX - LEVEL_METER_DB_MIN)) * 100;
  return Math.max(0, Math.min(100, level));
}

class ConcealerWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { chunkSize = 2400, outputRingSize = 48000 * 2, aecChunkSize = 0 } = options.processorOptions || {};
    this.chunkSize = chunkSize;
    this.inputBuf = new Float32Array(chunkSize);
    this.inputPos = 0;

    // "Speaker mode" (see main.js's own AEC_CAPTURE_SUBDIVISIONS comment) -
    // a finer-grained REPLACEMENT for the input tap above, not an addition
    // to it (see process() below - it's one or the other, never both):
    // flushed every aecChunkSize samples instead of chunkSize, so
    // EchoCanceller.processCapture (see worker-engine.js's handleAecChunk)
    // can start working on a given slice of mic audio well before the full
    // chunkSize block it belongs to would otherwise have finished
    // accumulating - up to (chunkSize-aecChunkSize) samples' worth of
    // real latency saved on the capture side. 0 (the default, Speaker mode
    // off) disables this entirely - aecInputBuf stays null, the original
    // chunkSize-only behavior above runs unmodified, zero added cost.
    this.aecChunkSize = aecChunkSize;
    this.aecInputBuf = aecChunkSize > 0 ? new Float32Array(aecChunkSize) : null;
    this.aecInputPos = 0;

    // Output ring buffer: filled by messages from the main thread (relayed
    // from the worker), drained sample-by-sample in process().
    this.outRing = new Float32Array(outputRingSize);
    this.outWrite = 0;
    this.outRead = 0;
    this.outAvailable = 0;

    this._inSumSq = 0;
    this._inCount = 0;
    this._outSumSq = 0;
    this._outCount = 0;
    this._lastLevelReport = 0;

    this.running = true;

    this.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'play') {
        const block = msg.playMix; // Float32Array
        for (let i = 0; i < block.length; i++) {
          this.outRing[this.outWrite] = block[i];
          this.outWrite = (this.outWrite + 1) % this.outRing.length;
        }
        this.outAvailable = Math.min(this.outAvailable + block.length, this.outRing.length);
      } else if (msg.type === 'stop') {
        this.running = false;
      }
    };
  }

  process(inputs, outputs) {
    if (!this.running) return false;

    const input = inputs[0];
    const output = outputs[0];
    const inCh = input && input[0] ? input[0] : null;
    const outCh = output[0];

    if (inCh) {
      for (let i = 0; i < inCh.length; i++) {
        this._inSumSq += inCh[i] * inCh[i];
        this._inCount++;
        // Either the fine-grained Speaker-mode tap or the normal one runs -
        // see aecInputBuf's own comment for why these are alternatives, not
        // both active together.
        if (this.aecInputBuf) {
          this.aecInputBuf[this.aecInputPos++] = inCh[i];
          if (this.aecInputPos === this.aecChunkSize) {
            const aecChunk = this.aecInputBuf.slice(0);
            this.port.postMessage({ type: 'aecChunk', chunk: aecChunk }, [aecChunk.buffer]);
            this.aecInputPos = 0;
          }
        } else {
          this.inputBuf[this.inputPos++] = inCh[i];
          if (this.inputPos === this.chunkSize) {
            // Transfer ownership of a fresh copy (postMessage with a
            // Transferable ArrayBuffer - zero-copy) so the worklet's own
            // buffer can keep being written to immediately.
            const chunk = this.inputBuf.slice(0);
            this.port.postMessage({ type: 'chunk', chunk }, [chunk.buffer]);
            this.inputPos = 0;
          }
        }
      }
    }

    for (let i = 0; i < outCh.length; i++) {
      if (this.outAvailable > 0) {
        outCh[i] = this.outRing[this.outRead];
        this.outRead = (this.outRead + 1) % this.outRing.length;
        this.outAvailable--;
      } else {
        outCh[i] = 0; // underrun: processing hasn't kept up yet - silence rather than stale/garbage audio
      }
      this._outSumSq += outCh[i] * outCh[i];
      this._outCount++;
    }

    if (currentTime - this._lastLevelReport >= LEVEL_METER_MIN_UPDATE_INTERVAL_S) {
      this._lastLevelReport = currentTime;
      this.port.postMessage({
        type: 'levels',
        in: levelFromSumSq(this._inSumSq, this._inCount),
        out: levelFromSumSq(this._outSumSq, this._outCount),
      });
      this._inSumSq = 0;
      this._inCount = 0;
      this._outSumSq = 0;
      this._outCount = 0;
    }

    return true;
  }
}

registerProcessor('concealer-worklet-processor', ConcealerWorkletProcessor);
