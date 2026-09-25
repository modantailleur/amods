// Main-thread glue: mic capture, AudioWorkletNode <-> Worker relay, idle
// mic-level preview, Ping, and UI wiring. Mirrors amods.gui as closely as
// the Web Audio API allows - see per-feature notes below for where it
// can't be identical (input-latency reporting, output-device latency
// tuning granularity, etc).
import { applyFade } from './audio-utils.js';

const CHUNK_SIZE_AT_48K = 2400; // 50ms, matches stream_config.buffer_duration in the Python default config

// "Ping" test tone - same constants as amods.gui (PING_FREQUENCY_HZ etc.).
const PING_FREQUENCY_HZ = 440.0;
const PING_DURATION_S = 0.7;
const PING_FADE_S = 0.05;
const PING_AMPLITUDE = 0.4;

// Matches amods.gui's LEVEL_METER_DB_RANGE exactly.
const LEVEL_METER_DB_MIN = -55.0;
const LEVEL_METER_DB_MAX = -5.0;

const els = {
  startStopBtn: document.getElementById('start-stop-btn'),
  micSelect: document.getElementById('mic-select'),
  speakerSelect: document.getElementById('speaker-select'),
  inLevelFill: document.getElementById('in-level-fill'),
  outLevelFill: document.getElementById('out-level-fill'),
  pingBtn: document.getElementById('ping-btn'),
  outputLatency: document.getElementById('output-latency'),
  outputLatencyValue: document.getElementById('output-latency-value'),
  feedbackWarning: document.getElementById('feedback-warning'),
  outputSinkWarning: document.getElementById('output-sink-warning'),
  concealingRate: document.getElementById('concealing-rate'),
  concealingRateValue: document.getElementById('concealing-rate-value'),
  concealerMemoryRate: document.getElementById('concealer-memory-rate'),
  concealerMemoryRateValue: document.getElementById('concealer-memory-rate-value'),
  concLevel: document.getElementById('conc-level'),
  concLevelValue: document.getElementById('conc-level-value'),
  denoiserSelect: document.getElementById('denoiser-select'),
  status: document.getElementById('status'),
  progress: document.getElementById('progress'),
  latency: document.getElementById('latency'),
};

let audioContext = null; // the real session's AudioContext (Start/Stop)
let micStream = null;
let sourceNode = null;
let workletNode = null;
let worker = null;
let running = false;

// Idle mic-level preview + Ping share this lightweight context, separate
// from the real session's (which needs a specific latencyHint - see
// start()) - mirrors amods.gui's separate "idle input monitor" vs. the
// real Stream.
let previewContext = null;
let previewSource = null;
let previewAnalyser = null;
let previewRafId = null;
let pingActive = false;

function rateToThreshold(rateStr) {
  return Math.round((1.0 - parseFloat(rateStr)) * 10) / 10;
}

function dbToMultiplier(dbStr) {
  return Math.pow(10, parseFloat(dbStr) / 20);
}

function levelFromRms(rms) {
  const db = 20 * Math.log10(rms + 1e-9);
  const level = ((db - LEVEL_METER_DB_MIN) / (LEVEL_METER_DB_MAX - LEVEL_METER_DB_MIN)) * 100;
  return Math.max(0, Math.min(100, level));
}

function selectedMicId() {
  return els.micSelect.value || null;
}

function selectedSpeakerId() {
  return els.speakerSelect.value || null;
}

// ── Same-device warning (mirrors amods.gui._update_warnings, minus the
// "built-in device name" heuristic - browser device labels don't reliably
// expose that the way ALSA hw: names do) ─────────────────────────────────
function updateWarnings() {
  const micId = selectedMicId();
  const spkId = selectedSpeakerId();
  if (micId && spkId && micId === spkId) {
    els.feedbackWarning.textContent =
      'Warning: same device selected for mic and speaker - this can cause audio feedback (howling) once concealing starts, or fail to open for simultaneous input+output on some systems.';
  } else {
    els.feedbackWarning.textContent = '';
  }
}

// ── Idle mic-level preview ────────────────────────────────────────────────
async function ensurePreviewContext() {
  if (!previewContext) previewContext = new AudioContext();
  if (previewContext.state === 'suspended') {
    try { await previewContext.resume(); } catch { /* needs a user gesture in some browsers - best effort */ }
  }
  return previewContext;
}

async function startInputPreview() {
  if (previewSource || pingActive) return;
  const micId = selectedMicId();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: micId ? { exact: micId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
    const ctx = await ensurePreviewContext();
    previewSource = ctx.createMediaStreamSource(stream);
    previewSource._rawStream = stream;
    previewAnalyser = ctx.createAnalyser();
    previewAnalyser.fftSize = 1024;
    previewSource.connect(previewAnalyser);
    pollPreviewLevel();
  } catch {
    // Best-effort preview only, same as amods.gui's own idle monitor - if
    // the device can't be opened here (e.g. busy elsewhere), just leave
    // the bar flat.
    els.inLevelFill.style.width = '0%';
  }
}

function pollPreviewLevel() {
  if (!previewAnalyser) return;
  const buf = new Float32Array(previewAnalyser.fftSize);
  previewAnalyser.getFloatTimeDomainData(buf);
  let sumSq = 0;
  for (let i = 0; i < buf.length; i++) sumSq += buf[i] * buf[i];
  els.inLevelFill.style.width = `${levelFromRms(Math.sqrt(sumSq / buf.length))}%`;
  previewRafId = requestAnimationFrame(pollPreviewLevel);
}

function stopInputPreview() {
  if (previewRafId) cancelAnimationFrame(previewRafId);
  previewRafId = null;
  if (previewSource) {
    previewSource.disconnect();
    previewSource._rawStream.getTracks().forEach((t) => t.stop());
    previewSource = null;
  }
  previewAnalyser = null;
  els.inLevelFill.style.width = '0%';
}

function restartInputPreview() {
  stopInputPreview();
  startInputPreview();
}

// ── Ping ──────────────────────────────────────────────────────────────────
async function onPing() {
  if (pingActive) return;
  const deviceOut = selectedSpeakerId();
  if (els.speakerSelect.options.length === 0) {
    els.status.textContent = 'Select a speaker first.';
    return;
  }
  pingActive = true;
  els.pingBtn.disabled = true;
  // Release the idle mic monitor for the duration of the test tone, same
  // simultaneous-input+output rationale as amods.gui.
  stopInputPreview();

  try {
    const ctx = await ensurePreviewContext();
    if (deviceOut && typeof ctx.setSinkId === 'function') {
      try {
        await ctx.setSinkId(deviceOut);
      } catch (e) {
        els.outputSinkWarning.textContent = `Could not switch output device for Ping: ${e.message}`;
      }
    }

    const sr = ctx.sampleRate;
    const n = Math.floor(sr * PING_DURATION_S);
    const tone = new Float32Array(n);
    for (let i = 0; i < n; i++) tone[i] = PING_AMPLITUDE * Math.sin((2 * Math.PI * PING_FREQUENCY_HZ * i) / sr);
    const fadeSize = Math.max(1, Math.floor(sr * PING_FADE_S));
    const faded = applyFade(tone, fadeSize); // same raised-cosine fade as every concealer clip

    const buffer = ctx.createBuffer(1, n, sr);
    buffer.copyToChannel(faded, 0);

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyser);
    analyser.connect(ctx.destination);

    const levelBuf = new Float32Array(analyser.fftSize);
    let animating = true;
    (function animate() {
      if (!animating) return;
      analyser.getFloatTimeDomainData(levelBuf);
      let sumSq = 0;
      for (let i = 0; i < levelBuf.length; i++) sumSq += levelBuf[i] * levelBuf[i];
      els.outLevelFill.style.width = `${levelFromRms(Math.sqrt(sumSq / levelBuf.length))}%`;
      requestAnimationFrame(animate);
    })();

    await new Promise((resolve) => {
      source.onended = resolve;
      source.start();
    });
    animating = false;
  } catch (e) {
    els.status.textContent = `Could not play test tone: ${e.message}`;
  } finally {
    els.outLevelFill.style.width = '0%';
    pingActive = false;
    els.pingBtn.disabled = false;
    startInputPreview();
  }
}

// ── Devices ───────────────────────────────────────────────────────────────
async function populateDevices() {
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch (e) {
    els.status.textContent = `Microphone permission is required: ${e.message}`;
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  els.micSelect.innerHTML = '';
  els.speakerSelect.innerHTML = '';
  for (const d of devices) {
    if (d.kind === 'audioinput') {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Microphone ${els.micSelect.length + 1}`;
      els.micSelect.appendChild(opt);
    } else if (d.kind === 'audiooutput') {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Speaker ${els.speakerSelect.length + 1}`;
      els.speakerSelect.appendChild(opt);
    }
  }
  if (els.speakerSelect.length === 0) {
    els.outputSinkWarning.textContent =
      'This browser did not report any output devices separately (audiooutput enumeration/setSinkId support varies by browser) - audio will play on the system default speaker.';
  }
  updateWarnings();
  startInputPreview();
}

// ── Start / Stop ────────────────────────────────────────────────────────
async function start() {
  els.startStopBtn.disabled = true;
  els.status.textContent = 'Starting…';

  stopInputPreview();
  if (pingActive) return; // Ping's own finally{} will restore state; don't fight it

  // Output latency, requested via AudioContext's latencyHint (the closest
  // Web Audio equivalent of PortAudio's per-stream `latency=` request) -
  // only takes effect at construction time, so (like amods.gui) it's a
  // Start-time snapshot, not live-adjustable.
  const latencyHintS = parseFloat(els.outputLatency.value) / 1000;
  audioContext = new AudioContext({ latencyHint: latencyHintS });
  const sr = audioContext.sampleRate;

  const micDeviceId = selectedMicId();
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: micDeviceId ? { exact: micDeviceId } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });

  await audioContext.audioWorklet.addModule('./js/worklet-processor.js');
  workletNode = new AudioWorkletNode(audioContext, 'concealer-worklet-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { chunkSize: CHUNK_SIZE_AT_48K, outputRingSize: sr * 2 },
  });

  worker = new Worker('./js/worker-engine.js', { type: 'module' });
  let latestTiming = { avgMs: 0, maxMs: 0 };
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'play') {
      workletNode.port.postMessage({ type: 'play', playMix: msg.playMix }, [msg.playMix.buffer]);
    } else if (msg.type === 'status') {
      latestTiming = msg.timing;
      updateStatus(msg.status, msg.timing);
    } else if (msg.type === 'error') {
      els.status.textContent = `Engine error: ${msg.message}`;
    } else if (msg.type === 'ready') {
      els.status.textContent = 'Running';
    }
  };
  workletNode.port.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'chunk') {
      worker.postMessage({ type: 'chunk', chunk: msg.chunk }, [msg.chunk.buffer]);
    } else if (msg.type === 'levels') {
      els.inLevelFill.style.width = `${msg.in}%`;
      els.outLevelFill.style.width = `${msg.out}%`;
    }
  };

  worker.postMessage({
    type: 'init',
    config: {
      sr,
      denoiserModel: els.denoiserSelect.value,
      concealingThreshold: rateToThreshold(els.concealingRate.value),
      concealerMemoryThreshold: rateToThreshold(els.concealerMemoryRate.value),
      concMultiplier: dbToMultiplier(els.concLevel.value),
      monitorGain: 0.0,
    },
  });

  sourceNode = audioContext.createMediaStreamSource(micStream);
  sourceNode.connect(workletNode);
  workletNode.connect(audioContext.destination);

  const deviceOut = selectedSpeakerId();
  if (deviceOut && typeof audioContext.setSinkId === 'function') {
    try {
      await audioContext.setSinkId(deviceOut);
    } catch (e) {
      els.outputSinkWarning.textContent = `Could not switch output device: ${e.message}`;
    }
  }

  running = true;
  els.startStopBtn.textContent = 'Stop';
  els.startStopBtn.disabled = false;
  setControlsEnabled(false);
}

function stop() {
  els.startStopBtn.disabled = true;
  if (worker) {
    worker.postMessage({ type: 'stop' });
    worker.terminate();
    worker = null;
  }
  if (workletNode) {
    workletNode.port.postMessage({ type: 'stop' });
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  running = false;
  els.startStopBtn.textContent = 'Start';
  els.startStopBtn.disabled = false;
  els.status.textContent = 'Idle — press Start to begin';
  els.progress.textContent = '—';
  els.latency.textContent = '—';
  els.inLevelFill.style.width = '0%';
  els.outLevelFill.style.width = '0%';
  setControlsEnabled(true);
  startInputPreview();
}

function setControlsEnabled(enabled) {
  els.micSelect.disabled = !enabled;
  els.speakerSelect.disabled = !enabled;
  els.denoiserSelect.disabled = !enabled;
  els.pingBtn.disabled = !enabled;
  // Output latency can only be applied when the AudioContext is created
  // (see start()), so - like amods.gui - it's locked while running.
  els.outputLatency.disabled = !enabled;
  // Concealing rate, concealer memory rate, and concealer level stay
  // enabled while running - they're live-adjustable (see the worker
  // message handlers below), same as the Python GUI.
}

// ── Latency display: in + out + algo, mirrors amods.gui._refresh_status ──
function updateStatus(status, timing) {
  if (status.target != null) {
    els.progress.textContent = `${status.progress} / ${status.target}`;
  } else {
    els.progress.textContent = String(status.progress);
  }
  if (!status.ready) {
    els.status.textContent = `Waiting for memory to warm up (${status.progress}/${status.threshold}) before concealing starts`;
  } else {
    els.status.textContent = 'Running';
  }

  // Output latency: AudioContext.outputLatency is a live, more accurate
  // estimate where supported; baseLatency is the spec-guaranteed minimum,
  // used as a fallback.
  const outLatencyS = audioContext ? (audioContext.outputLatency ?? audioContext.baseLatency ?? 0) : 0;
  const outMs = outLatencyS * 1000;

  // Input latency: the Web Audio API has no equivalent of PortAudio's
  // input_stream.latency; MediaStreamTrack.getSettings().latency exists in
  // some browsers for some devices, but isn't guaranteed - shown as "n/a"
  // rather than a misleading 0 when unavailable.
  const track = micStream ? micStream.getAudioTracks()[0] : null;
  const settings = track && track.getSettings ? track.getSettings() : {};
  const inMsKnown = typeof settings.latency === 'number';
  const inMs = inMsKnown ? settings.latency * 1000 : 0;

  const totalMs = (inMsKnown ? inMs : 0) + outMs + timing.avgMs;
  const inText = inMsKnown ? `${inMs.toFixed(0)} ms` : 'n/a';
  els.latency.textContent =
    `~${totalMs.toFixed(0)} ms  (in ${inText} + out ${outMs.toFixed(0)} ms + algo ` +
    `${timing.avgMs.toFixed(1)}/${timing.maxMs.toFixed(1)} ms avg/max)`;
}

// ── Event wiring ──────────────────────────────────────────────────────────
els.startStopBtn.addEventListener('click', () => {
  if (running) stop();
  else start().catch((e) => {
    els.status.textContent = `Failed to start: ${e.message}`;
    els.startStopBtn.disabled = false;
  });
});

els.pingBtn.addEventListener('click', () => onPing());

els.micSelect.addEventListener('change', () => {
  updateWarnings();
  restartInputPreview();
});
els.speakerSelect.addEventListener('change', () => updateWarnings());

els.outputLatency.addEventListener('input', () => {
  els.outputLatencyValue.textContent = `${els.outputLatency.value} ms`;
});

els.concealingRate.addEventListener('input', () => {
  const threshold = rateToThreshold(els.concealingRate.value);
  els.concealingRateValue.textContent = els.concealingRate.value;
  if (worker && running) worker.postMessage({ type: 'setConcealingThreshold', value: threshold });
});

els.concealerMemoryRate.addEventListener('input', () => {
  const threshold = rateToThreshold(els.concealerMemoryRate.value);
  els.concealerMemoryRateValue.textContent = els.concealerMemoryRate.value;
  if (worker && running) worker.postMessage({ type: 'setConcealerMemoryThreshold', value: threshold });
});

els.concLevel.addEventListener('input', () => {
  const mult = dbToMultiplier(els.concLevel.value);
  els.concLevelValue.textContent = `${parseFloat(els.concLevel.value) >= 0 ? '+' : ''}${els.concLevel.value} dB`;
  if (worker && running) worker.postMessage({ type: 'setConcMultiplier', value: mult });
});

populateDevices().catch((e) => {
  els.status.textContent = `Could not list audio devices: ${e.message}`;
});
