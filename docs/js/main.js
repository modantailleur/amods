// Main-thread glue: mic capture, AudioWorkletNode <-> Worker relay, and UI
// wiring. Mirrors amods.gui's Start/Stop + live-adjustable sliders.
const CHUNK_SIZE_AT_48K = 2400; // 50ms, matches stream_config.buffer_duration in the Python default config

const els = {
  startStopBtn: document.getElementById('start-stop-btn'),
  micSelect: document.getElementById('mic-select'),
  speakerSelect: document.getElementById('speaker-select'),
  concealingRate: document.getElementById('concealing-rate'),
  concealingRateValue: document.getElementById('concealing-rate-value'),
  concealerMemoryRate: document.getElementById('concealer-memory-rate'),
  concealerMemoryRateValue: document.getElementById('concealer-memory-rate-value'),
  concLevel: document.getElementById('conc-level'),
  concLevelValue: document.getElementById('conc-level-value'),
  denoiseCheckbox: document.getElementById('denoise-checkbox'),
  status: document.getElementById('status'),
  progress: document.getElementById('progress'),
  latency: document.getElementById('latency'),
  outputSinkWarning: document.getElementById('output-sink-warning'),
};

let audioContext = null;
let micStream = null;
let sourceNode = null;
let workletNode = null;
let worker = null;
let running = false;

function rateToThreshold(rateStr) {
  return Math.round((1.0 - parseFloat(rateStr)) * 10) / 10;
}

function dbToMultiplier(dbStr) {
  return Math.pow(10, parseFloat(dbStr) / 20);
}

async function populateDevices() {
  // Labels are empty until a permission has been granted at least once.
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
}

async function start() {
  els.startStopBtn.disabled = true;
  els.status.textContent = 'Starting…';

  audioContext = new AudioContext();
  const sr = audioContext.sampleRate;

  const micDeviceId = els.micSelect.value || undefined;
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
  worker.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'play') {
      workletNode.port.postMessage({ type: 'play', playMix: msg.playMix }, [msg.playMix.buffer]);
    } else if (msg.type === 'status') {
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
    }
  };

  worker.postMessage({
    type: 'init',
    config: {
      sr,
      denoise: els.denoiseCheckbox.checked,
      concealingThreshold: rateToThreshold(els.concealingRate.value),
      concealerMemoryThreshold: rateToThreshold(els.concealerMemoryRate.value),
      concMultiplier: dbToMultiplier(els.concLevel.value),
      monitorGain: 0.0,
    },
  });

  sourceNode = audioContext.createMediaStreamSource(micStream);
  sourceNode.connect(workletNode);
  workletNode.connect(audioContext.destination);

  if (els.speakerSelect.value && typeof audioContext.setSinkId === 'function') {
    try {
      await audioContext.setSinkId(els.speakerSelect.value);
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
  setControlsEnabled(true);
}

function setControlsEnabled(enabled) {
  els.micSelect.disabled = !enabled;
  els.speakerSelect.disabled = !enabled;
  els.denoiseCheckbox.disabled = !enabled;
  // Concealing rate, concealer memory rate, and concealer level stay
  // enabled while running - they're live-adjustable (see the worker
  // message handlers below), same as the Python GUI.
}

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
  els.latency.textContent = `algo ${timing.avgMs.toFixed(1)}/${timing.maxMs.toFixed(1)} ms avg/max`;
}

els.startStopBtn.addEventListener('click', () => {
  if (running) stop();
  else start().catch((e) => {
    els.status.textContent = `Failed to start: ${e.message}`;
    els.startStopBtn.disabled = false;
  });
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
