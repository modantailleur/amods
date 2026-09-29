// Main-thread glue: mic capture, AudioWorkletNode <-> Worker relay, idle
// mic-level preview, Ping, and UI wiring. Mirrors amods.gui as closely as
// the Web Audio API allows - see per-feature notes below for where it
// can't be identical (input-latency reporting, etc). Deliberately has no
// output-latency control: AudioContext's latencyHint is only a hint browsers
// are free to ignore or coarsely quantize, and even where honored it only
// affects the hardware output buffer (tens of ms) - nowhere near this app's
// actual latency sources (chunk size, VAD windows, concealer clip duration),
// so it wouldn't give users the PortAudio-style control amods.gui's
// equivalent slider does.
import { applyFade } from './audio-utils.js';
import { DENOISER_MODEL_PATHS } from './denoiser-models.js';
import { installDebugLogRelay } from './debug-log.js';

// Set by debug.html (never by index.html) before this module loads. Gates
// every debug-only feature below so the production page never wires any of
// it up - on index.html, #debug-start-stop-btn etc. simply don't exist in the DOM.
const DEBUG_MODE = window.DEBUG_MODE === true;
const DEBUG_AUDIO_PATH = './audios/ah.wav';

// Streams every console.log/warn/error from here on to a local terminal tool
// (run `node scripts/debug-log-server.mjs`) instead of requiring DevTools +
// copy-paste - see debug-log.js. worker-engine.js installs its own copy of
// this (gated on cfg.debugTelemetry) since a Worker has its own separate
// console; this call only covers the main thread's half.
if (DEBUG_MODE) installDebugLogRelay('main');

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
  stopBtn: document.getElementById('stop-btn'),
  micSelect: document.getElementById('mic-select'),
  speakerSelect: document.getElementById('speaker-select'),
  inLevelFill: document.getElementById('in-level-fill'),
  outLevelFill: document.getElementById('out-level-fill'),
  pingBtn: document.getElementById('ping-btn'),
  feedbackWarning: document.getElementById('feedback-warning'),
  outputSinkWarning: document.getElementById('output-sink-warning'),
  concealingRate: document.getElementById('concealing-rate'),
  concealingRateValue: document.getElementById('concealing-rate-value'),
  concealerMemoryRate: document.getElementById('concealer-memory-rate'),
  concealerMemoryRateValue: document.getElementById('concealer-memory-rate-value'),
  concLevel: document.getElementById('conc-level'),
  concLevelValue: document.getElementById('conc-level-value'),
  denoiserSelect: document.getElementById('denoiser-select'),
  loadingMessage: document.getElementById('loading-message'),
  loadingProgress: document.getElementById('loading-progress'),
  loadingProgressFill: document.getElementById('loading-progress-fill'),
  status: document.getElementById('status'),
  progress: document.getElementById('progress'),
  latency: document.getElementById('latency'),
  debugStartStopBtn: document.getElementById('debug-start-stop-btn'),
  debugStopBtn: document.getElementById('debug-stop-btn'),
  debugStatus: document.getElementById('debug-status'),
  debugProgress: document.getElementById('debug-progress'),
  debugProgressFill: document.getElementById('debug-progress-fill'),
  listenOriginalToggle: document.getElementById('listen-original-toggle'),
  listenConcealerToggle: document.getElementById('listen-concealer-toggle'),
  vizRtVad: document.getElementById('viz-rt-vad'),
  vizSelection: document.getElementById('viz-selection'),
  vizSelectionDot: document.getElementById('viz-selection-dot'),
  vizDenoise: document.getElementById('viz-denoise'),
  vizMemVad: document.getElementById('viz-mem-vad'),
  vizCandidates: document.getElementById('viz-candidates'),
  vizMemoryDots: document.getElementById('viz-memory-dots'),
  vizMemoryCount: document.getElementById('viz-memory-count'),
};

let audioContext = null; // the real session's AudioContext (Start/Stop)
let micStream = null;
let sourceNode = null;
let workletNode = null;
let worker = null;
let denoiserWorker = null; // separate Worker/thread for denoiser inference - see denoiser-proxy.js for why it can't share worker-engine.js's thread
let running = false; // a session (mic or debug replay) exists - true whether it's actively playing OR paused; only stop() clears it
let paused = false; // the existing session's AudioContext is suspended (see pause() and play()'s resume branch) - the worker/concealer/memory queue stay fully alive and untouched, only real-time chunk delivery halts
let activeSource = null; // 'mic' | 'debug' | null (idle) - which source currently feeds the one shared engine; see connectSource()
let debugMode = false; // true while the current session's input is DEBUG_AUDIO_PATH instead of the mic - kept in sync with activeSource === 'debug' for any other code still reading it
let debugElapsedTimer = null;

// Idle mic-level preview + Ping share this lightweight context, separate
// from the real session's - mirrors amods.gui's separate "idle input
// monitor" vs. the real Stream.
let previewContext = null;
let previewSource = null;
let previewAnalyser = null;
let previewRafId = null;
let pingActive = false;

function rateToThreshold(rateStr) {
  return Math.round((1.0 - parseFloat(rateStr)) * 10) / 10;
}

// "Concealing purity" is the concealer's own memory-branch VAD threshold,
// and unlike "Concealing likelihood" (where a higher slider value should
// make the VAD MORE readily call something speech, hence rateToThreshold's
// 1-rate inversion), a higher purity value is meant to mean "stricter/more
// conservative about what gets stored" - which for vad-silero.js's
// speechRatio > threshold pass rule means a HIGHER threshold, not a lower
// one. So this maps the slider value straight to the threshold, no
// inversion (see the "concealing purity" investigation in conversation
// history for why the inherited 1-rate mapping was wrong for this slider).
function purityToThreshold(rateStr) {
  return Math.round(parseFloat(rateStr) * 10) / 10;
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

// ── Model preload ─────────────────────────────────────────────────────────
// Fetches every model file on page load purely to warm the browser's HTTP
// cache, so that when the user presses Start, worker-engine.js's and
// denoiser-worker.js's own ort.InferenceSession.create() calls (which fetch
// these same URLs) resolve from cache instead of hitting the network -
// that's what was making the first Start after opening the page slow,
// especially for the ~50MB denoiser. The Start button stays disabled (see
// its `disabled` attribute in index.html) until this finishes.
// DENOISER_MODEL_PATHS' values are written relative to docs/js/ (that's
// where denoiser-worker.js itself resolves them from) - fetch() here
// resolves relative to the page (docs/) instead, hence the '../' -> './'
// swap. Deriving this from DENOISER_MODEL_PATHS.int8 instead of
// hardcoding the filename is what's needed for this to stay in sync
// automatically whenever that mapping changes (this list previously drifted
// out of sync with it, silently preloading the wrong model file).
const MODEL_URLS = ['./models/silero_vad.onnx', DENOISER_MODEL_PATHS.int8.replace(/^\.\.\//, './')];

async function preloadModels() {
  const totalPerFile = new Array(MODEL_URLS.length).fill(0);
  const loadedPerFile = new Array(MODEL_URLS.length).fill(0);

  function updateProgressBar() {
    const total = totalPerFile.reduce((a, b) => a + b, 0);
    if (total <= 0) return; // no Content-Length known yet for any file
    const loaded = loadedPerFile.reduce((a, b) => a + b, 0);
    const pct = Math.min(100, (loaded / total) * 100);
    els.loadingProgressFill.style.width = `${pct}%`;
  }

  try {
    await Promise.all(
      MODEL_URLS.map(async (url, i) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText} (${url})`);
        totalPerFile[i] = Number(response.headers.get('content-length')) || 0;

        if (!response.body) {
          // Streaming response bodies aren't supported here - fall back to
          // an all-at-once wait with no incremental progress for this file.
          await response.arrayBuffer();
          loadedPerFile[i] = totalPerFile[i];
          updateProgressBar();
          return;
        }

        // fetch()'s own promise resolves once response headers arrive, not
        // once the body finishes downloading - reading the stream out (via
        // a reader instead of response.arrayBuffer(), to get incremental
        // byte counts for the progress bar) is what actually waits for
        // (and caches) the full file.
        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          loadedPerFile[i] += value.length;
          updateProgressBar();
        }
      })
    );
    els.loadingMessage.hidden = true;
    els.loadingProgress.hidden = true;
  } catch (e) {
    els.loadingMessage.textContent = `Could not preload models: ${e.message}. You can still press ▶ to load them then.`;
    els.loadingProgress.hidden = true;
  } finally {
    els.startStopBtn.disabled = false;
    if (DEBUG_MODE && els.debugStartStopBtn) els.debugStartStopBtn.disabled = false;
  }
}

// ── Start / Pause / Stop ────────────────────────────────────────────────
// The debug source (DEBUG_AUDIO_PATH, see connectSource() below): instead of
// opening the mic, decodes that file into an AudioBuffer and feeds it into
// the exact same AudioWorkletNode graph via an AudioBufferSourceNode. This is
// deliberately NOT any kind of offline/batch decode - an AudioBufferSourceNode
// connected to a live, running AudioContext plays out at the context's own
// real-time rate, exactly like a live MediaStreamSource would, so a 2-minute
// file takes 2 real minutes and exercises every real-time code path (VAD
// timing, the queue/staleness logic in worker-engine.js, etc.) rather than
// skipping past them.
// Sets up everything SHARED regardless of source (AudioContext, worklet,
// worker, denoiser worker, the worker's one-time 'init' - which is what
// creates its GranSpeechMask concealer and its empty memory queue) - runs
// exactly once, on the idle->running transition. Deliberately does NOT touch
// the audio source (mic vs debug file) - see connectSource() for that -
// so that switching sources later (mic <-> debug replay) never re-runs this
// and never re-sends 'init', which is what lets the memory queue survive a
// source switch instead of starting over empty.
async function ensureEngine() {
  if (running) return;

  audioContext = new AudioContext();
  const sr = audioContext.sampleRate;

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
      // The worker's own status loop keeps ticking every 500ms regardless of
      // pause() (it's a separate thread, unaffected by the main-thread
      // AudioContext being suspended) - skip applying it to the UI while
      // paused so "Paused" isn't immediately overwritten back to "Running".
      // Nothing is actually changing underneath during a pause anyway (no
      // new chunks are reaching the worker), so there's nothing lost by not
      // re-rendering it.
      if (!paused) {
        updateStatus(msg.status, msg.timing);
        if (DEBUG_MODE && msg.debug) updateConcealerViz(msg.debug);
      }
    } else if (msg.type === 'error') {
      els.status.textContent = `Engine error: ${msg.message}`;
    } else if (msg.type === 'ready') {
      els.status.textContent = 'Running';
    } else if (msg.type === 'denoiseRequest') {
      // Relay to the separate denoiser worker (see below) rather than
      // handling it here - the whole point is that this heavy inference
      // must never run on worker-engine.js's own thread.
      if (denoiserWorker) denoiserWorker.postMessage(msg, [msg.y.buffer]);
      else worker.postMessage({ type: 'denoiseError', id: msg.id, message: 'denoiser worker not available' });
    } else if (msg.type === 'debugMemoryClip' && DEBUG_MODE) {
      playDebugMemoryClip(msg.clip, msg.sr);
    } else if (msg.type === 'rtVad' && DEBUG_MODE) {
      // Sent every chunk (~50ms), not on the ~500ms status-poll cadence
      // everything else in updateConcealerViz runs on - see worker-engine.js's
      // drainQueue. Updated directly here (not gated on `paused` the way
      // updateStatus/updateConcealerViz are) since drainQueue itself simply
      // doesn't run while paused, so this naturally freezes on its own.
      if (els.vizRtVad) els.vizRtVad.classList.toggle('viz-block-active', Boolean(msg.voiceActive));
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

  // Denoiser inference runs in its own dedicated Worker (its own OS
  // thread), never on worker-engine.js's - a single Worker has only one
  // thread, so even a "fire and forget" call to the denoiser there would
  // still block real-time chunk processing for as long as inference takes
  // (measured: up to ~1s). This is the fix for that; see
  // denoiser-proxy.js's header comment for the full explanation.
  const denoiserPath = DENOISER_MODEL_PATHS[els.denoiserSelect.value]; // undefined for "none"
  if (denoiserPath) {
    denoiserWorker = new Worker('./js/denoiser-worker.js', { type: 'module' });
    denoiserWorker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'denoiseResult' || msg.type === 'denoiseError') {
        worker.postMessage(msg, msg.result ? [msg.result.buffer] : []);
      } else if (msg.type === 'error') {
        els.status.textContent = `Denoiser failed to load: ${msg.message}`;
        worker.postMessage({ type: 'denoiserInitError', message: msg.message });
      }
    };
    denoiserWorker.postMessage({ type: 'init', path: denoiserPath, sr });
  }

  worker.postMessage({
    type: 'init',
    config: {
      sr,
      denoiserEnabled: Boolean(denoiserPath),
      concealingThreshold: rateToThreshold(els.concealingRate.value),
      concealerMemoryThreshold: purityToThreshold(els.concealerMemoryRate.value),
      concMultiplier: dbToMultiplier(els.concLevel.value),
      // Full volume, not silenced - listenOriginal below is now what decides
      // whether the original mic signal is audible at all (see stream.js's
      // playSum), same role conc_multiplier already plays for the concealer
      // track. Muting it here too, on top of that, would make the "Original
      // audio" toggle turn it on at volume zero - exactly the bug that made
      // toggling it produce no audible change until this was fixed.
      monitorGain: 1.0,
      debugTelemetry: DEBUG_MODE,
      // Toggles only exist in debug.html's DOM - index.html keeps the
      // original always-concealer-only-on-speaker behavior these defaults
      // (false/true) already matched before either toggle existed.
      listenOriginal: els.listenOriginalToggle?.checked ?? false,
      listenConcealer: els.listenConcealerToggle?.checked ?? true,
    },
  });

  workletNode.connect(audioContext.destination);

  const deviceOut = selectedSpeakerId();
  if (deviceOut && typeof audioContext.setSinkId === 'function') {
    try {
      await audioContext.setSinkId(deviceOut);
    } catch (e) {
      els.outputSinkWarning.textContent = `Could not switch output device: ${e.message}`;
    }
  }
}

// Tears down whichever source (mic or debug file) is currently feeding
// workletNode, if any, then connects `newSource` instead. Never touches
// worker/concealer/memory - that's what lets switching between mic and debug
// replay preserve the memory queue instead of restarting it empty (the
// concealer has no idea its audio's origin changed mid-stream, same as it
// wouldn't notice you switching microphones).
async function connectSource(newSource) {
  if (debugElapsedTimer) {
    clearInterval(debugElapsedTimer);
    debugElapsedTimer = null;
  }
  if (sourceNode) {
    if (typeof sourceNode.stop === 'function') {
      try {
        sourceNode.stop(); // AudioBufferSourceNode (debug) only - MediaStreamAudioSourceNode has no stop()
      } catch {
        // Already ended naturally (onended already fired) - harmless.
      }
    }
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (els.debugStatus) els.debugStatus.textContent = '';
  if (els.debugProgress) els.debugProgress.hidden = true;
  if (els.debugProgressFill) els.debugProgressFill.style.width = '0%';

  if (newSource === 'debug') {
    const response = await fetch(DEBUG_AUDIO_PATH);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText} (${DEBUG_AUDIO_PATH})`);
    const arrayBuffer = await response.arrayBuffer();
    let audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    // Force mono, taking only the first channel - matches the mic path,
    // which requests channelCount: 1 from getUserMedia explicitly. Without
    // this, a stereo debug file's default channel handling when connected to
    // workletNode (channelCountMode "max", not forced down to 1) is up to
    // the browser rather than something we control, so a stereo file could
    // behave differently from the mono debug file this was tested with.
    // Rebuilding a genuinely single-channel buffer removes that ambiguity
    // entirely, regardless of how many channels the source file has.
    if (audioBuffer.numberOfChannels > 1) {
      const mono = audioContext.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);
      mono.copyToChannel(audioBuffer.getChannelData(0), 0);
      audioBuffer = mono;
    }
    sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(workletNode);
    sourceNode.start();
    const startedSourceNode = sourceNode;
    sourceNode.onended = () => {
      // Natural end of the file - same cleanup as pressing a Stop button,
      // but only if debug is STILL the active source and this is still the
      // current node: switching away already replaced sourceNode before
      // this could fire, and a stale onended from the node that switch just
      // discarded shouldn't tear down whatever's now playing instead.
      if (running && activeSource === 'debug' && sourceNode === startedSourceNode) stop();
    };
    const totalS = audioBuffer.duration;
    const startedAt = audioContext.currentTime;
    if (els.debugProgress) els.debugProgress.hidden = false;
    debugElapsedTimer = setInterval(() => {
      const elapsedS = Math.min(totalS, audioContext.currentTime - startedAt);
      if (els.debugStatus) els.debugStatus.textContent = `${elapsedS.toFixed(0)}s / ${totalS.toFixed(0)}s`;
      if (els.debugProgressFill) els.debugProgressFill.style.width = `${(elapsedS / totalS) * 100}%`;
    }, 500);
  } else {
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
    sourceNode = audioContext.createMediaStreamSource(micStream);
    sourceNode.connect(workletNode);
  }

  activeSource = newSource;
  debugMode = newSource === 'debug'; // kept in sync for any other code still reading it
}

// Entry point for both transport rows' Play button. From idle, this is a
// full cold start (ensureEngine() + connectSource()). While a session is
// already running, clicking the INACTIVE source's Play switches to it
// in-place (see connectSource's docstring for why that's memory-preserving)
// and, since picking a different source to listen to implies wanting to
// actually hear it, also resumes if the engine was paused.
async function play(newSource) {
  setAllTransportButtonsDisabled(true);
  els.status.textContent = running
    ? `Switching to ${newSource === 'debug' ? 'debug replay' : 'mic'}…`
    : (newSource === 'debug' ? 'Starting debug replay…' : 'Starting…');

  stopInputPreview();
  if (pingActive) return; // Ping's own finally{} will restore state; don't fight it

  try {
    if (!running) {
      await ensureEngine();
      running = true;
      paused = false;
      setControlsEnabled(false);
      await connectSource(newSource);
    } else {
      if (paused) {
        audioContext.resume();
        paused = false;
      }
      if (activeSource !== newSource) await connectSource(newSource);
    }
  } finally {
    updateStartStopUI();
  }
}

// Tears down EVERYTHING (worker terminated - and with it, the concealer and
// its whole memory queue - AudioContext closed, mic released, debug source
// stopped) regardless of which source was active. Either transport's Stop
// button calls this same function: there is only one shared engine/memory
// queue underneath both, so there's no such thing as stopping "just" the mic
// side or "just" the debug side - stopping resets both at once.
function stop() {
  setAllTransportButtonsDisabled(true);
  if (debugElapsedTimer) {
    clearInterval(debugElapsedTimer);
    debugElapsedTimer = null;
  }
  if (worker) {
    worker.postMessage({ type: 'stop' });
    worker.terminate();
    worker = null;
  }
  if (denoiserWorker) {
    denoiserWorker.terminate();
    denoiserWorker = null;
  }
  if (workletNode) {
    workletNode.port.postMessage({ type: 'stop' });
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    if (typeof sourceNode.stop === 'function') {
      try {
        sourceNode.stop(); // AudioBufferSourceNode (debug) only - MediaStreamAudioSourceNode has no stop()
      } catch {
        // Already ended naturally (onended already fired) - harmless.
      }
    }
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
  paused = false;
  activeSource = null;
  debugMode = false;
  els.status.textContent = 'Idle — press ▶ to begin';
  els.progress.textContent = '—';
  els.latency.textContent = '—';
  els.inLevelFill.style.width = '0%';
  els.outLevelFill.style.width = '0%';
  if (els.debugStatus) els.debugStatus.textContent = '';
  if (els.debugProgress) els.debugProgress.hidden = true;
  if (els.debugProgressFill) els.debugProgressFill.style.width = '0%';
  resetConcealerViz();
  setControlsEnabled(true);
  updateStartStopUI();
  startInputPreview();
}

// Pause suspends the EXISTING session's AudioContext - unlike stop(), the
// worker, its GranSpeechMask concealer, and everything in its memory queue
// stay fully alive and untouched throughout. Suspending an AudioContext
// halts every node's processing (the worklet's process() calls stop firing,
// so no more 'chunk' messages reach the worker - see worklet-processor.js),
// including a debug-mode AudioBufferSourceNode's playback position, which
// freezes and resumes exactly where it left off - there's no native pause()
// on that node type, so suspending the whole context is the standard way to
// achieve it. The mic stream itself (if any) stays open throughout, so
// resuming needs no new getUserMedia permission prompt and no re-creating
// the worker/concealer from scratch. (Resuming from a pause with the SAME
// source still active is handled inline in play() above, alongside the
// switch-source case - there's no separate resume() since "press this
// source's Play button" covers both.)
function pause() {
  if (!running || paused || !audioContext) return;
  audioContext.suspend();
  paused = true;
  els.status.textContent = 'Paused';
  updateStartStopUI();
}

function setAllTransportButtonsDisabled(disabled) {
  els.startStopBtn.disabled = disabled;
  els.stopBtn.disabled = disabled;
  if (DEBUG_MODE && els.debugStartStopBtn) els.debugStartStopBtn.disabled = disabled;
  if (DEBUG_MODE && els.debugStopBtn) els.debugStopBtn.disabled = disabled;
}

// Mic and debug replay are two independent Play/Pause+Stop transports (see
// index.html's/debug.html's .transport-row markup) sharing the ONE engine
// above. Only one source can be actively PLAYING at a time - while one is,
// the other's Play button is fully disabled (not just relabeled), so you
// can't switch out from under whichever source you're actively listening
// to by accident. Pausing the active one un-disables the other: with
// neither actively playing, either Play button is fair game - the active
// source's resumes it, the other's switches to it (see play(), which also
// auto-resumes in that case, since choosing a different source to listen to
// implies wanting to actually hear it). Either Stop button is always
// enabled together and tears down the one shared engine entirely,
// regardless of which source was active.
function updateStartStopUI() {
  if (!running) {
    els.startStopBtn.textContent = '▶';
    els.startStopBtn.title = 'Start';
    els.startStopBtn.setAttribute('aria-label', 'Start');
    els.startStopBtn.disabled = false;
    els.stopBtn.disabled = true;
    if (DEBUG_MODE && els.debugStartStopBtn) {
      els.debugStartStopBtn.textContent = '▶';
      els.debugStartStopBtn.title = 'Start debug replay';
      els.debugStartStopBtn.setAttribute('aria-label', 'Start debug replay');
      els.debugStartStopBtn.disabled = false;
    }
    if (DEBUG_MODE && els.debugStopBtn) els.debugStopBtn.disabled = true;
    return;
  }

  els.stopBtn.disabled = false;
  if (DEBUG_MODE && els.debugStopBtn) els.debugStopBtn.disabled = false;

  const micActive = activeSource === 'mic';
  if (micActive) {
    els.startStopBtn.disabled = false;
    if (paused) {
      els.startStopBtn.textContent = '▶';
      els.startStopBtn.title = 'Resume';
      els.startStopBtn.setAttribute('aria-label', 'Resume');
    } else {
      els.startStopBtn.textContent = '⏸';
      els.startStopBtn.title = 'Pause';
      els.startStopBtn.setAttribute('aria-label', 'Pause');
    }
  } else {
    // Debug replay is the active source - mic's button only comes alive
    // once that's paused (see this function's own docstring above).
    els.startStopBtn.textContent = '▶';
    els.startStopBtn.title = paused ? 'Switch to mic' : 'Pause debug replay first to switch to mic';
    els.startStopBtn.setAttribute('aria-label', 'Switch to mic');
    els.startStopBtn.disabled = !paused;
  }

  if (DEBUG_MODE && els.debugStartStopBtn) {
    const debugActive = activeSource === 'debug';
    if (debugActive) {
      els.debugStartStopBtn.disabled = false;
      if (paused) {
        els.debugStartStopBtn.textContent = '▶';
        els.debugStartStopBtn.title = 'Resume';
        els.debugStartStopBtn.setAttribute('aria-label', 'Resume');
      } else {
        els.debugStartStopBtn.textContent = '⏸';
        els.debugStartStopBtn.title = 'Pause';
        els.debugStartStopBtn.setAttribute('aria-label', 'Pause');
      }
    } else {
      els.debugStartStopBtn.textContent = '▶';
      els.debugStartStopBtn.title = paused ? 'Switch to debug replay' : 'Pause mic first to switch to debug replay';
      els.debugStartStopBtn.setAttribute('aria-label', 'Switch to debug replay');
      els.debugStartStopBtn.disabled = !paused;
    }
  }
}

function setControlsEnabled(enabled) {
  els.micSelect.disabled = !enabled;
  els.speakerSelect.disabled = !enabled;
  els.denoiserSelect.disabled = !enabled;
  els.pingBtn.disabled = !enabled;
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

  // Input latency: the Web Audio API has no direct equivalent of
  // PortAudio's input_stream.latency, but the mic->worker handoff has a
  // real, always-known contribution of its own - the worklet only forwards
  // a chunk once CHUNK_SIZE_AT_48K samples have accumulated. On top of
  // that, MediaStreamTrack.getSettings().latency reports the browser/OS's
  // own hardware capture latency where available (reliability varies a lot
  // by platform - e.g. often unpopulated for real devices on Linux), so
  // it's added in only when present rather than assumed to be 0.
  const bufferMs = audioContext ? (CHUNK_SIZE_AT_48K / audioContext.sampleRate) * 1000 : 0;
  const track = micStream ? micStream.getAudioTracks()[0] : null;
  const settings = track && track.getSettings ? track.getSettings() : {};
  const hwLatencyMs = typeof settings.latency === 'number' ? settings.latency * 1000 : 0;
  const inMs = bufferMs + hwLatencyMs;

  const totalMs = inMs + outMs + timing.avgMs;
  els.latency.textContent =
    `~${totalMs.toFixed(0)} ms  (in ${inMs.toFixed(0)} ms + out ${outMs.toFixed(0)} ms + algo ` +
    `${timing.avgMs.toFixed(1)}/${timing.maxMs.toFixed(1)} ms avg/max)`;
}

// ── Concealer visualization (DEBUG_MODE only) ──────────────────────────────
// Purely illustrative - reflects worker-engine.js's periodic debugSnapshot()
// (see granspeechmask.js/stream.js), never drives any actual behavior. The
// underlying computation (memVadSeq/memVadDurationMs) is untouched by any
// of this - these constants and timeouts only control how long the display
// keeps something visible, not how long anything actually takes.
const MEM_VAD_MIN_PULSE_MS = 400; // the memory branch's own VAD step is usually much faster than the 500ms poll interval - without a floor here it would almost never visibly appear at all
const CANDIDATES_DISPLAY_MS = 800; // how long a candidate slot stays colored before reverting to its empty look, not how long the slot itself exists (it always exists)
// How long the memory queue's "grow past its normal slot count, then snap
// back" eviction animation runs - see renderMemoryTransition below. Matches
// viz-enter-pop/viz-exit-shrink's own CSS animation duration so the visual
// growth/shrink and the actual DOM collapse happen at the same moment.
const EVICT_TRANSITION_MS = 700;
// nConcealerBeforeDenoise's current value (ceil(pendingVoiceMaxDuration /
// concealer_duration) = ceil(2/0.3), both fixed in worker-engine.js's
// concealerConfig) - shown before any real telemetry exists so the row is
// never empty, corrected from debug.candidateSlots once a cycle actually
// reports it (in case those constants ever change).
const CANDIDATE_SLOTS_DEFAULT = 7;
// memory_maxlen's current value (worker-engine.js's concealerConfig) - same
// "shown by default, corrected once real telemetry exists" reasoning as
// CANDIDATE_SLOTS_DEFAULT above.
const MEMORY_SLOTS_DEFAULT = 100;
let lastSeenMemVadSeq = 0; // 0 means "no cycle observed yet" - matches GranSpeechMask's own starting value, so the very first status poll after Start never fires a spurious pulse
let lastSeenFeedCycleSeq = 0;
let lastKnownCandidateSlots = CANDIDATE_SLOTS_DEFAULT;
let lastKnownMemorySlots = MEMORY_SLOTS_DEFAULT;
// How many of the memory queue's most-recently-added entries to force green
// (mirroring the candidates row's own green) until greenFlashUntil - set
// together whenever a cycle adds anything, read on every render below.
let memoryFlashCount = 0;
let memoryFlashUntil = 0;
// The memory-dots block below re-renders (full innerHTML replace) on every
// ~500ms status tick, not just once when a flash starts - and CANDIDATES_
// DISPLAY_MS (800ms) outlives a single tick, so without this flag the same
// entries would get their entering animation class re-applied to freshly-
// recreated elements on the next tick too, restarting the CSS animation from
// scratch (visible as the dots growing twice in a row instead of once). Set
// true right after the first render that includes the flash (which may be
// the eviction-transition render below, if this cycle evicted anything, or
// otherwise the plain per-tick render), cleared whenever a new flash window
// begins.
let memoryFlashAnimated = false;
// debug.memory always reflects the post-merge state - by the time a cycle's
// evicted entries show up here, GranSpeechMask has already dropped them from
// its own array, so there is no "real" data left to animate them with. This
// holds the previous tick's debug.memory (captured at the end of every tick,
// below) purely so that WHEN an eviction is detected, we still have each
// evicted entry's actual last-seen classification (cooldown/tooCloseToBuffer)
// to render it fading out with, rather than guessing. Feed cycles are spaced
// seconds apart in practice (far more than the ~500ms poll interval), so
// "previous tick" reliably means "just before this cycle's merge".
let previousMemorySnapshot = [];
// Sitting at Date.now() + EVICT_TRANSITION_MS while an eviction's grow/shrink
// animation (see renderMemoryTransition) is on screen - the plain per-tick
// render at the bottom of updateConcealerViz backs off until this passes, so
// it doesn't collapse the transitional (over-100) view before the animation
// has had a chance to play.
let evictTransitionUntil = 0;

function renderDots(container, dotSpecs) {
  if (!container) return;
  container.innerHTML = dotSpecs
    .map(({ cls, title, index }) => {
      // Only memory-queue dots that correspond to a real, currently-live
      // memory entry carry `index` (see classifyMemoryEntry) - that's what
      // makes them clickable (see the delegated click listener below) and
      // everything else (empty/reserved slots, entries mid-eviction-fade-out)
      // inert, since there'd be nothing live left to play for those.
      const dataAttr = index != null ? ` data-mem-index="${index}"` : '';
      return `<span class="viz-dot ${cls}" title="${title || ''}"${dataAttr}></span>`;
    })
    .join('');
}

function emptyCandidateDots(count) {
  return Array.from({ length: count }, () => ({ cls: 'viz-dot-empty', title: 'empty slot' }));
}

function emptyMemoryDots(count) {
  return Array.from({ length: count }, () => ({ cls: 'viz-dot-empty', title: 'empty slot' }));
}

// A cycle can add at most candidateSlots entries (that's every candidate
// passing VAD), and evicts exactly that many once the queue is full - so the
// queue's rendered DOM element count needs to be able to reach maxlen +
// candidateSlots without ever growing the number of flex-wrapped rows versus
// what's shown at rest. Rendering that many elements at ALL times (idle,
// normal, mid-transition), padding with invisible-but-space-reserving
// placeholders wherever the real content doesn't fill it, keeps the
// rectangle's row count constant - so a transition never pops a new row into
// existence only to have it vanish again once eviction finishes.
function reservedMemoryDots(count) {
  return Array.from({ length: Math.max(0, count) }, () => ({ cls: 'viz-dot-reserved', title: '' }));
}

function classifyMemoryEntry(entry, i, debug) {
  // i is -1 for entries rendered from previousMemorySnapshot mid-eviction
  // (see renderMemoryTransition's evictedSpecs) - those have already been
  // dropped from the concealer's real memory array by the time we see them,
  // so there's no live index left to request playback for.
  const index = i >= 0 ? i : undefined;
  if (i === debug?.selectedIndex) return { cls: 'viz-dot-blue', title: 'currently selected - click to play', index };
  if (entry.cooldown > 0) return { cls: 'viz-dot-orange', title: `on hold - reused ${entry.cooldown} cycles ago - click to play`, index };
  if (entry.tooCloseToBuffer) return { cls: 'viz-dot-orange', title: 'on hold - too close to the live buffer - click to play', index };
  return { cls: 'viz-dot-neutral', title: 'available - click to play', index };
}

// The queue's normal, fixed-slot-count render - used both on every plain
// status tick and as the "snap back to size" step right after an eviction's
// transition animation finishes (see renderMemoryTransition).
function renderMemoryQueue(debug) {
  const memory = debug.memory || [];
  lastKnownMemorySlots = debug.memoryMaxlen || lastKnownMemorySlots;
  const flashing = Date.now() < memoryFlashUntil;
  const flashFromIndex = memory.length - memoryFlashCount;
  const animateThisRender = flashing && !memoryFlashAnimated;
  const candidateSlots = debug.candidateSlots || lastKnownCandidateSlots;
  const slots = Array.from({ length: lastKnownMemorySlots }, (_, i) => {
    if (i >= memory.length) return { cls: 'viz-dot-empty', title: 'empty slot' };
    if (flashing && i >= flashFromIndex) {
      const enterClass = animateThisRender ? ' viz-dot-entering' : '';
      return { cls: `viz-dot-green${enterClass}`, title: 'just added to memory - click to play', index: i };
    }
    return classifyMemoryEntry(memory[i], i, debug);
  });
  renderDots(els.vizMemoryDots, [...slots, ...reservedMemoryDots(candidateSlots)]);
  if (animateThisRender) memoryFlashAnimated = true;
}

// Plays the "grow past the normal slot count, then snap back" animation: the
// entries actually being evicted (their last-known look, from
// previousMemorySnapshot) render fading out at the far left, at the same
// moment the newly-added entries pop in at the far right - past the queue's
// normal end, not in place of anything - so the queue is visibly longer
// (maxlen + evictedSpecs.length dots) for EVICT_TRANSITION_MS. Once that
// timer fires, renderMemoryQueue's normal fixed-length render takes over
// again; since post-merge memory is exactly evictedSpecs-stripped-from-the-
// front, the remaining dots are already in their final resting positions -
// removing the (by-then invisible) faded-out prefix just closes the gap.
function renderMemoryTransition(evictedSpecs, debug) {
  const memory = debug.memory || [];
  const flashFromIndex = memory.length - memoryFlashCount;
  const exitingSpecs = evictedSpecs.map(({ cls, title }) => ({ cls: `${cls} viz-dot-exiting`, title }));
  const newSpecs = memory.map((entry, i) => {
    if (i >= flashFromIndex) return { cls: 'viz-dot-green viz-dot-entering', title: 'just added to memory - click to play', index: i };
    return classifyMemoryEntry(entry, i, debug);
  });
  // Same total element count as the normal render (maxlen + candidateSlots) -
  // just with fewer reserved placeholders this time, since evictedSpecs is
  // temporarily occupying part of that reserved room instead of sitting idle.
  const candidateSlots = debug.candidateSlots || lastKnownCandidateSlots;
  const reserved = reservedMemoryDots(candidateSlots - exitingSpecs.length);
  renderDots(els.vizMemoryDots, [...exitingSpecs, ...newSpecs, ...reserved]);
  memoryFlashAnimated = true; // the entering pop-in for the new tail already played above - the follow-up renderMemoryQueue call must not replay it
}

// Adds the active class right away, then removes it after durationMs -
// restarting the timer if called again before the previous one fired, so
// back-to-back pulses don't cut each other's display time short.
function pulseActive(el, durationMs) {
  if (!el) return;
  el.classList.add('viz-block-active');
  clearTimeout(el.__pulseTimeout);
  el.__pulseTimeout = setTimeout(() => el.classList.remove('viz-block-active'), durationMs);
}

// Like pulseActive, but for a row of dots: shows dotSpecs immediately, then
// reverts to idleSpecs after durationMs - "reverts to", not "clears to
// nothing", so the slots themselves are always present (see
// CANDIDATE_SLOTS_DEFAULT above) and only their fill ever changes.
function pulseDots(container, dotSpecs, durationMs, idleSpecs) {
  if (!container) return;
  renderDots(container, dotSpecs);
  clearTimeout(container.__pulseTimeout);
  container.__pulseTimeout = setTimeout(() => renderDots(container, idleSpecs), durationMs);
}

// Both rows are "there by default", even before Start is ever pressed -
// not tied to any worker message.
if (DEBUG_MODE && els.vizCandidates) renderDots(els.vizCandidates, emptyCandidateDots(lastKnownCandidateSlots));
if (DEBUG_MODE && els.vizMemoryDots) {
  renderDots(els.vizMemoryDots, [...emptyMemoryDots(lastKnownMemorySlots), ...reservedMemoryDots(lastKnownCandidateSlots)]);
  // Delegated (not per-dot) since renderDots fully replaces the container's
  // children on every render - a listener attached to an individual <span>
  // would be gone the next time this queue re-renders. Only dots carrying
  // data-mem-index (see classifyMemoryEntry/renderDots) are live memory
  // entries; clicking an empty/reserved/fading-out one does nothing.
  els.vizMemoryDots.addEventListener('click', (event) => {
    const dot = event.target.closest('[data-mem-index]');
    if (!dot || !worker) return;
    worker.postMessage({ type: 'debugPlayMemoryClip', index: Number(dot.dataset.memIndex) });
  });
}

// Debug-visualization only - plays a stored memory clip's actual audio
// (clicked from the memory queue above) directly to the speakers, bypassing
// the whole real-time worklet/mixing pipeline entirely - this is just a
// one-shot preview, not something that should touch pendingConc/recording.
// A dedicated context, separate from the real session's audioContext (same
// idea as previewContext for the idle mic-level meter/Ping below) - reusing
// audioContext directly would mean a click while paused silently produces no
// sound, since a suspended AudioContext doesn't render ANY node's output,
// including a freshly started one. This one is never suspended by pause(),
// so clicking a memory dot plays it regardless of whether the main session
// is running, paused, or (once memory has any entries) even stopped.
let debugPlaybackContext = null;

function playDebugMemoryClip(clip, clipSr) {
  if (!debugPlaybackContext) debugPlaybackContext = new AudioContext();
  if (debugPlaybackContext.state === 'suspended') debugPlaybackContext.resume();
  const buffer = debugPlaybackContext.createBuffer(1, clip.length, clipSr);
  buffer.copyToChannel(clip, 0);
  const src = debugPlaybackContext.createBufferSource();
  src.buffer = buffer;
  // Same gain a real concealer insertion would get in the live pipeline
  // (see conc_multiplier in worker-engine.js/stream.js) - so a clip that
  // sounds too loud/quiet here reflects how it'd actually sound if picked
  // as a real concealer, not just its own raw recorded level.
  const gain = debugPlaybackContext.createGain();
  gain.gain.value = dbToMultiplier(els.concLevel.value);
  src.connect(gain);
  gain.connect(debugPlaybackContext.destination);
  src.start();
}

function updateConcealerViz(debug) {
  // vizRtVad is NOT updated here anymore - it now reacts live to the
  // worker's own dedicated per-chunk 'rtVad' message (see worker.onmessage
  // above) instead of this ~500ms poll, so it visibly lights up/off at
  // real detection speed (~50ms) rather than in up-to-500ms-late steps.

  // The block's own status dot (purple, like every other block) says "the
  // concealing branch is active"; the separate blue dot after the label
  // specifically says "and a memory entry is currently selected" - same
  // color and same timing (both driven by this one `concealing` value) as
  // that entry's own blue dot in the memory queue below, to visually tie
  // the two together.
  const concealing = debug.selectedIndex != null;
  if (els.vizSelection) els.vizSelection.classList.toggle('viz-block-active', concealing);
  if (els.vizSelectionDot) {
    els.vizSelectionDot.className = `viz-dot ${concealing ? 'viz-dot-blue' : 'viz-dot-empty'}`;
    els.vizSelectionDot.title = concealing ? `memory index #${debug.selectedIndex} is selected` : 'no clip currently selected';
  }

  // The denoise pass is fast enough in practice that a live elapsed-time
  // readout isn't worth the label-width juggling it used to require - the
  // status dot lighting up for exactly as long as denoiseRunning is true is
  // enough on its own to show when it starts and ends.
  if (els.vizDenoise) els.vizDenoise.classList.toggle('viz-block-active', Boolean(debug.denoiseRunning));

  // A new feed cycle starting is when we first know how many candidates
  // there will be (candidateSlots is a fixed constant, not counted after
  // the fact). The slots themselves are already visible by default (see
  // module init above) - this just makes sure they're back to empty/white
  // right as a new cycle begins, in case stale colors from a previous
  // cycle's display window are still showing.
  if (debug.feedCycleSeq != null && debug.feedCycleSeq > lastSeenFeedCycleSeq) {
    lastSeenFeedCycleSeq = debug.feedCycleSeq;
    lastKnownCandidateSlots = debug.candidateSlots || lastKnownCandidateSlots;
    if (els.vizCandidates) clearTimeout(els.vizCandidates.__pulseTimeout);
    renderDots(els.vizCandidates, emptyCandidateDots(lastKnownCandidateSlots));
  }

  // Unlike denoising (which genuinely takes long enough to poll mid-flight),
  // the memory branch's VAD step is over almost instantly - polling for
  // "is it running right now" would essentially never catch it. Instead,
  // memVadSeq incrementing at all tells us a cycle just completed, and we
  // pulse the indicator + candidates for a floor duration regardless of how
  // fast the real step was.
  if (debug.memVadSeq != null && debug.memVadSeq > lastSeenMemVadSeq) {
    lastSeenMemVadSeq = debug.memVadSeq;
    const candidates = debug.lastCandidates || [];
    pulseActive(els.vizMemVad, Math.max(debug.memVadDurationMs || 0, MEM_VAD_MIN_PULSE_MS));
    pulseDots(
      els.vizCandidates,
      candidates.map((passed) => ({
        cls: passed ? 'viz-dot-green' : 'viz-dot-red',
        title: passed ? 'passed - added to memory' : 'failed - discarded',
      })),
      CANDIDATES_DISPLAY_MS,
      emptyCandidateDots(candidates.length || lastKnownCandidateSlots)
    );

    // Candidates that passed VAD get pushed onto the end of memory (see
    // GranSpeechMask._feedMemory's merge step - new entries are always
    // appended, and any trimming to stay within memory_maxlen only ever
    // removes from the front) - so the last N = (passed count) entries are
    // exactly this cycle's newly-added ones. Flash them green too, for the
    // same duration as the candidates row, so it reads as "these candidates
    // became these memory entries" rather than two unrelated color changes.
    const addedCount = candidates.filter(Boolean).length;
    if (addedCount > 0) {
      memoryFlashCount = addedCount;
      memoryFlashUntil = Date.now() + CANDIDATES_DISPLAY_MS;
      memoryFlashAnimated = false;
      // debug.memory already reflects the post-merge state - if it was
      // already at capacity, adding addedCount entries while staying at
      // that same capacity necessarily evicted addedCount entries from the
      // front (see the comment above: trimming only ever removes from the
      // front). If it wasn't full yet, nothing was evicted, just grown - the
      // plain per-tick render below already handles that case (entries just
      // pop in at the tail, the queue itself isn't at its limit yet).
      const maxlen = debug.memoryMaxlen || lastKnownMemorySlots;
      const evictCount = (debug.memory || []).length >= maxlen ? addedCount : 0;
      if (evictCount > 0) {
        const evictedSpecs = previousMemorySnapshot
          .slice(0, evictCount)
          .map((entry) => classifyMemoryEntry(entry, -1, null)); // -1/null: these positions have already shifted out of the new array, so "currently selected" can't apply to them
        clearTimeout(els.vizMemoryDots.__evictTimeout);
        renderMemoryTransition(evictedSpecs, debug);
        evictTransitionUntil = Date.now() + EVICT_TRANSITION_MS;
        els.vizMemoryDots.__evictTimeout = setTimeout(() => {
          evictTransitionUntil = 0;
          renderMemoryQueue(debug);
        }, EVICT_TRANSITION_MS);
      }
    }
  }

  // While an eviction's grow/shrink transition is showing (see above), skip
  // the plain fixed-slot render - it would otherwise immediately overwrite
  // the transitional (over-limit) view with the collapsed one before the
  // animation has had a chance to play. The scheduled setTimeout above (or,
  // if nothing is evicting, this tick's own call right here) takes over once
  // it's safe to.
  if (Date.now() >= evictTransitionUntil) renderMemoryQueue(debug);
  // Captured AFTER this tick's own merge-detection block above (if any) has
  // already read it, so it always holds "memory just before the next merge" -
  // see its declaration for why that's exactly what an eviction render needs.
  previousMemorySnapshot = debug.memory || [];
  if (els.vizMemoryCount) els.vizMemoryCount.textContent = `${(debug.memory || []).length} / ${debug.memoryMaxlen ?? '—'}`;
}

function resetConcealerViz() {
  if (!DEBUG_MODE) return;
  lastSeenMemVadSeq = 0; // so a fresh session's first real cycle pulses again, rather than being mistaken for a repeat of the last session's
  lastSeenFeedCycleSeq = 0;
  memoryFlashCount = 0;
  memoryFlashUntil = 0;
  memoryFlashAnimated = false;
  previousMemorySnapshot = [];
  evictTransitionUntil = 0;
  if (els.vizMemoryDots) clearTimeout(els.vizMemoryDots.__evictTimeout);
  if (els.vizRtVad) els.vizRtVad.classList.remove('viz-block-active');
  if (els.vizSelection) els.vizSelection.classList.remove('viz-block-active');
  if (els.vizSelectionDot) {
    els.vizSelectionDot.className = 'viz-dot viz-dot-empty';
    els.vizSelectionDot.title = 'no clip currently selected';
  }
  if (els.vizDenoise) els.vizDenoise.classList.remove('viz-block-active');
  if (els.vizMemVad) {
    els.vizMemVad.classList.remove('viz-block-active');
    clearTimeout(els.vizMemVad.__pulseTimeout);
  }
  if (els.vizCandidates) clearTimeout(els.vizCandidates.__pulseTimeout);
  renderDots(els.vizCandidates, emptyCandidateDots(lastKnownCandidateSlots)); // slots stay visible even when idle/stopped - see module init above
  renderDots(els.vizMemoryDots, [...emptyMemoryDots(lastKnownMemorySlots), ...reservedMemoryDots(lastKnownCandidateSlots)]);
  if (els.vizMemoryCount) els.vizMemoryCount.textContent = '';
}

// ── Event wiring ──────────────────────────────────────────────────────────
els.startStopBtn.addEventListener('click', () => {
  if (running && activeSource === 'mic' && !paused) {
    pause();
  } else {
    // Covers idle (cold start), paused-on-mic (resume), and paused-on-debug
    // (switch to mic + auto-resume) - see play()'s own docstring. This
    // button is disabled by updateStartStopUI() whenever debug is actively
    // (unpaused) playing, so that case never reaches here.
    play('mic').catch((e) => {
      els.status.textContent = `Failed to start: ${e.message}`;
      updateStartStopUI();
    });
  }
});

els.stopBtn.addEventListener('click', () => {
  if (running) stop();
});

if (DEBUG_MODE && els.debugStartStopBtn) {
  els.debugStartStopBtn.addEventListener('click', () => {
    if (running && activeSource === 'debug' && !paused) {
      pause();
    } else {
      play('debug').catch((e) => {
        els.status.textContent = `Failed to start debug replay: ${e.message}`;
        updateStartStopUI();
      });
    }
  });
}

if (DEBUG_MODE && els.debugStopBtn) {
  els.debugStopBtn.addEventListener('click', () => {
    if (running) stop();
  });
}

els.pingBtn.addEventListener('click', () => onPing());

els.micSelect.addEventListener('change', () => {
  updateWarnings();
  restartInputPreview();
});
els.speakerSelect.addEventListener('change', () => updateWarnings());

els.concealingRate.addEventListener('input', () => {
  const threshold = rateToThreshold(els.concealingRate.value);
  els.concealingRateValue.textContent = els.concealingRate.value;
  if (worker && running) worker.postMessage({ type: 'setConcealingThreshold', value: threshold });
});

els.concealerMemoryRate.addEventListener('input', () => {
  const threshold = purityToThreshold(els.concealerMemoryRate.value);
  els.concealerMemoryRateValue.textContent = els.concealerMemoryRate.value;
  if (worker && running) worker.postMessage({ type: 'setConcealerMemoryThreshold', value: threshold });
});

els.concLevel.addEventListener('input', () => {
  const mult = dbToMultiplier(els.concLevel.value);
  els.concLevelValue.textContent = `${parseFloat(els.concLevel.value) >= 0 ? '+' : ''}${els.concLevel.value} dB`;
  if (worker && running) worker.postMessage({ type: 'setConcMultiplier', value: mult });
});

if (DEBUG_MODE && els.listenOriginalToggle) {
  els.listenOriginalToggle.addEventListener('change', () => {
    if (worker && running) worker.postMessage({ type: 'setListenOriginal', value: els.listenOriginalToggle.checked });
  });
}

if (DEBUG_MODE && els.listenConcealerToggle) {
  els.listenConcealerToggle.addEventListener('change', () => {
    if (worker && running) worker.postMessage({ type: 'setListenConcealer', value: els.listenConcealerToggle.checked });
  });
}

populateDevices().catch((e) => {
  els.status.textContent = `Could not list audio devices: ${e.message}`;
});
preloadModels();
