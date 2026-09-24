import os
import threading
import time as _time

import numpy as np
import soundfile as sf

from .audio import limit_peak
from .config import resolve_config
from .models.concealer import select_concealer_model
from .models.forecaster import select_forecaster_model
from .models.vad import select_vad_model


class _AudioRingBuffer:
    """
    Thread-safe FIFO of mono float32 samples.

    Lets the algorithm keep computing in fixed ~50ms chunks (needed for
    VAD/feature extraction) while a separately-scheduled output stream drains
    samples using its own, typically much smaller, blocksize - instead of
    forcing both directions to share one buffering granularity (see
    input_callback/output_callback on Stream).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, samples):
        """Append ``samples`` to the end of the buffer."""
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])

    def pull(self, n):
        """Remove and return the first ``n`` samples, zero-padded if fewer than ``n`` are available."""
        with self._lock:
            available = len(self._buffer)
            if available >= n:
                out = self._buffer[:n].copy()
                self._buffer = self._buffer[n:]
            else:
                out = np.zeros(n, dtype=np.float32)
                if available:
                    out[:available] = self._buffer
                self._buffer = np.zeros(0, dtype=np.float32)
        return out

    def clear(self):
        """Drop all buffered samples."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)


class Stream:
    def __init__(self, source_vad_config_name, concealer_config_name, stream_config_name, forecaster_config_name, is_stream, freeze_learning=False,
                 record_mode="memory", record_outdir=None, record_outprefix=""):
        """
        Each ``*_config_name`` accepts either a config name/path (str,
        resolved via :func:`amods.config.load_config`) or an already-built
        config dict, used directly - the latter lets you embed a Stream
        without any YAML files, or override just a few fields on top of a
        named config:

            config = amods.config.load_config("concealer", "default")
            config["denoise"] = False
            Stream(source_vad_config_name="default_source",
                   concealer_config_name=config, ...)

        ``record_mode`` controls what happens to the original/concealer/mix
        audio produced every callback:

        - ``"memory"`` (default): accumulate it in the ``rec_original`` /
          ``rec_concealer`` / ``rec_mix`` lists, exactly like every previous
          version of this class - the caller is responsible for
          concatenating and writing them (see ``run_file_mode``). Fine for
          short/offline runs, but for a long live session the ever-growing
          lists and the single giant concatenate-then-write at the end can
          make stopping take a very long time and use a lot of memory.
        - ``"disk"``: write each block straight to disk as it arrives
          (``record_outdir``/``record_outprefix`` name the files), and never
          accumulate the in-memory lists at all - stopping is then just
          closing already-fully-written files, regardless of session length.
          Call :meth:`finalize_recording` when done to close them.
        - ``"none"``: don't keep or write the audio at all - for a live
          session where nothing needs to be saved.
        """
        self.source_vad_config_name = source_vad_config_name
        self.stream_config_name = stream_config_name
        self.concealer_config_name = concealer_config_name
        self.forecaster_config_name = forecaster_config_name
        self.stream_config = resolve_config("stream", stream_config_name)
        self.concealer_config = resolve_config("concealer", concealer_config_name)
        # Lets the concealer's own VAD config (named inside the concealer
        # config) be found next to a custom concealer config, e.g.
        # "myconfigs/concealer/foo.yaml" -> "myconfigs/vad/...". Meaningless
        # (and left empty) when concealer_config_name is an inline dict
        # rather than a path - there's no directory to derive a sibling from.
        self.config_path = (
            os.path.dirname(os.path.dirname(concealer_config_name))
            if isinstance(concealer_config_name, str) else ""
        )
        self.source_vad_config = resolve_config("vad", source_vad_config_name)
        self.forecaster_config = resolve_config("forecaster", forecaster_config_name)
        self.is_stream = is_stream
        self.concealer_config["is_stream"] = is_stream
        self.concealer_config["freeze_learning"] = freeze_learning

        self.pending_conc_max_size = 3 * self.stream_config["sr"]  # 3 seconds max pending concealer
        self.concealer_config["pending_conc_max_size"] = self.pending_conc_max_size
        self.pending_conc = np.zeros(self.pending_conc_max_size, dtype=np.float32)

        # Gain state carried across chunks so limit_peak's ramp stays
        # continuous from one call to the next instead of snapping to a
        # freshly-computed value each time (see its docstring) - one per
        # signal path, since play_mix and the recorded mix can need
        # different amounts of limiting at any given moment.
        self._play_mix_gain = 1.0
        self._rec_mix_gain = 1.0

        # Derived sizes
        self.buffer_size = int(self.stream_config["sr"] * self.stream_config["buffer_duration"])

        self.concealer = select_concealer_model(self.stream_config["sr"], self.config_path, self.concealer_config)
        self.source_vad = select_vad_model(self.source_vad_config, sr=self.stream_config["sr"])
        self.forecaster = select_forecaster_model(self.forecaster_config, sr=self.stream_config["sr"])
        self.rec_original = []   # raw input (mono) - populated only in record_mode="memory"
        self.rec_concealer = []  # concealer output (mono) - populated only in record_mode="memory"
        self.rec_concealer_metadata = []
        self.rec_mix = []        # recorded mix (stereo) - populated only in record_mode="memory"

        if record_mode not in ("memory", "disk", "none"):
            raise ValueError(f"record_mode must be 'memory', 'disk', or 'none', got {record_mode!r}")
        self.record_mode = record_mode
        self.record_paths = None
        self._record_writers = None
        if record_mode == "disk":
            self._open_record_writers(record_outdir, record_outprefix)

        # Used only by the split-stream real-time mode (input_callback/
        # output_callback): decouples the algorithm's fixed ~50ms chunk size
        # from the output stream's own (typically much smaller) blocksize.
        self._output_ring = _AudioRingBuffer()

        # Algorithm compute time per callback, collected by pop_callback_timing()
        # (e.g. from a UI polling loop) to show the algorithm's own contribution
        # to end-to-end latency, on top of the audio device's I/O latency.
        self._callback_ms_sum = 0.0
        self._callback_ms_max = 0.0
        self._callback_ms_count = 0

    def pop_callback_timing(self):
        """Return (avg_ms, max_ms) of callback compute time since the last call, then reset."""
        count = self._callback_ms_count
        avg_ms = (self._callback_ms_sum / count) if count else 0.0
        max_ms = self._callback_ms_max
        self._callback_ms_sum = 0.0
        self._callback_ms_max = 0.0
        self._callback_ms_count = 0
        return avg_ms, max_ms

    def _open_record_writers(self, record_outdir, record_outprefix):
        """Open the original/concealer/mix .wav files for incremental (record_mode="disk") writing."""
        os.makedirs(record_outdir, exist_ok=True)
        prefix = os.path.join(record_outdir, record_outprefix)
        sr = self.stream_config["sr"]
        n_out_channels = self.stream_config["channels_out"]
        self.record_paths = {
            "original": f"{prefix}_original.wav",
            "concealer": f"{prefix}_concealer.wav",
            "mix": f"{prefix}_mix.wav",
        }
        self._record_writers = {
            "original": sf.SoundFile(self.record_paths["original"], mode="w", samplerate=sr, channels=1),
            "concealer": sf.SoundFile(self.record_paths["concealer"], mode="w", samplerate=sr, channels=1),
            "mix": sf.SoundFile(self.record_paths["mix"], mode="w", samplerate=sr, channels=n_out_channels),
        }

    def finalize_recording(self):
        """
        Close the disk writers opened for record_mode="disk" (a no-op, close
        to instant, since every block was already written as it arrived) and
        return the dict of file paths that were written, or None if this
        Stream wasn't recording to disk.
        """
        if self._record_writers is None:
            return None
        for writer in self._record_writers.values():
            writer.close()
        self._record_writers = None
        return self.record_paths

    def _process_chunk(self, x, n_out_channels):
        """
        Run VAD + concealer + mixing for one input chunk. Returns the mono
        play_mix to send to the speaker, and records original/concealer/mix
        for saving. Shared by the combined single-stream callback() and the
        split-stream input_callback().
        """
        forecast = self.forecaster.predict(x)

        voice_activity = self.source_vad.predict(x)
        if voice_activity:
            concealer_audio, voice_name = self.concealer.get_concealer(forecast)
            concealer_audio = (concealer_audio * self.stream_config["conc_multiplier"]).astype(np.float32, copy=False)
            self.pending_conc[:len(concealer_audio)] += concealer_audio
        else:
            self.concealer.refresh(x)
            voice_name = ""

        # ---- Produce concealer for this chunk ----
        frames = len(x)
        conc_block = self.pending_conc[:frames].copy()

        self.pending_conc[:-frames] = self.pending_conc[frames:]
        self.pending_conc[-frames:] = 0.0

        # ---- Build playback output (what you hear) ----
        play_mic = self.stream_config["monitor_gain"] * x
        play_mix, self._play_mix_gain = limit_peak(
            play_mic + conc_block, limit=0.95, prev_gain=self._play_mix_gain, sr=self.stream_config["sr"]
        )

        # ---- Build recorded mix (what goes into *_mix.wav) ----
        rec_mic = self.stream_config["record_mic_gain"] * x
        rec_mix_mono, self._rec_mix_gain = limit_peak(
            rec_mic + conc_block, limit=0.95, prev_gain=self._rec_mix_gain, sr=self.stream_config["sr"]
        )

        # ---- Record (shared) ----
        if self.record_mode == "memory":
            self.rec_original.append(x.copy())
            self.rec_concealer.append(conc_block.copy())
            self.rec_concealer_metadata.append(voice_name)
            self.rec_mix.append(np.tile(rec_mix_mono[:, np.newaxis], (1, n_out_channels)))
        elif self.record_mode == "disk":
            self._record_writers["original"].write(x)
            self._record_writers["concealer"].write(conc_block)
            self._record_writers["mix"].write(np.tile(rec_mix_mono[:, np.newaxis], (1, n_out_channels)))
        # record_mode == "none": nothing kept or written.

        return play_mix

    def _record_callback_timing(self, t0, frames):
        """Record this callback's compute time (started at ``t0``) and warn if it exceeded its time budget."""
        elapsed = _time.perf_counter() - t0
        elapsed_ms = elapsed * 1000
        self._callback_ms_sum += elapsed_ms
        self._callback_ms_max = max(self._callback_ms_max, elapsed_ms)
        self._callback_ms_count += 1

        budget = frames / self.stream_config["sr"]
        if elapsed > budget:
            print(f"[SLOW CALLBACK] took {elapsed_ms:.1f}ms, budget was {budget*1000:.1f}ms")

    def callback(self, indata, outdata, frames, time, status):
        """
        Combined single-stream callback: input and output tied to the same
        fixed blocksize (used by the file-mode emulation, where there's no
        real device buffering trade-off to decouple).
        """
        if status:
            print(status)

        _t0 = _time.perf_counter()

        x = indata[:, 0].astype(np.float32, copy=False)
        play_mix = self._process_chunk(x, outdata.shape[1])
        outdata[:] = play_mix[:, np.newaxis]

        self._record_callback_timing(_t0, frames)

    def input_callback(self, indata, frames, time, status):
        """
        Input half of the split-stream real-time mode: runs the algorithm on
        a fixed ~50ms chunk exactly like callback(), but instead of writing
        directly to an output buffer, pushes the result into a ring buffer
        that a separate, independently-scheduled OutputStream drains (see
        output_callback) - so the output stream isn't forced to share the
        input's ~50ms buffering granularity.
        """
        if status:
            print(status)

        _t0 = _time.perf_counter()

        x = indata[:, 0].astype(np.float32, copy=False)
        play_mix = self._process_chunk(x, self.stream_config["channels_out"])
        self._output_ring.push(play_mix)

        self._record_callback_timing(_t0, frames)

    def output_callback(self, outdata, frames, time, status):
        """
        Output half of the split-stream real-time mode: drains the ring
        buffer fed by input_callback, using whatever (typically much
        smaller) blocksize this independent OutputStream negotiates.
        """
        if status:
            print(status)

        play_mix = self._output_ring.pull(frames)
        outdata[:] = play_mix[:, np.newaxis]

    def reset_state(self):
        """Reset all algorithm state + recording buffers."""
        self.pending_conc = np.zeros(self.pending_conc_max_size, dtype=np.float32)
        self._play_mix_gain = 1.0
        self._rec_mix_gain = 1.0

        self.rec_original.clear()
        self.rec_concealer.clear()
        self.rec_concealer_metadata.clear()
        self.rec_mix.clear()
        self._output_ring.clear()

    def copy(self, freeze_learning=False):
        """Return an independent Stream with the same configs and a copy of the concealer's runtime state."""
        new_stream = Stream(
            self.source_vad_config_name,
            self.concealer_config_name,
            self.stream_config_name,
            self.forecaster_config_name,
            self.is_stream,
            freeze_learning=freeze_learning
        )

        # Copy runtime state explicitly
        new_stream.concealer = self.concealer.copy()

        return new_stream


def seconds_to_hms(seconds):
    """Format a duration in seconds as ``HH:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_file_mode(
    indir,
    infile,
    outdir,
    outprefix,
    concealerconfig,
    streamconfig,
    sourcevadconfig,
    forecasterconfig,
    debug=False,
):
    """Simulate the real-time callback over a pre-recorded audio file."""
    import librosa
    import soundfile as sf

    stream = Stream(sourcevadconfig, concealerconfig, streamconfig, forecasterconfig, is_stream=False)
    stream.reset_state()
    debug_dict = {}

    y, _ = librosa.load(os.path.join(indir, infile), sr=stream.stream_config["sr"], mono=True)
    y = y.astype(np.float32, copy=False)

    n_orig = len(y)

    # Pad to full blocks
    n_blocks = int(np.ceil(n_orig / stream.buffer_size))
    pad = n_blocks * stream.buffer_size - n_orig
    if pad > 0:
        y = np.pad(y, (0, pad), mode="constant")

    for b in range(n_blocks):
        block = y[b * stream.buffer_size: (b + 1) * stream.buffer_size]
        indata = block.reshape(-1, 1)                                   # (frames, 1)
        outdata = np.zeros((stream.buffer_size, 2), dtype=np.float32)   # (frames, 2)
        stream.callback(indata, outdata, stream.buffer_size, None, None)

        processed = min((b + 1) * stream.buffer_size, n_orig)
        elapsed_s = processed / stream.stream_config["sr"]
        total_s = n_orig / stream.stream_config["sr"]
        remaining_s = max(total_s - elapsed_s, 0.0)

        if debug:
            print(
                f"\rProgress: {seconds_to_hms(elapsed_s)}/{seconds_to_hms(total_s)} "
                f"({100.0 * processed / n_orig:5.1f}%) | "
                f"remaining: {seconds_to_hms(remaining_s)}",
                end="",
                flush=True,
            )
            debug_dict[elapsed_s] = stream.concealer.status()["progress"]

    # Concatenate recordings
    original = np.concatenate(stream.rec_original, axis=0)[:n_orig]
    concealer = np.concatenate(stream.rec_concealer, axis=0)[:n_orig]
    mix = np.concatenate(stream.rec_mix, axis=0)[:n_orig]

    # Calculate dB conc vs original (use mono RMS for comparison)
    original_rms = np.sqrt(np.mean(original**2))
    output_rms = np.sqrt(np.mean((original + concealer)**2))
    db_conc = 20 * np.log10(output_rms / original_rms) if original_rms > 0 else 0

    print(f"Original RMS (mono): {original_rms:.6f}")
    print(f"Output RMS (downmixed): {output_rms:.6f}")
    print(f"dB conc (downmixed): {db_conc:.2f} dB")

    out_prefix = outprefix if outprefix != "" else infile.split(".")[0]
    os.makedirs(outdir, exist_ok=True)

    print(f'Original: {original.shape}, Concealer: {concealer.shape}, Mix: {mix.shape}')

    sf.write(os.path.join(outdir, f"{out_prefix}_original.wav"), original, stream.stream_config["sr"])
    sf.write(os.path.join(outdir, f"{out_prefix}_concealer.wav"), concealer, stream.stream_config["sr"])
    sf.write(os.path.join(outdir, f"{out_prefix}_mix.wav"), mix, stream.stream_config["sr"])

    if debug:
        np.save(os.path.join(outdir, f"{out_prefix}_debug.npy"), debug_dict)

    print(f"Saved:\n  {out_prefix}_original.wav\n  {out_prefix}_concealer.wav\n  {out_prefix}_mix.wav")
    return stream


def run_stream_mode(outdir, outprefix, concealerconfig, streamconfig, sourcevadconfig, forecasterconfig):
    """Run the concealer live against the system mic/speaker until Ctrl+C."""
    import sounddevice as sd
    import soundfile as sf

    stream = Stream(sourcevadconfig, concealerconfig, streamconfig, forecasterconfig, is_stream=True)
    stream.reset_state()

    print(stream.stream_config)

    print("No added algorithmic latency.")
    print("Headphones: monitor_gain*mic + concealer")
    print("Recording mix: record_mic_gain*mic + concealer")
    print("Press Ctrl+C to stop and save:\n"
          "  stream_original.wav\n"
          "  stream_concealer.wav\n"
          "  stream_mix.wav\n")

    stream_config = stream.stream_config
    try:
        # Two independent streams instead of one combined duplex stream: the
        # input side keeps the algorithm's fixed ~50ms blocksize (needed for
        # VAD/feature extraction), while the output stream is free to use
        # whatever (typically much smaller) blocksize the backend prefers,
        # instead of both directions being forced to share one buffering
        # granularity. They're connected by a ring buffer (see
        # Stream.input_callback/output_callback).
        with sd.InputStream(
            samplerate=stream_config["sr"],
            blocksize=stream.buffer_size,
            dtype=stream_config["dtype"],
            channels=stream_config["channels_in"],
            device=stream_config["device_in"],
            callback=stream.input_callback,
            latency=stream_config.get("input_latency", "low"),
        ), sd.OutputStream(
            samplerate=stream_config["sr"],
            dtype=stream_config["dtype"],
            channels=stream_config["channels_out"],
            device=stream_config["device_out"],
            callback=stream.output_callback,
            latency=stream_config.get("output_latency", "high"),
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nStopping, writing files...")

        original = np.concatenate(stream.rec_original, axis=0).astype(np.float32)
        concealer = np.concatenate(stream.rec_concealer, axis=0).astype(np.float32)
        mix = np.concatenate(stream.rec_mix, axis=0).astype(np.float32)

        out_prefix = outprefix if outprefix != "" else "stream"
        os.makedirs(outdir, exist_ok=True)

        sf.write(os.path.join(outdir, f"{out_prefix}_original.wav"), original, stream.stream_config["sr"])
        sf.write(os.path.join(outdir, f"{out_prefix}_concealer.wav"), concealer, stream.stream_config["sr"])
        sf.write(os.path.join(outdir, f"{out_prefix}_mix.wav"), mix, stream.stream_config["sr"])

        print(f"Saved: {out_prefix}_original.wav, {out_prefix}_concealer.wav, {out_prefix}_mix.wav")
