"""
Desktop UI for the Granular Speech Masker demo: configure mic/speaker, VAD, and
denoiser, then start/stop real-time concealing without hand-editing YAML
files and launching the CLI by hand.

Settings picked here are written to config files that :class:`amods.stream.Stream`
reads too (a local ``./configs/`` directory when run from a checkout of this
repository, otherwise ``~/.amods/configs/``), so the ``amods`` CLI picks up
whatever was last configured here.
"""
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
import yaml

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError as exc:  # pragma: no cover - platform dependent
    raise ImportError(
        "The GUI needs Tkinter, which is not installed. On Debian/Ubuntu: "
        "`sudo apt install python3-tk`."
    ) from exc

from .config import load_config, user_config_dir
from .stream import Stream

OUTPUT_DIR = "./output/"

WEBRTC_DEFAULT_LOGIT_THRESHOLD = 0.5


def _writable_config_path(component, name):
    """
    Where the GUI should write a config file so that a Stream built right
    after picks it up: prefer a local ``./configs/`` directory (a checkout of
    this repository, or any directory the user has set one up in), otherwise
    fall back to the per-user ``~/.amods/configs/`` directory.
    """
    local_dir = Path("configs") / component
    if local_dir.is_dir():
        return local_dir / name

    path = user_config_dir() / component / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path):
    """Load a YAML file into a dict, or return ``{}`` if it doesn't exist yet."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data):
    """Write ``data`` to ``path`` as YAML, preserving key order."""
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def list_devices(direction):
    """List ``(index, name)`` for every audio device with at least one channel in the given ``direction`` ("in" or "out")."""
    channel_key = "max_input_channels" if direction == "in" else "max_output_channels"
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d[channel_key] > 0]


def is_builtin_device_name(name):
    """Best-effort guess at whether ``name`` is the laptop's own built-in sound hardware (see comment below)."""
    # Raw ALSA hardware endpoints of the laptop's own sound chip look like
    # "sof-hda-dsp: ... (hw:0,x)" - unambiguously built-in. Generic gateway
    # devices ("pulse", "pipewire", "default") can resolve to anything
    # (built-in, Bluetooth, USB) depending on the OS's live routing, so they
    # can't be reliably classified as built-in or not from the name alone.
    name = name.lower()
    return "hw:" in name or "sof-hda-dsp" in name


class ConcealerGUI:
    """Tkinter window for the Granular Speech Masker: builds all widgets, then starts/stops a Stream on demand."""

    def __init__(self, root):
        """Build every widget onto ``root`` and pre-select sensible device/VAD/denoiser defaults."""
        self.root = root
        root.title("Granular Speech Masker")
        root.resizable(False, False)

        self.input_stream = None
        self.output_stream = None
        self.concealer_stream = None
        self._refresh_job = None

        self.in_devices = list_devices("in")
        self.out_devices = list_devices("out")
        default_in, default_out = sd.default.device

        pad = {"padx": 10, "pady": 6}

        tk.Label(root, text="Granular Speech Masker", font=("", 16, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(12, 4)
        )

        # ── Audio devices ────────────────────────────────────────────────
        devices_frame = ttk.LabelFrame(root, text="Audio devices")
        devices_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(devices_frame, text="Mic (in)").grid(row=0, column=0, sticky="w", **pad)
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(
            devices_frame, textvariable=self.in_var, state="readonly", width=42,
            values=[f"{i}: {name}" for i, name in self.in_devices],
        )
        self.in_combo.grid(row=0, column=1, **pad)
        self._preselect(self.in_combo, self.in_devices, default_in)
        self.in_combo.bind("<<ComboboxSelected>>", lambda e: self._update_warnings())

        ttk.Label(devices_frame, text="Speaker (out)").grid(row=1, column=0, sticky="w", **pad)
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(
            devices_frame, textvariable=self.out_var, state="readonly", width=42,
            values=[f"{i}: {name}" for i, name in self.out_devices],
        )
        self.out_combo.grid(row=1, column=1, **pad)
        self._preselect(self.out_combo, self.out_devices, default_out)
        self._avoid_same_device_default()
        self.out_combo.bind("<<ComboboxSelected>>", lambda e: self._update_warnings())

        # Warnings (Larsen-effect feedback risk + same-device duplex crash risk)
        self.feedback_warning = tk.Label(
            root, text="", fg="#cc5500", font=("", 9, "bold"),
            wraplength=420, justify="center",
        )
        self.feedback_warning.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)

        # ── VAD settings ─────────────────────────────────────────────────
        vad_frame = ttk.LabelFrame(root, text="VAD settings")
        vad_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(vad_frame, text="VAD type").grid(row=0, column=0, sticky="w", **pad)
        self.vad_type_var = tk.StringVar(value="ten")
        self.vad_type_combo = ttk.Combobox(
            vad_frame, textvariable=self.vad_type_var, state="readonly", width=20,
            values=["ten", "silero", "webrtc"],
        )
        self.vad_type_combo.grid(row=0, column=1, sticky="w", **pad)
        self.vad_type_combo.bind("<<ComboboxSelected>>", lambda e: self._on_vad_type_change())

        self.vad_param_label_var = tk.StringVar(value="Threshold")
        ttk.Label(vad_frame, textvariable=self.vad_param_label_var).grid(
            row=1, column=0, sticky="w", **pad
        )
        self.vad_param_var = tk.DoubleVar(value=0.5)
        self.vad_param_value_label = ttk.Label(vad_frame, text="0.5")
        self.vad_param_value_label.grid(row=1, column=2, sticky="w")
        self.vad_param_scale = tk.Scale(
            vad_frame, from_=0.1, to=0.9, resolution=0.1, orient="horizontal",
            variable=self.vad_param_var, showvalue=False, length=200,
            command=lambda v: self.vad_param_value_label.config(text=v),
        )
        self.vad_param_scale.grid(row=1, column=1, **pad)

        # ── Denoiser ─────────────────────────────────────────────────────
        denoiser_frame = ttk.LabelFrame(root, text="Denoiser")
        denoiser_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(denoiser_frame, text="Denoiser").grid(row=0, column=0, sticky="w", **pad)
        self.denoiser_var = tk.StringVar(value="FbDM")
        self.denoiser_combo = ttk.Combobox(
            denoiser_frame, textvariable=self.denoiser_var, state="readonly", width=20,
            values=["FbDM", "None"],
        )
        self.denoiser_combo.grid(row=0, column=1, sticky="w", **pad)

        # ── Start/Stop + status ──────────────────────────────────────────
        self.start_stop_btn = ttk.Button(root, text="Start", command=self._on_start_stop)
        self.start_stop_btn.grid(row=5, column=0, columnspan=2, pady=(4, 8))

        status_frame = ttk.Frame(root)
        status_frame.grid(row=6, column=0, columnspan=2, pady=(0, 4))
        ttk.Label(status_frame, text="Progress:").grid(row=0, column=0, sticky="e")
        self.progress_var = tk.StringVar(value="—")
        ttk.Label(status_frame, textvariable=self.progress_var, font=("", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=(6, 0)
        )
        ttk.Label(status_frame, text="Latency:").grid(row=1, column=0, sticky="e")
        self.latency_var = tk.StringVar(value="—")
        ttk.Label(status_frame, textvariable=self.latency_var, font=("", 10, "bold")).grid(
            row=1, column=1, sticky="w", padx=(6, 0)
        )

        self.status_var = tk.StringVar(value="Idle — press Start to begin")
        ttk.Label(root, textvariable=self.status_var, foreground="gray").grid(
            row=7, column=0, columnspan=2, pady=(0, 12)
        )

        self._on_vad_type_change()
        self._update_warnings()

    # ── Device selection helpers ─────────────────────────────────────────

    def _preselect(self, combo, devices, default_index):
        """Select ``default_index`` in ``combo`` if it's one of ``devices``, else the first device, else nothing."""
        indices = [i for i, _ in devices]
        if default_index in indices:
            combo.current(indices.index(default_index))
        elif devices:
            combo.current(0)

    def _selected_index(self, combo, devices):
        """Return the device index currently selected in ``combo``, or None if nothing is selected."""
        if not combo.get():
            return None
        return devices[combo.current()][0]

    def _avoid_same_device_default(self):
        """If the pre-selected mic and speaker are the same device, switch the speaker to a different one if possible."""
        # PortAudio's own reported default is often the same generic gateway
        # device for both directions (e.g. ALSA "default" on Linux) - exactly
        # the combination known to crash on simultaneous input+output. If a
        # different output option exists, prefer it over the initial default
        # so the app doesn't launch pre-armed with a known-bad pairing.
        in_idx = self._selected_index(self.in_combo, self.in_devices)
        out_idx = self._selected_index(self.out_combo, self.out_devices)
        if in_idx is None or out_idx is None or in_idx != out_idx:
            return
        for pos, (idx, _) in enumerate(self.out_devices):
            if idx != in_idx:
                self.out_combo.current(pos)
                return

    def _update_warnings(self):
        """Refresh the on-screen warning banner for the currently selected mic/speaker pair (same-device or built-in-both risk)."""
        in_idx = self._selected_index(self.in_combo, self.in_devices)
        out_idx = self._selected_index(self.out_combo, self.out_devices)
        in_name = self.in_devices[self.in_combo.current()][1] if self.in_combo.get() else ""
        out_name = self.out_devices[self.out_combo.current()][1] if self.out_combo.get() else ""

        if in_idx is not None and in_idx == out_idx:
            self.feedback_warning.config(
                text="Warning: Same device selected for mic and speaker: this has been observed to "
                     "crash (segfault) when opened for simultaneous input+output. Pick two "
                     "different devices."
            )
        elif is_builtin_device_name(in_name) and is_builtin_device_name(out_name):
            self.feedback_warning.config(
                text="Warning: Using the laptop's built-in mic and built-in loudspeaker together may "
                     "cause audio feedback (howling) once concealing starts. Consider "
                     "headphones for the speaker, or a headset mic for the input."
            )
        else:
            self.feedback_warning.config(text="")

    # ── VAD threshold / aggressiveness slider ────────────────────────────

    def _on_vad_type_change(self):
        """Reconfigure the threshold/aggressiveness slider's range and default for the newly selected VAD type."""
        vad_type = self.vad_type_var.get()
        if vad_type == "webrtc":
            self.vad_param_label_var.set("Aggressiveness")
            self.vad_param_scale.config(from_=0, to=3, resolution=1)
            self.vad_param_var.set(3)
            self.vad_param_value_label.config(text="3")
        else:
            default_threshold = 0.5
            self.vad_param_label_var.set("Threshold")
            self.vad_param_scale.config(from_=0.1, to=0.9, resolution=0.1)
            self.vad_param_var.set(default_threshold)
            self.vad_param_value_label.config(text=str(default_threshold))

    # ── Config writing ────────────────────────────────────────────────────

    def _write_configs(self, device_in, device_out):
        """Persist the currently selected devices, VAD type/threshold, and denoiser choice to the stream/concealer/vad configs."""
        stream_path = _writable_config_path("stream", "default.yaml")
        stream_config = load_yaml(stream_path) or load_config("stream", "default")
        stream_config["device_in"] = device_in
        stream_config["device_out"] = device_out
        save_yaml(stream_path, stream_config)

        concealer_path = _writable_config_path("concealer", "default.yaml")
        concealer_config = load_yaml(concealer_path) or load_config("concealer", "default")
        concealer_config["denoise"] = self.denoiser_var.get() == "FbDM"
        concealer_config["random_reverse"] = False
        concealer_config["concealer_vad_config"] = "default_concealer.yaml"
        save_yaml(concealer_path, concealer_config)

        vad_type = self.vad_type_var.get()
        vad_config = {"vad_type": vad_type}
        if vad_type == "webrtc":
            vad_config["logit_threshold"] = WEBRTC_DEFAULT_LOGIT_THRESHOLD
            vad_config["aggressiveness"] = int(self.vad_param_var.get())
        else:
            vad_config["logit_threshold"] = round(float(self.vad_param_var.get()), 1)
        save_yaml(_writable_config_path("vad", "default_source.yaml"), vad_config)
        save_yaml(_writable_config_path("vad", "default_concealer.yaml"), vad_config)

    # ── Start / Stop ──────────────────────────────────────────────────────

    def _on_start_stop(self):
        """Handle the Start/Stop button: stop the running stream if there is one, else start a new one."""
        if self.input_stream is not None:
            self._stop()
        else:
            self._start()

    def _start(self):
        """Validate the selected devices, persist the current settings, and open the live input/output streams."""
        device_in = self._selected_index(self.in_combo, self.in_devices)
        device_out = self._selected_index(self.out_combo, self.out_devices)

        if device_in is None or device_out is None:
            messagebox.showerror("Granular Speech Masker", "Select a mic and a speaker first.")
            return

        if device_in == device_out:
            messagebox.showwarning(
                "Granular Speech Masker",
                "Mic and speaker are set to the same device. This has been observed to "
                "crash (segfault) when opened for simultaneous input+output on some "
                "systems. Starting anyway.",
            )

        try:
            self._write_configs(device_in, device_out)
            self.concealer_stream = Stream(
                "default_source", "default", "default", "default", is_stream=True
            )
            self.concealer_stream.reset_state()

            stream_config = self.concealer_stream.stream_config
            # Two independent streams instead of one combined duplex stream:
            # the input side keeps the algorithm's fixed ~50ms blocksize
            # (needed for VAD/feature extraction), while the output stream is
            # free to use whatever (typically much smaller) blocksize the
            # backend prefers, instead of both directions sharing one
            # buffering granularity. They're connected by a ring buffer (see
            # Stream.input_callback/output_callback in amods.stream).
            self.input_stream = sd.InputStream(
                samplerate=stream_config["sr"],
                blocksize=self.concealer_stream.buffer_size,
                dtype=stream_config["dtype"],
                channels=stream_config["channels_in"],
                device=stream_config["device_in"],
                callback=self.concealer_stream.input_callback,
                latency=stream_config.get("input_latency", "low"),
            )
            self.output_stream = sd.OutputStream(
                samplerate=stream_config["sr"],
                dtype=stream_config["dtype"],
                channels=stream_config["channels_out"],
                device=stream_config["device_out"],
                callback=self.concealer_stream.output_callback,
                latency=stream_config.get("output_latency", "high"),
            )
            self.input_stream.start()
            self.output_stream.start()
        except Exception as e:
            messagebox.showerror("Granular Speech Masker", f"Failed to start: {e}")
            if self.input_stream is not None:
                self.input_stream.close()
            if self.output_stream is not None:
                self.output_stream.close()
            self.input_stream = None
            self.output_stream = None
            self.concealer_stream = None
            return

        self.start_stop_btn.config(text="Stop")
        self._set_controls_enabled(False)
        self._refresh_status()

    def _stop(self):
        """Close the live input/output streams and save the recorded original/concealer/mix WAVs."""
        if self._refresh_job is not None:
            self.root.after_cancel(self._refresh_job)
            self._refresh_job = None

        self.input_stream.stop()
        self.input_stream.close()
        self.input_stream = None

        self.output_stream.stop()
        self.output_stream.close()
        self.output_stream = None

        stream = self.concealer_stream
        self.concealer_stream = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if stream.rec_original:
            original = np.concatenate(stream.rec_original, axis=0).astype(np.float32)
            concealer = np.concatenate(stream.rec_concealer, axis=0).astype(np.float32)
            mix = np.concatenate(stream.rec_mix, axis=0).astype(np.float32)
            sr = stream.stream_config["sr"]
            sf.write(f"{OUTPUT_DIR}/stream_original.wav", original, sr)
            sf.write(f"{OUTPUT_DIR}/stream_concealer.wav", concealer, sr)
            sf.write(f"{OUTPUT_DIR}/stream_mix.wav", mix, sr)
            self.status_var.set(f"Stopped — saved to {OUTPUT_DIR}")
        else:
            self.status_var.set("Stopped")

        self.start_stop_btn.config(text="Start")
        self.progress_var.set("—")
        self.latency_var.set("—")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled):
        """Enable or disable every setting widget (locked while a stream is running, see comment below)."""
        # None of these take effect on an already-running Stream (it's built
        # once, from a snapshot of these values, when Start is pressed) - lock
        # them all while running so it's clear that moving the slider now
        # would do nothing until Stop/Start again.
        combo_state = "readonly" if enabled else "disabled"
        scale_state = "normal" if enabled else "disabled"
        self.in_combo.config(state=combo_state)
        self.out_combo.config(state=combo_state)
        self.vad_type_combo.config(state=combo_state)
        self.vad_param_scale.config(state=scale_state)
        self.denoiser_combo.config(state=combo_state)

    def _refresh_status(self):
        """Poll the running stream's progress/latency and update the status labels; reschedules itself every 500ms."""
        if self.concealer_stream is not None:
            status = self.concealer_stream.concealer.status()
            progress, target, threshold, label = (
                status["progress"], status["target"], status.get("threshold"), status["label"]
            )
            if progress is not None and target is not None:
                self.progress_var.set(f"{progress} / {target}" + (f" ({label})" if label else ""))
            elif progress is not None:
                self.progress_var.set(str(progress))
            else:
                self.progress_var.set("—")

            if not status["ready"]:
                what = label or "the algorithm"
                if progress is not None and threshold is not None:
                    self.status_var.set(f"Waiting for {what} to warm up ({progress}/{threshold}) before concealing starts")
                else:
                    self.status_var.set(f"Waiting for {what} to warm up before concealing starts")
            else:
                self.status_var.set("")

            in_ms = self.input_stream.latency * 1000
            out_ms = self.output_stream.latency * 1000
            algo_avg_ms, algo_max_ms = self.concealer_stream.pop_callback_timing()
            total_ms = in_ms + out_ms + algo_avg_ms
            self.latency_var.set(
                f"~{total_ms:.0f} ms  (in {in_ms:.0f} ms + out {out_ms:.0f} ms + algo "
                f"{algo_avg_ms:.1f}/{algo_max_ms:.1f} ms avg/max)"
            )

            self._refresh_job = self.root.after(500, self._refresh_status)


def main():
    """Entry point for the ``amods-gui`` console script."""
    root = tk.Tk()
    ConcealerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
