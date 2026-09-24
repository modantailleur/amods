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
import threading
import time
from pathlib import Path

import numpy as np
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

DEFAULT_OUTPUT_DIR = "./output/"

WEBRTC_DEFAULT_LOGIT_THRESHOLD = 0.5

# Level meters: RMS-in-dBFS is mapped onto this range to get a 0-100 bar
# value. Chosen so that normal speech (roughly -25 to -15 dBFS) sits in the
# upper half of the bar rather than pinned near either end.
LEVEL_METER_DB_RANGE = (-55.0, -5.0)
LEVEL_METER_MIN_UPDATE_INTERVAL = 0.05  # seconds; caps bar redraws to ~20/s

# "Ping" test tone played through the selected speaker.
PING_FREQUENCY_HZ = 440.0
PING_DURATION_S = 0.7
PING_FADE_S = 0.05
PING_AMPLITUDE = 0.4

# Output-latency slider: tunable so a user can raise it until Ping stops
# lagging on a device whose default is too aggressive.
OUTPUT_LATENCY_MIN_MS = 0
OUTPUT_LATENCY_MAX_MS = 200
OUTPUT_LATENCY_STEP_MS = 5
OUTPUT_LATENCY_DEFAULT_MS = 80


class _Tooltip:
    """Minimal hover tooltip: shows `text` in a small borderless window near `widget` while the mouse is over it."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tipwindow = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self._tipwindow is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, font=("", 9), wraplength=220,
        ).pack(ipadx=4, ipady=2)

    def _hide(self, _event=None):
        if self._tipwindow is not None:
            self._tipwindow.destroy()
            self._tipwindow = None


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

        # Idle mic-level preview: a lightweight input-only stream open
        # whenever the real concealer isn't running, so the level bar next to
        # the mic selector reacts before the user presses Start.
        self._input_monitor = None
        self._ping_stream = None
        self._ping_busy = False
        self._start_busy = False
        self._last_level_ts = {"in": 0.0, "out": 0.0}

        style = ttk.Style()
        style.theme_use("clam")  # 'clam' reliably honors the custom colors below on every platform
        style.configure(
            "Level.Horizontal.TProgressbar",
            troughcolor="#e3e6ea", bordercolor="#c7cdd6",
            background="#2f6fed", lightcolor="#2f6fed", darkcolor="#2f6fed",
            thickness=12,
        )

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
        self.in_combo.bind("<<ComboboxSelected>>", lambda e: self._on_input_device_change())

        self.in_level_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            devices_frame, orient="horizontal", mode="determinate", length=90,
            maximum=100, variable=self.in_level_var, style="Level.Horizontal.TProgressbar",
        ).grid(row=0, column=2, padx=(0, 10), pady=6)

        # ── Output latency (tunable via Ping - see the info tooltip) ───────
        latency_box = ttk.Frame(devices_frame)
        latency_box.grid(row=0, column=3, padx=(0, 10), pady=6, sticky="ew")

        latency_header = ttk.Frame(latency_box)
        latency_header.pack(fill="x")
        ttk.Label(latency_header, text="Latency", font=("", 8)).pack(side="left")
        # Drawn by hand instead of using a Unicode "info" glyph (e.g. ⓘ)
        # - not every system font includes one, and a missing glyph renders
        # as a dotted placeholder box instead. A small filled circle + plain
        # "i" can't have that problem. The canvas background is matched to
        # the surrounding ttk frame so only the circular badge is visible,
        # not a mismatched square behind it.
        bg_color = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        info_icon = tk.Canvas(latency_header, width=16, height=16, highlightthickness=0, bg=bg_color)
        info_icon.pack(side="left", padx=(2, 0))
        info_icon.create_oval(0, 0, 13, 13, fill="#2f6fed", outline="")
        info_icon.create_text(8, 8, text="i", font=("", 7), fill="white")
        _Tooltip(info_icon, "Set the latency to a level where the ping doesn't lag.")
        self.output_latency_value_label = ttk.Label(latency_header, text="", font=("", 8))
        self.output_latency_value_label.pack(side="right")

        self.output_latency_var = tk.DoubleVar()
        self.output_latency_scale = tk.Scale(
            latency_box, from_=OUTPUT_LATENCY_MIN_MS, to=OUTPUT_LATENCY_MAX_MS,
            resolution=OUTPUT_LATENCY_STEP_MS, orient="horizontal", showvalue=False,
            length=100, variable=self.output_latency_var, command=self._on_output_latency_change,
        )
        self.output_latency_scale.pack(fill="x")

        ttk.Label(devices_frame, text="Speaker (out)").grid(row=1, column=0, sticky="w", **pad)
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(
            devices_frame, textvariable=self.out_var, state="readonly", width=42,
            values=[f"{i}: {name}" for i, name in self.out_devices],
        )
        self.out_combo.grid(row=1, column=1, **pad)
        self._preselect(self.out_combo, self.out_devices, default_out)
        self._avoid_same_device_default()
        self._reset_output_latency_default()
        self.out_combo.bind("<<ComboboxSelected>>", lambda e: self._on_output_device_change())

        self.out_level_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            devices_frame, orient="horizontal", mode="determinate", length=90,
            maximum=100, variable=self.out_level_var, style="Level.Horizontal.TProgressbar",
        ).grid(row=1, column=2, padx=(0, 10), pady=6)

        self.ping_btn = ttk.Button(devices_frame, text="Ping", width=6, command=self._on_ping)
        self.ping_btn.grid(row=1, column=3, padx=(0, 10), pady=6)

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

        # ── Recording ────────────────────────────────────────────────────
        # Off by default: a session only gets written to disk if you ask for
        # it, and doing so streams straight to the output files as audio
        # arrives (see Stream's record_mode="disk") instead of holding the
        # whole session in memory and writing it all at once on Stop - so
        # stopping stays fast no matter how long you were recording.
        recording_frame = ttk.Frame(root)
        recording_frame.grid(row=5, column=0, columnspan=2, pady=(0, 4))

        self.save_recording_var = tk.BooleanVar(value=False)
        self.save_check = ttk.Checkbutton(
            recording_frame, text="Save recording to:", variable=self.save_recording_var,
        )
        self.save_check.grid(row=0, column=0, padx=(0, 6))

        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_dir_entry = ttk.Entry(recording_frame, textvariable=self.output_dir_var, width=28)
        self.output_dir_entry.grid(row=0, column=1)

        # ── Start/Stop + status ──────────────────────────────────────────
        self.start_stop_btn = ttk.Button(root, text="Start", command=self._on_start_stop)
        self.start_stop_btn.grid(row=6, column=0, columnspan=2, pady=(4, 8))

        status_frame = ttk.Frame(root)
        status_frame.grid(row=7, column=0, columnspan=2, pady=(0, 4))
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
            row=8, column=0, columnspan=2, pady=(0, 12)
        )

        self._on_vad_type_change()
        self._update_warnings()
        self._start_input_monitor()
        self._prefetch_models()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """
        Close the window and hard-exit the process immediately, bypassing
        Python's normal interpreter shutdown entirely.

        A background thread closing streams (an earlier version of this)
        isn't enough: sounddevice itself registers an atexit hook
        (_exit_handler) that calls .stop()/.close() again on the
        most-recently-used stream during normal shutdown - on the main
        thread, after main() returns, completely outside our control. If
        that device blocks (the same "device I/O can hang forever" issue as
        Ping/Start - see _ping_worker), the interpreter hangs there in
        native code, which even Ctrl+C can't interrupt (SIGINT is only
        handled between Python bytecode instructions, never inside a
        blocked C call). os._exit() skips atexit handlers and all other
        Python-level cleanup, so nothing sounddevice does afterwards can
        block process exit - the OS reclaims every audio device handle when
        the process actually terminates anyway.
        """
        self.root.destroy()
        os._exit(0)

    # ── Model prefetch ───────────────────────────────────────────────────

    def _prefetch_models(self):
        """
        Load the Silero VAD model and download the FbDM denoiser weights now,
        in the background, instead of leaving that to happen the moment
        Start is first pressed. Silero's model ships bundled inside its
        package (loading it is fast and fully offline); the FbDM denoiser's
        weights are the one genuine network download here, fetched on first
        use and cached to disk afterwards, so this is a no-op on every launch
        after the first on a given machine. (webrtc/ten VAD need neither, so
        they're not touched here.)

        Start is disabled meanwhile: building the real Stream on Start would
        otherwise risk a second, concurrent download of the same file if the
        user was fast enough to press it before this finished.
        """
        self.start_stop_btn.config(state="disabled")
        self.status_var.set("Preparing models (first launch only)…")
        threading.Thread(target=self._prefetch_models_worker, daemon=True).start()

    def _prefetch_models_worker(self):
        """Runs on a background thread; only touches Tk state via `root.after`."""
        error = None
        try:
            from silero_vad import load_silero_vad
            load_silero_vad()

            from denoiser import pretrained
            pretrained.dns64()
        except Exception as e:
            error = str(e)

        self.root.after(0, self._on_models_prefetched, error)

    def _on_models_prefetched(self, error):
        """Re-enable Start once prefetch finishes; runs on the Tk main thread."""
        self.start_stop_btn.config(state="normal")
        if error is not None:
            self.status_var.set(f"Idle — press Start to begin (model prefetch failed: {error})")
        else:
            self.status_var.set("Idle — press Start to begin")

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

    def _on_input_device_change(self):
        """React to the mic combo changing: refresh warnings and point the idle level monitor at the new device."""
        self._update_warnings()
        # Captured now, on the Tk main thread, since _restart_input_monitor
        # runs in the background (see its docstring) and can't safely read
        # the combo box itself.
        device_in = self._selected_index(self.in_combo, self.in_devices)
        threading.Thread(target=self._restart_input_monitor, args=(device_in,), daemon=True).start()

    def _on_output_device_change(self):
        """React to the speaker combo changing: refresh warnings (the latency slider's default doesn't depend on the device)."""
        self._update_warnings()

    # ── Output latency ──────────────────────────────────────────────────────

    def _reset_output_latency_default(self):
        """Reset the latency slider to its fixed default (not device-dependent - see OUTPUT_LATENCY_DEFAULT_MS)."""
        self.output_latency_var.set(OUTPUT_LATENCY_DEFAULT_MS)
        self._on_output_latency_change(OUTPUT_LATENCY_DEFAULT_MS)

    def _on_output_latency_change(self, value):
        """Update the little "XX ms" label next to the latency slider as it moves."""
        self.output_latency_value_label.config(text=f"{float(value):.0f} ms")

    # ── Level meters ──────────────────────────────────────────────────────

    def _push_level(self, which, var, block):
        """
        Update ``var`` (an ``in``/``out`` level bar, 0-100) from one audio
        block. Runs on PortAudio's callback thread, so the actual widget
        update is marshalled onto the Tk main thread via ``after``; updates
        are throttled to ``LEVEL_METER_MIN_UPDATE_INTERVAL`` since callbacks
        fire far more often than the bar needs to redraw.
        """
        now = time.monotonic()
        if now - self._last_level_ts[which] < LEVEL_METER_MIN_UPDATE_INTERVAL:
            return
        self._last_level_ts[which] = now

        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        db = 20.0 * np.log10(rms + 1e-9)
        lo, hi = LEVEL_METER_DB_RANGE
        level = float(np.clip(np.interp(db, [lo, hi], [0.0, 100.0]), 0.0, 100.0))
        try:
            self.root.after(0, var.set, level)
        except RuntimeError:
            # A stream can fire one last callback while it's being torn down
            # (e.g. right as Start/Ping closes the idle monitor) - by then the
            # window may already be gone; the level update itself is moot.
            pass

    def _start_input_monitor(self, device_in=None):
        """
        Open a lightweight input-only stream on the selected mic so its level
        bar reacts even while idle. Pass ``device_in`` explicitly to call
        this safely from a background thread (it avoids reading the combo
        box, which is only safe from the Tk main thread); omitted, it reads
        the current selection itself (main-thread callers only).
        """
        if self._input_monitor is not None:
            return
        if device_in is None:
            device_in = self._selected_index(self.in_combo, self.in_devices)
        if device_in is None:
            return
        try:
            self._input_monitor = sd.InputStream(
                channels=1, device=device_in,
                callback=lambda indata, frames, time_info, status: self._push_level("in", self.in_level_var, indata),
            )
            self._input_monitor.start()
        except Exception:
            # Best-effort preview only - if the device can't be opened here
            # (e.g. it's busy elsewhere), just leave the bar flat at 0.
            self._input_monitor = None
            self.root.after(0, self.in_level_var.set, 0)

    def _stop_input_monitor(self):
        """Close the idle mic-level stream, e.g. before the real concealer stream opens the same device. Safe to call from any thread."""
        if self._input_monitor is not None:
            try:
                self._input_monitor.stop()
                self._input_monitor.close()
            except Exception:
                pass
            self._input_monitor = None
        self.root.after(0, self.in_level_var.set, 0)

    def _restart_input_monitor(self, device_in=None):
        """
        Stop and reopen the idle mic-level monitor. Real device I/O, with
        the same "can block indefinitely on some devices" risk as Ping (see
        _ping_worker) - always run this on a background thread, never the
        Tk main thread.
        """
        self._stop_input_monitor()
        self._start_input_monitor(device_in)

    # ── Ping (speaker test tone) ─────────────────────────────────────────

    def _on_ping(self):
        """Play a short fade-in/fade-out sine through the selected speaker, animating its level bar."""
        if self._ping_stream is not None or self._ping_busy:
            return  # already playing, or a previous attempt is still stuck opening a device
        device_out = self._selected_index(self.out_combo, self.out_devices)
        if device_out is None:
            messagebox.showerror("Granular Speech Masker", "Select a speaker first.")
            return

        # Captured now, on the Tk main thread, so the background work below
        # never needs to touch the combo box (or the latency slider) itself
        # to know which mic to release/reopen, or what latency to request.
        device_in = self._selected_index(self.in_combo, self.in_devices)
        output_latency_ms = self.output_latency_var.get()

        self._ping_busy = True
        self.ping_btn.config(state="disabled")
        threading.Thread(
            target=self._ping_worker, args=(device_out, device_in, output_latency_ms), daemon=True
        ).start()

    def _ping_worker(self, device_out, device_in, output_latency_ms):
        """
        Runs entirely on its own background thread, never the Tk main
        thread: opening (or closing) an audio stream can block indefinitely
        on some devices - confirmed in practice with a raw ALSA hardware
        endpoint already claimed by the system's sound server, where
        sd.OutputStream(...).start() simply never returns. Doing this here
        instead of on the main thread means a single bad device choice can
        only get this one Ping attempt stuck - the rest of the app,
        including trying Ping again with a different device, keeps working,
        instead of the whole window freezing.
        """
        # Avoid the same simultaneous-input+output risk _update_warnings
        # warns about, in case mic and speaker happen to be the same
        # device: release the idle mic monitor for the duration of the test
        # tone.
        self._stop_input_monitor()
        try:
            stream = self._build_ping_stream(device_out, device_in, output_latency_ms)
            stream.start()
        except Exception as e:
            self.root.after(0, self._on_ping_start_failed, str(e), device_in)
            return
        self.root.after(0, self._on_ping_started, stream)

    def _build_ping_stream(self, device_out, device_in, output_latency_ms):
        """Construct (but don't start) the fade-in/fade-out test-tone OutputStream; runs on the ping worker thread."""
        sr = int(sd.query_devices(device_out)["default_samplerate"])
        n_samples = int(sr * PING_DURATION_S)
        t = np.arange(n_samples) / sr
        tone = (PING_AMPLITUDE * np.sin(2 * np.pi * PING_FREQUENCY_HZ * t)).astype(np.float32)

        fade_n = max(1, int(sr * PING_FADE_S))
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        tone[:fade_n] *= ramp
        tone[-fade_n:] *= ramp[::-1]

        position = {"i": 0}

        def callback(outdata, frames, time_info, status):
            start = position["i"]
            end = min(start + frames, n_samples)
            chunk = tone[start:end]
            outdata[: len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0.0
            self._push_level("out", self.out_level_var, outdata[: len(chunk)] if len(chunk) else outdata)
            position["i"] = end
            if end >= n_samples:
                raise sd.CallbackStop()

        def on_finished():
            # finished_callback runs on PortAudio's own notification thread,
            # where closing this stream and reopening the idle mic monitor
            # (both real, possibly-blocking device I/O) aren't safe to do
            # directly - hand off to a fresh background thread instead.
            threading.Thread(
                target=self._ping_close_worker, args=(stream, device_in), daemon=True
            ).start()

        stream = sd.OutputStream(
            samplerate=sr, channels=1, device=device_out,
            callback=callback, finished_callback=on_finished,
            latency=output_latency_ms / 1000.0,
        )
        return stream

    def _ping_close_worker(self, stream, device_in):
        """Runs on its own background thread: closes the finished ping stream and reopens the idle mic monitor."""
        try:
            stream.close()
        except Exception:
            pass
        self._start_input_monitor(device_in)
        self.root.after(0, self._on_ping_done)

    def _on_ping_started(self, stream):
        """Runs on the Tk main thread once the ping stream has actually started playing."""
        self._ping_stream = stream
        self._ping_busy = False

    def _on_ping_start_failed(self, error, device_in):
        """Runs on the Tk main thread if the ping stream failed to start (a quick, non-hanging failure)."""
        self._ping_busy = False
        messagebox.showerror("Granular Speech Masker", f"Could not play test tone: {error}")
        self.ping_btn.config(state="normal")
        self._start_input_monitor(device_in)

    def _on_ping_done(self):
        """Runs on the Tk main thread once _ping_close_worker finishes."""
        self._ping_stream = None
        self.out_level_var.set(0)
        self.ping_btn.config(state="normal")

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

    def _write_configs(self, device_in, device_out, output_latency_ms):
        """Persist the currently selected devices, output latency, VAD type/threshold, and denoiser choice to the stream/concealer/vad configs."""
        stream_path = _writable_config_path("stream", "default.yaml")
        stream_config = load_yaml(stream_path) or load_config("stream", "default")
        stream_config["device_in"] = device_in
        stream_config["device_out"] = device_out
        stream_config["output_latency"] = output_latency_ms / 1000.0
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
        """Validate the selected devices, persist the current settings, build the Stream, and kick off opening the live input/output streams in the background."""
        if self._start_busy:
            return
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
            self._write_configs(device_in, device_out, self.output_latency_var.get())
            if self.save_recording_var.get():
                record_mode = "disk"
                record_outdir = self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR
                # Timestamped so successive recordings never silently
                # overwrite each other.
                record_outprefix = "session_" + time.strftime("%Y%m%d_%H%M%S")
            else:
                record_mode = "none"
                record_outdir = None
                record_outprefix = ""
            concealer_stream = Stream(
                "default_source", "default", "default", "default", is_stream=True,
                record_mode=record_mode, record_outdir=record_outdir, record_outprefix=record_outprefix,
            )
            concealer_stream.reset_state()
        except Exception as e:
            messagebox.showerror("Granular Speech Masker", f"Failed to start: {e}")
            return

        self._start_busy = True
        self.start_stop_btn.config(state="disabled", text="Starting…")
        threading.Thread(
            target=self._start_worker, args=(concealer_stream, device_in, device_out), daemon=True
        ).start()

    def _start_worker(self, concealer_stream, device_in, device_out):
        """
        Runs entirely on its own background thread, never the Tk main
        thread: opening an audio stream can block indefinitely on some
        devices (confirmed in practice - see the comment in _ping_worker
        for the reproduced case). Keeping every sd.*Stream(...) call off
        the main thread means a bad device choice can only get this one
        Start attempt stuck, instead of freezing the whole window.
        """
        # Release the idle mic-level preview stream so the real InputStream
        # below can open the same device, and cut off a still-playing Ping
        # test tone so it doesn't hold the speaker device open too.
        self._stop_input_monitor()
        if self._ping_stream is not None:
            try:
                self._ping_stream.abort()
                self._ping_stream.close()
            except Exception:
                pass
            self._ping_stream = None
            self.root.after(0, self.out_level_var.set, 0)

        stream_config = concealer_stream.stream_config

        def input_callback_with_meter(indata, frames, time_info, status):
            concealer_stream.input_callback(indata, frames, time_info, status)
            self._push_level("in", self.in_level_var, indata)

        def output_callback_with_meter(outdata, frames, time_info, status):
            concealer_stream.output_callback(outdata, frames, time_info, status)
            self._push_level("out", self.out_level_var, outdata)

        input_stream = None
        output_stream = None
        try:
            # Two independent streams instead of one combined duplex stream:
            # the input side keeps the algorithm's fixed ~50ms blocksize
            # (needed for VAD/feature extraction), while the output stream is
            # free to use whatever (typically much smaller) blocksize the
            # backend prefers, instead of both directions sharing one
            # buffering granularity. They're connected by a ring buffer (see
            # Stream.input_callback/output_callback in amods.stream).
            input_stream = sd.InputStream(
                samplerate=stream_config["sr"],
                blocksize=concealer_stream.buffer_size,
                dtype=stream_config["dtype"],
                channels=stream_config["channels_in"],
                device=stream_config["device_in"],
                callback=input_callback_with_meter,
                latency=stream_config.get("input_latency", "low"),
            )
            output_stream = sd.OutputStream(
                samplerate=stream_config["sr"],
                dtype=stream_config["dtype"],
                channels=stream_config["channels_out"],
                device=stream_config["device_out"],
                callback=output_callback_with_meter,
                latency=stream_config.get("output_latency", "high"),
            )
            input_stream.start()
            output_stream.start()
        except Exception as e:
            if input_stream is not None:
                try:
                    input_stream.close()
                except Exception:
                    pass
            if output_stream is not None:
                try:
                    output_stream.close()
                except Exception:
                    pass
            self.root.after(0, self._on_start_failed, str(e), device_in)
            return

        self.root.after(0, self._on_started, concealer_stream, input_stream, output_stream)

    def _on_started(self, concealer_stream, input_stream, output_stream):
        """Runs on the Tk main thread once the real input/output streams have actually started."""
        self.concealer_stream = concealer_stream
        self.input_stream = input_stream
        self.output_stream = output_stream
        self._start_busy = False
        self.start_stop_btn.config(text="Stop", state="normal")
        self._set_controls_enabled(False)
        self._refresh_status()

    def _on_start_failed(self, error, device_in):
        """Runs on the Tk main thread if opening the real streams failed outright (a quick, non-hanging failure)."""
        self._start_busy = False
        messagebox.showerror("Granular Speech Masker", f"Failed to start: {error}")
        self.start_stop_btn.config(text="Start", state="normal")
        self._start_input_monitor(device_in)

    def _stop(self):
        """Kick off stopping the live stream in the background (see _stop_worker) and show an immediate "Stopping…" state."""
        if self._refresh_job is not None:
            self.root.after_cancel(self._refresh_job)
            self._refresh_job = None

        self.start_stop_btn.config(state="disabled", text="Stopping…")
        self.status_var.set("Stopping…")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        """
        Runs on a background thread, not the Tk main thread: stream.stop()/
        close() block until any audio callback currently in flight returns,
        which can take a while if the VAD/denoiser momentarily fell behind
        real-time - doing that here instead of directly in _stop() keeps the
        window responsive (showing "Stopping…") instead of freezing solid
        until it's done.
        """
        error = None
        saved_paths = None
        try:
            self.input_stream.stop()
            self.input_stream.close()

            self.output_stream.stop()
            self.output_stream.close()

            # Already written incrementally as audio arrived (record_mode=
            # "disk") - closing the files here is fast regardless of how
            # long the session was. Returns None if "Save recording" wasn't
            # checked.
            saved_paths = self.concealer_stream.finalize_recording()
        except Exception as e:
            error = str(e)

        self.root.after(0, self._on_stopped, saved_paths, error)

    def _on_stopped(self, saved_paths, error):
        """Runs on the Tk main thread once _stop_worker finishes; only now are the stream handles released."""
        self.input_stream = None
        self.output_stream = None
        self.concealer_stream = None

        if error is not None:
            self.status_var.set(f"Stopped with an error: {error}")
        elif saved_paths is not None:
            self.status_var.set(f"Stopped — saved to {saved_paths['mix']}")
        else:
            self.status_var.set("Stopped — not saved")

        self.start_stop_btn.config(text="Start", state="normal")
        self.progress_var.set("—")
        self.latency_var.set("—")
        self.in_level_var.set(0)
        self.out_level_var.set(0)
        self._set_controls_enabled(True)
        self._start_input_monitor()

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
        self.ping_btn.config(state="normal" if enabled else "disabled")
        self.output_latency_scale.config(state=scale_state)
        self.save_check.config(state="normal" if enabled else "disabled")
        self.output_dir_entry.config(state="normal" if enabled else "disabled")

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
