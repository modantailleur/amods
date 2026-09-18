# AMODS — Adaptive Masking of Distracting Speech

A framework for real-time algorithms that make nearby speech unintelligible to listeners.

## Background

Intelligible background speech is one of the most disruptive sources of noise in shared and open-plan offices. It can increase stress and annoyance, disrupt social interactions, and reduce work performance. Staying focused around distracting conversations is a common concern when working in a shared space.

Two approaches are commonly used to address this problem. Sound-masking systems broadcast masking noise throughout the room, but because they do not target periods of speech activity, they need to operate at relatively high levels to be effective. Active noise cancellation (ANC), typically deployed on a per-worker basis through headphones, works well against steady and predictable sounds but is less effective against speech, whose spectral and temporal content varies rapidly.

AMODS explores a different approach: adapting the masking to speech activity, both by activating it when speech occurs and by tailoring it to the characteristics of the speech being masked. This approach is complementary to ANC on headphones. We call this general approach *speech concealing*, and the masking sound it produces a *speech concealer*.

AMODS is designed to support the development and comparison of different speech-concealing algorithms. See [Using your own concealment algorithm](#using-your-own-concealment-algorithm) to learn how to integrate a new algorithm.

The currently included concealment algorithm is **GranSpeechMask** ("granular speech
masking"). It continuously records short clips of the speaker's recent voice and,
once a sufficient number have accumulated, replays the clip whose spectral content
most closely matches the current speech. Using the speaker's own voice as the masking
signal keeps it spectrally close to the speech being covered, allowing it to operate
without reaching a distracting level.

## Install

AMODS requires **Python 3.10, 3.11, or 3.12**.

```bash
conda create -n amods python=3.11
conda activate amods
pip install amods
```

On Linux, two system libraries are also required (on macOS and Windows, the
equivalent binaries are already bundled inside the Python packages): `libportaudio2` and `libc++1`. On
Debian/Ubuntu:

```bash
sudo apt install libportaudio2 libc++1
```

If `ten_vad` fails to install, it can be
force-reinstalled with:

```bash
pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
```

To develop AMODS itself, for example to run the test suite, clone the
repository and install it in editable mode:

```bash
git clone https://github.com/modantailleur/amods.git
cd amods
pip install -e ".[dev]"
pytest
```


## Real-Time Concealing (GUI)

The desktop UI provides the primary interface for running the demonstration. It allows
selection of the microphone, speaker, VAD type and threshold, and denoiser, and displays
the current memory size:

```
amods-gui
```

Settings selected in the UI are saved to configuration files
that are also read by the `amods` CLI (see [Configuration](#configuration)).

The GUI needs Tkinter, which ships with Python on most platforms; on some minimal
Linux installs it needs a separate system package, e.g. `sudo apt install python3-tk`.

## Real-Time Concealing (CLI)

To conceal speech in real time from the command line, use headphones and run:

```
amods
```

By default, the CLI uses the package's bundled default configurations (see
[Configuration](#configuration)). To use a different configuration, specify it for any
of the components described in the [Configuration](#configuration) section:

```
amods --concealerconfig yourpath/yourconfig.yaml
```

The original stream, the concealer output, and the mix between original and concealer
will be saved to:

- `./output/stream_original.wav`
- `./output/stream_concealer.wav`
- `./output/stream_mix.wav`

## Concealing a Pre-Recorded Audio (Real-Time Emulation)

Concealing a pre-recorded audio file provides a way to test the concealer with a
specified audio input, although this mode has no direct application purpose.

To do so, run:

```
amods --indir mydir/to/audio/ --infile myaudio.wav
```

The original audio, the concealer output, and the mix between original and concealer
will be saved to:

- `./output/myaudio_original.wav`
- `./output/myaudio_concealer.wav`
- `./output/myaudio_mix.wav`

## Configuration

The configuration components are:

- `concealer` — the concealment algorithm to run and its parameters (e.g. which
  `concealer_type` is registered under `amods.models.concealer`, how much audio its
  memory holds, and denoising/timing settings). Select its configuration with
  `--concealerconfig`.
- `stream` — real-time audio I/O settings, including sample rate, channel counts,
  buffer duration, input/output devices, gains, and latency hints. Select its
  configuration with `--streamconfig`.
- `vad` — voice activity detection settings, including the VAD backend and threshold. Used
  in two places: as the "source" VAD, which determines when speech is present, and
  internally by a concealer algorithm to filter its candidate clips. Select the source
  VAD configuration with `--sourcevadconfig`; an algorithm's internal VAD is specified
  in that algorithm's configuration, for example with `concealer_vad_config` for
  `GranSpeechMask`.
- `forecaster` — short-term prediction of the next audio frame before it is passed to
  the concealer (currently only `identity`, which passes the frame through unchanged).
  Select its configuration with `--forecasterconfig`.

Each configuration option accepts either a configuration name or a path to a YAML
file. For example, `--concealerconfig default` refers to the `default` concealer
configuration, while `--concealerconfig path/to/config.yaml` refers to a specific file.

When a configuration name is provided, AMODS searches for the corresponding file in
the following order:

1. `./configs/<component>/<name>.yaml`, relative to the current directory. This is
  used when running from a repository checkout or another directory containing a
  `configs/` folder.
2. `~/.amods/configs/<component>/<name>.yaml`, where settings saved by the GUI are
  stored and can be used regardless of the current directory.
3. The package's bundled default configuration, which allows `amods` and `amods-gui`
  to run immediately after `pip install` without a local `configs/` directory.

If the value already contains a `/`, it is treated as a path and used as provided.

## Using amods as a library

### Real-time, on your own mic/speaker

The following function blocks until Ctrl+C is pressed and then saves the
three WAV files, equivalent to running `amods` without `--infile`:

```python
from amods.stream import run_stream_mode

run_stream_mode(
    outdir="./output",
    outprefix="session1",
    concealerconfig="default",
    streamconfig="default",
    sourcevadconfig="default_source",
    forecasterconfig="default",
)
```

The bundled default stream configuration uses `device_in`/`device_out: null`, which
selects the system's default microphone and speaker. To target specific devices,
override the stream configuration with a dictionary (see below) using the indices
reported by `amods-latency --list-devices`.

For finer control, such as specifying a stop condition or processing audio as it arrives,
drive `Stream` with `sounddevice` directly. This is the approach used internally by
`run_stream_mode`:

```python
import sounddevice as sd
from amods import Stream

stream = Stream("default_source", "default", "default", "default", is_stream=True)
stream.reset_state()
sc = stream.stream_config

with sd.InputStream(samplerate=sc["sr"], blocksize=stream.buffer_size, dtype=sc["dtype"],
                     channels=sc["channels_in"], device=sc["device_in"],
                     callback=stream.input_callback, latency=sc.get("input_latency", "low")), \
     sd.OutputStream(samplerate=sc["sr"], dtype=sc["dtype"], channels=sc["channels_out"],
                      device=sc["device_out"], callback=stream.output_callback,
                      latency=sc.get("output_latency", "high")):
    sd.sleep(10_000)  # run for 10s; swap in your own stop condition

# stream.rec_original / rec_concealer / rec_mix now hold the recorded chunks,
# as they do after run_stream_mode; concatenate and write with soundfile as needed.
```

### A pre-recorded audio file

The following function reads a WAV file, processes it through the same
per-chunk pipeline as real-time mode, and writes the three output WAVs, equivalent to
`amods --infile ...`:

```python
from amods.stream import run_file_mode

run_file_mode(
    indir="./audios/",
    infile="voice_sample.wav",
    outdir="./output",
    outprefix="",  # empty -> uses the input filename as the prefix
    concealerconfig="default",
    streamconfig="default",
    sourcevadconfig="default_source",
    forecasterconfig="default",
)
```

If the audio is already available as a NumPy array in memory (mono `float32`, at the
stream configuration's sample rate), rather than as a file on disk, pass it to
`Stream.callback` in fixed-size blocks. This is the approach used internally by
`run_file_mode`:

```python
import numpy as np
from amods import Stream

stream = Stream("default_source", "default", "default", "default", is_stream=False)
stream.reset_state()

y = my_audio_array.astype(np.float32)
n = len(y)
n_blocks = int(np.ceil(n / stream.buffer_size))
y = np.pad(y, (0, n_blocks * stream.buffer_size - n))

for b in range(n_blocks):
    block = y[b * stream.buffer_size : (b + 1) * stream.buffer_size]
    indata = block.reshape(-1, 1)
    outdata = np.zeros((stream.buffer_size, 2), dtype=np.float32)
    stream.callback(indata, outdata, stream.buffer_size, None, None)

concealed = np.concatenate(stream.rec_concealer, axis=0)[:n]  # concealer output only
mix = np.concatenate(stream.rec_mix, axis=0)[:n]              # original + concealer, mixed
```

### Customizing configs from code

Each `*_config_name` argument accepts either a name/path (resolved as described above)
**or an already-built dictionary**, which is used directly. This permits YAML to be
omitted entirely, or allows selected fields to be overridden in a named configuration:

```python
from amods.config import load_config

config = load_config("concealer", "default")
config["denoise"] = False          # tweak just this one field
stream = Stream(source_vad_config_name="default_source",
                 concealer_config_name=config,
                 stream_config_name="default",
                 forecaster_config_name="default",
                 is_stream=False)
```

The same applies one level down: a concealer configuration's `concealer_vad_config`
field can also be a name/path or an inline dictionary. If only one component, such as a
VAD, is required rather than the complete `Stream`, the lower-level functions
(`amods.models.vad.select_vad_model`, `amods.models.concealer.select_concealer_model`,
etc.) already take plain config dicts directly.

### Using your own concealment algorithm

`concealer_type` in a concealer config selects which registered algorithm runs (e.g.
`granspeechmask`). To add a custom algorithm, implement `amods.models.concealer.ConcealerModel`
and decorate it with `@register_concealer("your_type_name")` — see the module docstring
in `amods/models/concealer/base.py` for the exact contract. No changes to `Stream`, the
CLI, or the examples above are required; reference the `type_name` in a concealer
configuration.

## Testing mic/speaker devices

The `amods-latency` tool can be used to identify available audio devices and verify a
microphone/speaker pair before starting a real-time stream. First, list the available
devices and their indices:

```
amods-latency --list-devices
```

Then provide the input and output device indices to test a specific pair. The tool
verifies that the pair is functioning and reports the PortAudio-reported input/output
latency:

```
amods-latency --device-in 15 --device-out 11
```

## Fundings

This work is supported by the French National Research Agency (ANR) through the
[ReNAR project](https://anr.fr/Projet-ANR-23-CE33-0012) (ANR-23-CE33-0012),
*"Augmentation des Environnements Sonores pour la Réduction de la Gêne"*,
conducted at LS2N, LORIA, and IRCAM.

<p>
<img src="https://raw.githubusercontent.com/modantailleur/amods/master/assets/logos/anr.jpg" alt="ANR" height="60">
&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/modantailleur/amods/master/assets/logos/cnrs.png" alt="CNRS" height="60">
&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/modantailleur/amods/master/assets/logos/ls2n.jpeg" alt="LS2N" height="60">
&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/modantailleur/amods/master/assets/logos/loria.jpg" alt="LORIA" height="60">
&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/modantailleur/amods/master/assets/logos/ircam.jpg" alt="IRCAM" height="60">
</p>

## Authors

- Modan Tailleur
- Aine Drelingyte
- Clara Boukhemia
- Mathieu Lagrange
- Romain Serizel
- Nicolas Misdariis

## Changelog

### 0.1.5

- Documented the Linux-only system library requirements (`libportaudio2`,
  `libc++1`).
- The GUI now loads the Silero VAD model and downloads the FbDM denoiser
  weights in the background as soon as it launches, instead of waiting for
  the first Start press.
- Fixed `ten_vad`'s dependency floor: versions below `1.0.6.5` bundle a
  Linux-only compiled binary despite claiming to be platform-independent, so
  they fail to import at all on Windows/macOS. Raised the floor to `1.0.6.5`.
- Excluded `webrtcvad` on Windows: its C extension hard-codes a POSIX-only
  build flag and has never published a Windows wheel, so it can't actually
  be installed there. The `"webrtc"` VAD type is unavailable on Windows as a
  result — use `"silero"` or `"ten"` instead.
- CI now also runs on Windows and macOS, not just Linux (still across both
  the lowest and newest version each dependency range allows).

### 0.1.4

- Fixed `torch`'s dependency floor
- Fixed `librosa`'s dependency floor
- Added `onnxruntime` as an explicit dependency, as required by the `silero` VAD backend

### 0.1.3

- Replaced the exact version pins for Python and every dependency with
  ranges, since exact pins were too strict for most environments.
- Added a CI matrix (`.github/workflows/tests.yml`) that installs AMODS at both the lowest and the newest version each range allows, on Python 3.10, 3.11, and 3.12, and runs the test suite against every combination.
- `input_latency` / `output_latency` in the stream config now accept `null`, to use the host API's own default latency instead of a fixed buffer or the `'low'`/`'high'` presets. Documented in `configs/stream/default.yaml`, and `output_latency` now defaults to `null` instead of a fixed `0.15`.

### 0.1.2

- Added input/output level meters to the GUI
- Added a "Ping" button to the GUI to play a test tone.

### 0.1.1

- Pinned exact versions for Python (`3.11.15`) and every dependency.
- Removed `scipy` and `torchaudio` from the dependencies.
- Added new authors.
- Added a Funding section.

### 0.1.0

- Initial release.