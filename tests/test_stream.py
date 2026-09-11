import numpy as np
import soundfile as sf

from amods.stream import Stream, run_file_mode

SR = 8000

CONCEALER_YAML = """\
concealer_type: granspeechmask
concealer_vad_config: none_concealer
memory_maxlen: 5
min_memory_to_conceal: 2
max_countdown_reuse: 1
max_concealer_distance_to_buffer: 1
concealer_duration: 0.1
concealing_min_timeout_ratio: 0.5
concealing_max_timeout_ratio: 0.5
fade_duration: 0.01
random_reverse: false
denoise: false
decision_win: 0.1
"""

STREAM_YAML = f"""\
conc_multiplier: 1.0
monitor_gain: 0.0
record_mic_gain: 1.0
max_level: 1.0
sr: {SR}
channels_in: 1
channels_out: 1
device_in: null
device_out: null
dtype: float32
buffer_duration: 0.05
"""


def _write_test_configs(root):
    (root / "configs" / "concealer").mkdir(parents=True)
    (root / "configs" / "concealer" / "test.yaml").write_text(CONCEALER_YAML)
    (root / "configs" / "stream").mkdir(parents=True)
    (root / "configs" / "stream" / "test.yaml").write_text(STREAM_YAML)


CONCEALER_DICT = {
    "concealer_type": "granspeechmask",
    "concealer_vad_config": {"vad_type": "none"},  # inline dict, not a name
    "memory_maxlen": 5,
    "min_memory_to_conceal": 2,
    "max_countdown_reuse": 1,
    "max_concealer_distance_to_buffer": 1,
    "concealer_duration": 0.1,
    "concealing_min_timeout_ratio": 0.5,
    "concealing_max_timeout_ratio": 0.5,
    "fade_duration": 0.01,
    "random_reverse": False,
    "denoise": False,
    "decision_win": 0.1,
}

STREAM_DICT = {
    "conc_multiplier": 1.0,
    "monitor_gain": 0.0,
    "record_mic_gain": 1.0,
    "max_level": 1.0,
    "sr": SR,
    "channels_in": 1,
    "channels_out": 1,
    "device_in": None,
    "device_out": None,
    "dtype": "float32",
    "buffer_duration": 0.05,
}


def _make_stream(is_stream=False):
    return Stream(
        source_vad_config_name="none_source",
        concealer_config_name="test",
        stream_config_name="test",
        forecaster_config_name="default",
        is_stream=is_stream,
    )


def test_stream_builds_with_identity_vad_and_forecaster(isolated_env):
    _write_test_configs(isolated_env)
    stream = _make_stream()
    assert stream.buffer_size == int(SR * 0.05)
    assert stream.stream_config["sr"] == SR


def test_callback_is_silent_before_memory_warms_up(isolated_env):
    _write_test_configs(isolated_env)
    stream = _make_stream()
    stream.reset_state()

    frames = stream.buffer_size
    indata = (0.3 * np.ones((frames, 1))).astype(np.float32)
    outdata = np.zeros((frames, 2), dtype=np.float32)
    stream.callback(indata, outdata, frames, None, None)

    # First chunk: memory is empty, so the concealer contributes nothing
    # and playback is just monitor_gain (0.0 in this config) times the mic.
    assert np.allclose(outdata, 0.0)


def test_run_file_mode_produces_output_files(isolated_env):
    _write_test_configs(isolated_env)

    t = np.arange(SR * 3) / SR
    tone = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    (isolated_env / "audios").mkdir()
    sf.write(isolated_env / "audios" / "tone.wav", tone, SR)

    outdir = isolated_env / "output"
    stream = run_file_mode(
        indir=str(isolated_env / "audios"),
        infile="tone.wav",
        outdir=str(outdir),
        outprefix="",
        concealerconfig="test",
        streamconfig="test",
        sourcevadconfig="none_source",
        forecasterconfig="default",
        debug=False,
    )

    for name in ("tone_original.wav", "tone_concealer.wav", "tone_mix.wav"):
        assert (outdir / name).is_file()

    original, sr_read = sf.read(outdir / "tone_original.wav")
    assert sr_read == SR
    assert len(original) == len(tone)

    # With ~3s of tone fed through and pending_voice_max_duration=2s, the
    # memory should have picked up at least one candidate.
    assert len(stream.concealer.memory) > 0


def test_reset_state_clears_recordings(isolated_env):
    _write_test_configs(isolated_env)
    stream = _make_stream()
    stream.rec_original.append(np.zeros(10, dtype=np.float32))
    stream.rec_concealer.append(np.zeros(10, dtype=np.float32))
    stream.rec_mix.append(np.zeros((10, 2), dtype=np.float32))

    stream.reset_state()

    assert stream.rec_original == []
    assert stream.rec_concealer == []
    assert stream.rec_mix == []


def test_stream_can_be_built_entirely_from_inline_dicts(isolated_env):
    # No ./configs/ directory at all - every config is passed as a dict,
    # including the concealer's own nested concealer_vad_config.
    stream = Stream(
        source_vad_config_name={"vad_type": "none"},
        concealer_config_name=dict(CONCEALER_DICT),
        stream_config_name=dict(STREAM_DICT),
        forecaster_config_name={"forecaster_type": "identity"},
        is_stream=False,
    )
    stream.reset_state()

    assert stream.buffer_size == int(SR * 0.05)
    assert stream.stream_config["sr"] == SR
    assert stream.concealer.memory_maxlen == 5


def test_stream_with_dict_configs_does_not_mutate_callers_dicts(isolated_env):
    concealer_config = dict(CONCEALER_DICT)
    stream_config = dict(STREAM_DICT)

    Stream(
        source_vad_config_name={"vad_type": "none"},
        concealer_config_name=concealer_config,
        stream_config_name=stream_config,
        forecaster_config_name={"forecaster_type": "identity"},
        is_stream=True,
    )

    # Stream.__init__ adds is_stream/freeze_learning/pending_conc_max_size to
    # its own copy of the concealer config - the caller's dict must be untouched.
    assert concealer_config == CONCEALER_DICT
    assert stream_config == STREAM_DICT


def test_concealer_config_dict_overrides_are_respected(isolated_env):
    config = dict(CONCEALER_DICT)
    config["memory_maxlen"] = 3

    stream = Stream(
        source_vad_config_name={"vad_type": "none"},
        concealer_config_name=config,
        stream_config_name=dict(STREAM_DICT),
        forecaster_config_name={"forecaster_type": "identity"},
        is_stream=False,
    )

    assert stream.concealer.memory_maxlen == 3
