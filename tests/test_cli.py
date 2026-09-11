import numpy as np
import soundfile as sf

from amods.cli import build_parser, main

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


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.infile is None
    assert args.concealerconfig == "default"
    assert args.debug is False


def test_main_file_mode_end_to_end(isolated_env):
    (isolated_env / "configs" / "concealer").mkdir(parents=True)
    (isolated_env / "configs" / "concealer" / "test.yaml").write_text(CONCEALER_YAML)
    (isolated_env / "configs" / "stream").mkdir(parents=True)
    (isolated_env / "configs" / "stream" / "test.yaml").write_text(STREAM_YAML)

    t = np.arange(SR * 2) / SR
    tone = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    (isolated_env / "audios").mkdir()
    sf.write(isolated_env / "audios" / "tone.wav", tone, SR)

    main([
        "--infile", "tone.wav",
        "--indir", str(isolated_env / "audios"),
        "--outdir", str(isolated_env / "output"),
        "--concealerconfig", "test",
        "--streamconfig", "test",
        "--sourcevadconfig", "none_source",
    ])

    assert (isolated_env / "output" / "tone_original.wav").is_file()
    assert (isolated_env / "output" / "tone_concealer.wav").is_file()
    assert (isolated_env / "output" / "tone_mix.wav").is_file()
