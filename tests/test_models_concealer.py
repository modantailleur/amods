import numpy as np
import pytest

from amods.models.concealer import select_concealer_model


def _make_config(**overrides):
    config = {
        "concealer_type": "granspeechmask",
        "concealer_vad_config": "none_concealer",  # IdentityVAD: no heavy backend needed
        "memory_maxlen": 5,
        "min_memory_to_conceal": 2,
        "max_countdown_reuse": 1,
        "max_concealer_distance_to_buffer": 1,
        "concealer_duration": 0.2,
        "concealing_min_timeout_ratio": 0.5,
        "concealing_max_timeout_ratio": 0.5,
        "fade_duration": 0.01,
        "is_stream": False,
        "pending_conc_max_size": 3 * 8000,
        "freeze_learning": False,
        "denoise": False,
        "random_reverse": False,
        "distance_type": "cosine",
        "decision_win": 0.2,
    }
    config.update(overrides)
    return config


def _sine_chunk(n, sr, freq=220.0, amplitude=0.5, phase=0.0):
    t = (np.arange(n) + phase) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_select_concealer_model_unknown_type_raises(isolated_env):
    with pytest.raises(ValueError):
        select_concealer_model(8000, "", {"concealer_type": "not_a_real_type"})


def test_concealer_is_silent_until_memory_has_enough_candidates(isolated_env):
    sr = 8000
    cm = select_concealer_model(sr, "", _make_config())

    chunk = _sine_chunk(400, sr)
    concealer, voice_name = cm.get_concealer(chunk)

    assert voice_name == ""
    assert np.all(concealer == 0)

    status = cm.status()
    assert status["ready"] is False
    assert status["progress"] == 0
    assert status["target"] == cm.memory_maxlen
    assert status["threshold"] == cm.min_memory_to_conceal
    assert status["label"] == "Memory"


def test_concealer_starts_concealing_once_memory_fills(isolated_env):
    sr = 8000
    cm = select_concealer_model(sr, "", _make_config())

    # pending_voice_max_duration is fixed at 2s; feed enough non-silent
    # 50ms-ish chunks to cross that threshold and trigger memory building
    # (synchronous here since is_stream=False), then confirm concealing
    # eventually produces non-silent output.
    chunk_size = 400
    got_non_silent = False
    for i in range(80):
        chunk = _sine_chunk(chunk_size, sr, phase=i * chunk_size)
        concealer, _ = cm.get_concealer(chunk)
        if np.any(concealer != 0):
            got_non_silent = True
            break

    assert got_non_silent
    assert len(cm.memory) > 0
    assert len(cm.memory) <= cm.memory_maxlen


def test_refresh_clears_concealing_state(isolated_env):
    sr = 8000
    cm = select_concealer_model(sr, "", _make_config())
    cm.cur_concealing = True
    cm.concealing_countdown = 500

    cm.refresh(_sine_chunk(400, sr))

    assert cm.cur_concealing is False
    assert cm.concealing_countdown == 0


def test_save_and_load_memory_round_trip(tmp_path, isolated_env):
    sr = 8000
    cm = select_concealer_model(sr, "", _make_config())
    cm._feed_memory(_sine_chunk(16000, sr), voice_name="alice")
    assert len(cm.memory) > 0

    save_path = tmp_path / "memory.npz"
    cm.save_memory(save_path)

    cm2 = select_concealer_model(sr, "", _make_config())
    cm2.load_memory(save_path)

    assert len(cm2.memory) == len(cm.memory)
    for (c1, f1, _, n1), (c2, f2, _, n2) in zip(cm.memory, cm2.memory):
        assert np.allclose(c1, c2)
        assert np.allclose(f1, f2)
        assert n1 == n2 == "alice"


def test_copy_preserves_memory_but_resets_cooldowns(isolated_env):
    sr = 8000
    cm = select_concealer_model(sr, "", _make_config())
    cm._feed_memory(_sine_chunk(16000, sr))
    for i in range(len(cm.memory)):
        concealer, feat, _, name = cm.memory[i]
        cm.memory[i] = (concealer, feat, 3, name)

    cm_copy = cm.copy()

    assert len(cm_copy.memory) == len(cm.memory)
    assert all(count == 0 for _, _, count, _ in cm_copy.memory)
