import numpy as np
import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """
    Run a test from an empty temp directory with an empty $HOME, so config
    resolution (amods.config.load_config) can't accidentally pick up this
    machine's real ./configs/ checkout or ~/.amods/configs/ settings, and
    instead exercises the packaged default configs.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path


@pytest.fixture
def sine_wave():
    def _make(freq=440.0, sr=16000, duration=1.0, amplitude=0.5):
        t = np.arange(int(sr * duration)) / sr
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return _make


@pytest.fixture
def silence():
    def _make(sr=16000, duration=1.0):
        return np.zeros(int(sr * duration), dtype=np.float32)
    return _make
