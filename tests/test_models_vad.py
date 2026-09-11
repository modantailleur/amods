import numpy as np
import pytest

from amods.models.vad import IdentityVAD, RmsVAD, select_vad_model


def test_identity_vad_always_true():
    vad = IdentityVAD()
    assert vad.predict(np.zeros(100, dtype=np.float32)) is True


def test_rms_vad_detects_loud_signal(sine_wave):
    vad = RmsVAD(db_threshold=-50.0)
    loud = sine_wave(amplitude=0.9)
    assert vad.predict(loud) is np.True_ or vad.predict(loud)


def test_rms_vad_rejects_silence(silence):
    vad = RmsVAD(db_threshold=-50.0)
    assert not vad.predict(silence())


def test_select_vad_model_rms():
    vad = select_vad_model({"vad_type": "rms", "db_threshold": -40.0})
    assert isinstance(vad, RmsVAD)


def test_select_vad_model_none():
    vad = select_vad_model({"vad_type": "none"})
    assert isinstance(vad, IdentityVAD)


def test_select_vad_model_unknown_raises():
    with pytest.raises(ValueError):
        select_vad_model({"vad_type": "not_a_real_vad"})


def test_webrtc_vad_smoke(sine_wave):
    pytest.importorskip("webrtcvad")
    from amods.models.vad import WebrtcVAD

    vad = WebrtcVAD(logit_threshold=0.5, aggressiveness=1, sr=16000)
    loud = sine_wave(sr=16000, amplitude=0.9)
    result = vad.predict(loud)
    assert isinstance(result, (bool, np.bool_))


def test_webrtc_vad_handles_empty_input():
    pytest.importorskip("webrtcvad")
    from amods.models.vad import WebrtcVAD

    vad = WebrtcVAD(sr=16000)
    assert vad.predict(np.zeros(0, dtype=np.float32)) is False


def test_silero_vad_smoke(sine_wave):
    pytest.importorskip("silero_vad")
    pytest.importorskip("torch")
    from amods.models.vad import SileroVAD

    vad = SileroVAD(logit_threshold=None, sr=16000)
    loud = sine_wave(sr=16000, duration=0.5, amplitude=0.9)
    result = vad.predict(loud)
    assert 0.0 <= float(result) <= 1.0


def test_ten_vad_smoke(sine_wave):
    pytest.importorskip("ten_vad")
    from amods.models.vad import TenVAD

    vad = TenVAD(logit_threshold=0.5, sr=16000)
    loud = sine_wave(sr=16000, duration=0.5, amplitude=0.9)
    result = vad.predict(loud)
    assert isinstance(result, (bool, np.bool_))
