import pytest


def test_fb_dm_predict_returns_similar_length_audio(sine_wave):
    pytest.importorskip("torch")
    pytest.importorskip("denoiser")
    from amods.models.denoiser import FbDM

    sr = 16000
    denoiser = FbDM(sr=sr)
    x = sine_wave(sr=sr, duration=0.5, amplitude=0.5)
    y = denoiser.predict(x)

    # dns64 runs at its own sample rate internally and resamples back, so the
    # output length can be off by a sample or two from resampling rounding.
    assert abs(len(y) - len(x)) <= 4
