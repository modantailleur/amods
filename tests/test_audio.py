import numpy as np

from amods.audio import apply_fade, apply_fade_in, apply_fade_out, soft_clip


def test_apply_fade_in_ramps_from_zero():
    x = np.ones(100, dtype=np.float32)
    y = apply_fade_in(x, 10)
    assert y[0] == 0.0
    assert np.isclose(y[9], 1.0, atol=0.15)
    assert np.all(y[10:] == 1.0)


def test_apply_fade_out_ramps_to_zero():
    x = np.ones(100, dtype=np.float32)
    y = apply_fade_out(x, 10)
    assert y[-1] == 0.0
    assert np.all(y[:90] == 1.0)


def test_apply_fade_does_both_ends():
    x = np.ones(100, dtype=np.float32)
    y = apply_fade(x, 10)
    assert y[0] == 0.0
    assert y[-1] == 0.0
    assert np.all(y[10:90] == 1.0)


def test_apply_fade_does_not_mutate_input():
    x = np.ones(50, dtype=np.float32)
    x_copy = x.copy()
    apply_fade(x, 5)
    assert np.array_equal(x, x_copy)


def test_apply_fade_clamps_oversized_fade_to_half_length():
    # fade_size larger than half the signal should not crash or overlap-corrupt
    x = np.ones(10, dtype=np.float32)
    y = apply_fade(x, 1000)
    assert len(y) == 10
    assert y[0] == 0.0
    assert y[-1] == 0.0


def test_apply_fade_noop_on_short_or_zero_size():
    x = np.array([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(apply_fade(x, 0), x)
    assert np.array_equal(apply_fade_in(x, 0), x)
    assert np.array_equal(apply_fade_out(x, 0), x)


def test_soft_clip_limits_amplitude():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    y = soft_clip(x, limit=0.95)
    assert np.all(y <= 0.95)
    assert np.all(y >= -0.95)
    assert y[1] == -0.5
    assert y[3] == 0.5
