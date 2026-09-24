import numpy as np

from amods.audio import apply_fade, apply_fade_in, apply_fade_out, limit_peak


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


def test_limit_peak_scales_down_when_over_limit():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    y = limit_peak(x, limit=0.95)
    assert np.all(y <= 0.95)
    assert np.all(y >= -0.95)
    # scaled proportionally, not flattened: the peak lands exactly on the
    # limit, and every sample keeps the same ratio to the others as before
    assert np.isclose(np.max(np.abs(y)), 0.95)
    expected_scale = 0.95 / 2.0
    assert np.allclose(y, x * expected_scale)


def test_limit_peak_leaves_in_range_signal_unchanged():
    x = np.array([-0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    y = limit_peak(x, limit=0.95)
    assert np.array_equal(y, x)


def test_limit_peak_handles_all_zero_input():
    x = np.zeros(10, dtype=np.float32)
    y = limit_peak(x, limit=0.95)
    assert np.array_equal(y, x)
