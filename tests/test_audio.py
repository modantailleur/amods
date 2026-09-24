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


def test_apply_fade_in_uses_raised_cosine_not_linear():
    # A linear and a raised-cosine ramp coincidentally agree at the very
    # ends (0, 1) and the midpoint (0.5), so check a point off-center: at
    # 1/4 of the way through, raised-cosine sits well below where a linear
    # ramp would (0.146 vs 0.25) - the whole point of the smoother curve
    # being zero-slope (flatter) right where it leaves silence.
    x = np.ones(300, dtype=np.float32)  # long enough that fade_size=100 isn't clamped to n//2
    y = apply_fade_in(x, 100)
    quarter_point = y[24]
    linear_value = 0.25
    raised_cosine_value = 0.5 * (1 - np.cos(np.pi * 0.25))
    assert not np.isclose(quarter_point, linear_value, atol=0.02)
    assert np.isclose(quarter_point, raised_cosine_value, atol=0.02)


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


def test_limit_peak_scales_down_when_over_limit_in_steady_state():
    # prev_gain already equal to the needed gain (steady state, envelope
    # has nothing to chase) isolates plain scaling from the attack/release
    # behavior: every sample should land on (or extremely close to) the
    # same ratio to the limit.
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    expected_scale = 0.95 / 2.0
    y, new_gain = limit_peak(x, limit=0.95, prev_gain=expected_scale)
    assert np.all(y <= 0.95)
    assert np.all(y >= -0.95)
    # scaled proportionally, not flattened: the peak lands on the limit,
    # and every sample keeps very nearly the same ratio to the others
    assert np.isclose(np.max(np.abs(y)), 0.95, atol=1e-3)
    assert np.allclose(y, x * expected_scale, atol=1e-2)
    assert np.isclose(new_gain, expected_scale, atol=1e-2)


def test_limit_peak_leaves_in_range_signal_unchanged():
    x = np.array([-0.5, 0.0, 0.5, 0.9], dtype=np.float32)
    y, new_gain = limit_peak(x, limit=0.95, prev_gain=1.0)
    assert np.array_equal(y, x)
    assert new_gain == 1.0


def test_limit_peak_recovers_gain_gradually_not_instantly():
    # A block that itself needs no limiting, but starts right after a
    # heavily-reduced previous block (prev_gain=0.1), should recover
    # toward 1.0 gradually (release-timed) rather than snapping straight
    # back on the very next sample - the exact scenario that used to
    # cause an audible click/pump right after a concealer clip ends.
    sr = 1000
    release_s = 0.05  # small, deliberately, so the test array can be short
    x = np.full(200, 0.5, dtype=np.float32)  # 200ms at this sr - several release constants
    y, new_gain = limit_peak(x, limit=0.95, prev_gain=0.1, sr=sr, attack_s=0.002, release_s=release_s)
    # first sample has only just started moving away from 0.1, not
    # snapped straight to whatever 0.5 alone would need (1.0)
    assert y[0] < 0.5 * 0.5
    # monotonically recovering, and essentially fully recovered by the end
    assert np.all(np.diff(y) >= -1e-9)
    # after 4 release time constants there's still a small (~2%) residual
    # left to go, by design of exponential decay - not fully converged,
    # just close
    assert np.isclose(y[-1], 0.5, atol=0.02)
    assert np.isclose(new_gain, 1.0, atol=0.02)


def test_limit_peak_reacts_fast_to_a_sudden_loud_sample():
    # The reverse case: a block starting quiet (prev_gain=1.0) then
    # jumping to a loud, sustained value partway through must clamp down
    # quickly (attack-timed, much faster than release) rather than
    # overshooting the limit for long.
    sr = 1000
    x = np.concatenate([np.full(20, 0.1), np.full(180, 5.0)]).astype(np.float32)
    y, new_gain = limit_peak(x, limit=0.95, prev_gain=1.0, sr=sr, attack_s=0.002, release_s=0.05)
    assert np.max(np.abs(y)) <= 0.95 + 1e-6
    assert new_gain < 1.0


def test_limit_peak_gain_continuity_across_consecutive_calls():
    # Feeding the returned new_gain back in as the next call's prev_gain
    # must make the two blocks join smoothly at the seam (block2 picks up
    # right where block1 left off, then only moves a single smoothing
    # step from there) rather than resetting from scratch - the exact
    # mechanism that keeps a concealer clip's end from clicking.
    # attack/release scaled to match these short (50-sample) test blocks -
    # the default time constants assume real ~48kHz, 2400-sample chunks.
    kwargs = dict(sr=1000, attack_s=0.002, release_s=0.05)
    block1 = np.full(50, 5.0, dtype=np.float32)  # loud: forces real reduction
    block2 = np.full(50, 0.1, dtype=np.float32)  # then quiet: needs none
    y1, gain_after_1 = limit_peak(block1, limit=0.95, prev_gain=1.0, **kwargs)
    y2, _ = limit_peak(block2, limit=0.95, prev_gain=gain_after_1, **kwargs)
    assert gain_after_1 < 1.0  # block1 genuinely needed reducing
    assert np.isclose(y1[-1] / block1[-1], gain_after_1)
    # the seam: block2's own gain[0] is only one smoothing step away from
    # gain_after_1 - not reset straight back to 1.0
    seam_gain = y2[0] / block2[0]
    assert abs(seam_gain - gain_after_1) < 0.05
    assert np.max(np.abs(y2)) <= 0.95 + 1e-6


def test_limit_peak_never_exceeds_limit_even_when_a_block_jumps_instantly():
    # A block that jumps to a huge, constant value with no lead-in at all
    # (unrealistic for this app - concealer clips always have their own
    # fade-in - but worth covering) can force the safety net to override
    # the ramp's very first sample, breaking perfect seam continuity in
    # this one pathological case. That's an acceptable, inherent trade-off
    # for a limiter with no lookahead: the hard "never exceed limit"
    # guarantee always wins over smoothness.
    block = np.full(50, 5.0, dtype=np.float32)
    y, new_gain = limit_peak(block, limit=0.95, prev_gain=1.0)
    assert np.max(np.abs(y)) <= 0.95 + 1e-6
    # the safety correction is applied uniformly across the whole ramp, so
    # it can end up more conservative than strictly necessary right at the
    # tail (a fine trade-off: never exceeding the limit matters more than
    # exact optimality) - but it must never end up permitting an overshoot
    assert new_gain * 5.0 <= 0.95 + 1e-6


def test_limit_peak_handles_all_zero_input():
    x = np.zeros(10, dtype=np.float32)
    y, new_gain = limit_peak(x, limit=0.95)
    assert np.array_equal(y, x)
    assert new_gain == 1.0
