"""Small, dependency-free signal helpers shared across the pipeline."""
import math

import numpy as np


def apply_fade(x: np.ndarray, fade_size: int) -> np.ndarray:
    """Apply a linear fade-in and fade-out to both ends of ``x``."""
    n = len(x)
    if fade_size <= 0 or n < 2:
        return x
    fade_size = min(fade_size, n // 2)
    if fade_size <= 0:
        return x
    y = x.copy()
    fade_in = np.linspace(0.0, 1.0, fade_size, dtype=y.dtype)
    fade_out = np.linspace(1.0, 0.0, fade_size, dtype=y.dtype)
    y[:fade_size] *= fade_in
    y[-fade_size:] *= fade_out
    return y


def apply_fade_in(x: np.ndarray, fade_size: int) -> np.ndarray:
    """Apply a linear fade-in to the start of ``x``."""
    n = len(x)
    if fade_size <= 0 or n < 2:
        return x
    fade_size = min(fade_size, n // 2)
    if fade_size <= 0:
        return x
    y = x.copy()
    fade_in = np.linspace(0.0, 1.0, fade_size, dtype=y.dtype)
    y[:fade_size] *= fade_in
    return y


def apply_fade_out(x: np.ndarray, fade_size: int) -> np.ndarray:
    """Apply a linear fade-out to the end of ``x``."""
    n = len(x)
    if fade_size <= 0 or n < 2:
        return x
    fade_size = min(fade_size, n // 2)
    if fade_size <= 0:
        return x
    y = x.copy()
    fade_out = np.linspace(1.0, 0.0, fade_size, dtype=y.dtype)
    y[-fade_size:] *= fade_out
    return y


def limit_peak(x: np.ndarray, limit: float = 0.95, prev_gain: float = 1.0,
               sr: int = 48000, attack_s: float = 0.002, release_s: float = 0.15):
    """
    Scale ``x`` down so its peak absolute value never exceeds ``limit``,
    using a real attack/release envelope, like a hardware/plugin limiter,
    rather than one flat scale factor per call. Unlike hard-clipping
    (``np.clip``), this preserves the waveform's shape - it scales rather
    than flattens - so it never actually saturates/distorts, only ever
    gets quieter when it would have.

    Per sample, the gain needed to keep that one sample within ``limit``
    is computed, then the *applied* gain chases it: quickly (``attack_s``)
    when more reduction is suddenly needed, slowly (``release_s``) when
    recovering back toward unity. A single scale factor - or even a
    single linear ramp - per call, recomputed from scratch each time, can
    still swing wildly between one call and the next whenever the signal's
    own peak does (e.g. a burst of loud content starting or ending) -
    audible as a click or pumping. An attack/release envelope is what
    actually prevents that, by carrying its own state continuously across
    calls instead of resetting every time.

    Returns ``(y, new_gain)``; pass ``new_gain`` back in as ``prev_gain``
    on the next call over the same signal to keep the envelope continuous
    across an entire stream (a fresh ``prev_gain=1.0`` starts it cleanly).
    ``sr`` must match the signal's actual sample rate for the attack/
    release times to mean what they say.

    A peak occurring right at the very start of ``x``, before even a fast
    attack has had one sample to react, could in principle still slip
    past ``limit`` - so the result is re-checked and, if needed, scaled
    down once more, guaranteeing ``limit`` is never exceeded even then.
    """
    n = len(x)
    if n == 0:
        return x, prev_gain
    abs_x = np.abs(x.astype(np.float64))
    instant_gain = np.where(abs_x > limit, limit / np.maximum(abs_x, 1e-12), 1.0)
    attack_coeff = 1.0 - math.exp(-1.0 / (max(attack_s, 1e-6) * sr))
    release_coeff = 1.0 - math.exp(-1.0 / (max(release_s, 1e-6) * sr))

    gain = np.empty(n, dtype=np.float64)
    g = prev_gain
    for i in range(n):
        target = instant_gain[i]
        coeff = attack_coeff if target < g else release_coeff
        g += coeff * (target - g)
        gain[i] = g

    y = x.astype(np.float64) * gain
    y_peak = np.max(np.abs(y))
    if y_peak > limit:
        safety = limit / y_peak
        gain = gain * safety
        y = y * safety
    return y.astype(x.dtype), float(gain[-1])
