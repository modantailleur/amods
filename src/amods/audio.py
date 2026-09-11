"""Small, dependency-free signal helpers shared across the pipeline."""
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


def soft_clip(x: np.ndarray, limit: float = 0.95) -> np.ndarray:
    """Hard-limit ``x`` to ``[-limit, limit]``."""
    return np.clip(x, -limit, limit)
