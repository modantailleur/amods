import numpy as np


class ReverseAM:
    """Randomly plays a concealer candidate forwards or reversed."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return ``x`` reversed with 50% probability, otherwise unchanged (always a copy)."""
        if np.random.choice([-1, 1]) == -1:
            return x[::-1].copy()
        return x.copy()
