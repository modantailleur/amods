import numpy as np


def select_forecaster_model(config, sr=None):
    """Build the forecaster named by ``config["forecaster_type"]``."""
    if config["forecaster_type"] == "identity":
        return IdentityFM()
    else:
        raise ValueError(f"Unknown forecaster model config name: {config['forecaster_type']}")


class IdentityFM:
    """Predicts that the next frame is identical to the current one."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return ``x`` unchanged."""
        return x
