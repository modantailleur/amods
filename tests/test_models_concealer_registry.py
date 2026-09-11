import numpy as np
import pytest

from amods.models.concealer import ConcealerModel, register_concealer, select_concealer_model


def test_concealer_model_is_abstract():
    with pytest.raises(TypeError):
        ConcealerModel()


def test_default_status_is_always_ready():
    class Minimal(ConcealerModel):
        def get_concealer(self, x):
            return np.zeros_like(x), ""

        def refresh(self, x):
            pass

        def copy(self):
            return self

    status = Minimal().status()
    assert status["ready"] is True
    assert status["progress"] is None
    assert status["target"] is None
    assert status["threshold"] is None


def test_registering_a_new_algorithm_makes_it_selectable(isolated_env):
    # This is the exact extension point a new concealer algorithm would use:
    # implement ConcealerModel, decorate with @register_concealer, and it
    # becomes selectable via config without touching select_concealer_model
    # or any other dispatcher code.
    @register_concealer("test_passthrough")
    class PassthroughCM(ConcealerModel):
        def __init__(self, sr, config_path, config):
            self.sr = sr
            self.calls = []

        def get_concealer(self, x):
            self.calls.append(("get_concealer", len(x)))
            return x.copy(), "passthrough"

        def refresh(self, x):
            self.calls.append(("refresh", len(x)))

        def copy(self):
            return self

    cm = select_concealer_model(16000, "", {"concealer_type": "test_passthrough"})
    assert isinstance(cm, PassthroughCM)

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    audio, label = cm.get_concealer(x)
    assert np.array_equal(audio, x)
    assert label == "passthrough"

    cm.refresh(x)
    assert cm.calls == [("get_concealer", 3), ("refresh", 3)]
