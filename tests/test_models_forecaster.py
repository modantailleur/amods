import numpy as np
import pytest

from amods.models.forecaster import IdentityFM, select_forecaster_model


def test_identity_forecaster_returns_input_unchanged():
    fm = IdentityFM()
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert fm.predict(x) is x


def test_select_forecaster_model_identity():
    fm = select_forecaster_model({"forecaster_type": "identity"})
    assert isinstance(fm, IdentityFM)


def test_select_forecaster_model_unknown_raises():
    with pytest.raises(ValueError):
        select_forecaster_model({"forecaster_type": "not_a_real_type"})
