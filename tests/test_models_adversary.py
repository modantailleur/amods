import numpy as np

from amods.models.adversary import ReverseAM


def test_reverse_am_can_reverse(monkeypatch):
    am = ReverseAM()
    monkeypatch.setattr(np.random, "choice", lambda choices: -1)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = am.predict(x)
    assert np.array_equal(y, x[::-1])


def test_reverse_am_can_pass_through(monkeypatch):
    am = ReverseAM()
    monkeypatch.setattr(np.random, "choice", lambda choices: 1)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = am.predict(x)
    assert np.array_equal(y, x)


def test_reverse_am_does_not_mutate_input(monkeypatch):
    am = ReverseAM()
    monkeypatch.setattr(np.random, "choice", lambda choices: -1)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x_copy = x.copy()
    am.predict(x)
    assert np.array_equal(x, x_copy)
