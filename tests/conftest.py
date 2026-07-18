"""Shared fixtures for the MomentRFI test suite.

Note: these tests import ``MomentRFI``, which imports ``MomentEmu``. If MomentEmu
is installed editable but its path is stale, run
``pip install -e /path/to/MomentEmu`` or set ``PYTHONPATH`` to its ``src`` dir.
"""
import numpy as np
import pytest


def smooth_baseline(nt, nf):
    """A low-order strictly-positive baseline (sky x gain) in linear units.

    Degree 2 in frequency, degree 1 in time, plus a cross term — fully
    representable by the default anisotropic polynomial surface.
    """
    t = np.linspace(-1.0, 1.0, nt)[:, None]
    f = np.linspace(-1.0, 1.0, nf)[None, :]
    return 10.0 ** (1.0 + 0.3 * f + 0.2 * f ** 2 + 0.1 * t - 0.05 * t * f)


@pytest.fixture
def make_waterfall():
    """Factory: ``make_waterfall(nt, nf, noise, seed)`` -> baseline * noise."""
    def _make(nt=120, nf=200, noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        return smooth_baseline(nt, nf) * rng.normal(1.0, noise, size=(nt, nf))
    return _make


@pytest.fixture
def baseline():
    """The smooth-baseline builder as a fixture (avoids importing conftest)."""
    return smooth_baseline
