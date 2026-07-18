"""Smoke tests for plotting (new history schema) and HDF5 io round-trip."""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # headless

from MomentRFI import IterativeSurfaceFitter, load_waterfall, validate_waterfall
from MomentRFI.plotting import plot_convergence, plot_summary


def test_plot_convergence_and_summary_smoke(make_waterfall):
    wf = make_waterfall(nt=60, nf=90, seed=3)
    wf[20, 40] *= 40.0
    fitter = IterativeSurfaceFitter(degree_freq=6, degree_time=4, verbose=False)
    fitter.fit(wf, kernels=(np.ones((3, 3)), np.eye(3)))
    axes = plot_convergence(fitter.history)
    assert len(axes) == 3
    fig = plot_summary(wf, fitter)
    assert fig is not None


def test_plot_convergence_handles_empty_history():
    # Safety-abort / no-iteration case: must not raise.
    empty = {"round0": {"sigma": None, "iterations": []}, "broad_rounds": []}
    axes = plot_convergence(empty)
    assert len(axes) == 3


def test_io_roundtrip(tmp_path, make_waterfall):
    import h5py
    wf = make_waterfall(nt=10, nf=12, seed=4)
    freqs = np.linspace(60.0, 80.0, 12)
    times = np.arange(10, dtype=float)
    path = tmp_path / "obs.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("sdr")
        g.create_dataset("sdr_waterfall", data=wf)
        g.create_dataset("sdr_freqs", data=freqs)
        g.create_dataset("sdr_times", data=times)

    wf2, f2, t2 = load_waterfall(str(path))
    assert np.allclose(wf2, wf)
    assert np.allclose(f2, freqs)
    assert np.allclose(t2, times)

    info = validate_waterfall(wf2)
    assert info["shape"] == wf.shape
    assert not info["has_negative"]


def test_validate_waterfall_flags_problems(make_waterfall):
    wf = make_waterfall(nt=8, nf=8, seed=5)
    wf[0, 0] = -1.0
    with pytest.raises(ValueError):
        validate_waterfall(wf)
