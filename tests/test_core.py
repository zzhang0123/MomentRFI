"""Tests for IterativeSurfaceFitter's round-based algorithm.

Follows the boundary-validation methodology: exercise the bright/broad round
dispatch, extreme kernel geometries, and the input-hardening/edge boundaries.
"""
import numpy as np
import pytest

from MomentRFI import IterativeSurfaceFitter


def _fitter(**kw):
    kw.setdefault("degree_freq", 6)
    kw.setdefault("degree_time", 4)
    kw.setdefault("verbose", False)
    return IterativeSurfaceFitter(**kw)


# ---------------------------------------------------------------------------
# Round 0 (bright RFI) vs broad rounds
# ---------------------------------------------------------------------------

def test_round0_flags_bright_narrow_rfi(make_waterfall):
    wf = make_waterfall(seed=1)
    wf[30, 50] *= 50.0        # single bright pixel
    wf[80:83, 120] *= 20.0    # short bright streak
    mask = _fitter(sigma_threshold=4.0).fit(wf)
    assert mask[30, 50]
    assert mask[80:83, 120].all()


def test_round0_misses_faint_broad(make_waterfall):
    # A faint, broad patch (~2.4 sigma per pixel, below the 4 sigma threshold)
    # should mostly escape round 0.
    wf = make_waterfall(seed=2)
    wf[60:70, 90:110] *= 1.05
    mask = _fitter(sigma_threshold=4.0).fit(wf)
    caught = mask[60:70, 90:110].mean()
    assert caught < 0.15


def test_broad_round_recovers_faint_broad(make_waterfall):
    # The same faint broad patch IS caught once a box-kernel round is added
    # (sqrt(K) SNR boost lifts ~2.4 sigma/pixel well above threshold).
    wf = make_waterfall(seed=2)
    wf[60:70, 90:110] *= 1.05
    mask = _fitter(sigma_threshold=4.0, broad_sigma_threshold=4.0).fit(
        wf, kernels=(np.ones((3, 5)),)
    )
    caught = mask[60:70, 90:110].mean()
    assert caught > 0.6


def test_kernels_none_equals_empty(make_waterfall):
    wf = make_waterfall(seed=3)
    wf[10, 10] *= 40.0
    m_none = _fitter().fit(wf)
    m_empty = _fitter().fit(wf, kernels=())
    assert np.array_equal(m_none, m_empty)


# ---------------------------------------------------------------------------
# sqrt(K) behaviour of the broad rounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [3, 5])
def test_broad_sigma_scales_as_sqrtK(make_waterfall, k):
    # On RFI-free data, the convolved-residual sigma of a (k,k) box round is
    # ~noise_sigma / k.  The estimate is exposed via history.
    wf = make_waterfall(nt=200, nf=300, noise=0.02, seed=4)
    fitter = _fitter(sigma_threshold=6.0)  # high threshold: flag ~nothing
    fitter.fit(wf, kernels=(np.ones((k, k)),))
    sigma0 = fitter.noise_sigma
    sigma_c = fitter.history["broad_rounds"][0]["sigma_c"]
    ratio = sigma0 / sigma_c
    assert abs(ratio - k) < 0.25 * k  # tight enough to reject a sigma/sqrt(k) bug


# ---------------------------------------------------------------------------
# Kernel geometry boundary cases (box / diagonal / 1D / oversized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", [
    np.ones((3, 3)),
    np.ones((1, 7)),
    np.ones((7, 1)),
    np.eye(3),
])
def test_kernel_geometries_run_and_finite(make_waterfall, kernel):
    wf = make_waterfall(seed=5)
    fitter = _fitter(sigma_threshold=4.0)
    mask = fitter.fit(wf, kernels=(kernel,))
    assert mask.dtype == bool
    assert np.isfinite(fitter.surface).all()
    assert np.isfinite(fitter.residuals).all()
    assert np.isfinite(fitter.history["broad_rounds"][0]["sigma_c"])


def test_kernel_larger_than_feature_does_not_flag_everything(make_waterfall):
    wf = make_waterfall(seed=6)
    wf[50:53, 100:103] *= 1.1  # small broad feature
    fitter = _fitter(sigma_threshold=4.0)
    mask = fitter.fit(wf, kernels=(np.ones((1, 41)),))  # kernel >> feature
    assert mask.mean() < 0.5  # must not runaway-flag


def test_diagonal_kernel_on_frequency_ripple_few_false_flags(baseline):
    # A low-order frequency ripple is removed by round 0; a diagonal kernel on
    # the resulting clean residuals must not invent broad diagonal detections.
    nt, nf = 120, 200
    rng = np.random.default_rng(7)
    ripple = 1.0 + 0.05 * np.cos(3 * np.linspace(-1, 1, nf))[None, :]
    wf = baseline(nt, nf) * ripple * rng.normal(1.0, 0.02, (nt, nf))
    fitter = _fitter(sigma_threshold=4.0)
    fitter.fit(wf, kernels=(np.eye(5),))
    n_new = fitter.history["broad_rounds"][0]["n_new"]
    assert n_new / wf.size < 0.02


# ---------------------------------------------------------------------------
# Input hardening and masking boundaries
# ---------------------------------------------------------------------------

def test_input_hardening_zeros_nan_neg(make_waterfall):
    wf = make_waterfall(seed=8)
    wf[10, 10] = 0.0
    wf[11, 11] = np.nan
    wf[12, 12] = -3.0
    fitter = _fitter()
    mask = fitter.fit(wf)
    assert mask[10, 10] and mask[11, 11] and mask[12, 12]
    assert np.isfinite(fitter.surface).all()
    assert np.isfinite(fitter.residuals).all()
    assert np.isfinite(fitter.noise_sigma)


def test_fully_masked_footprint_no_nan(make_waterfall):
    wf = make_waterfall(seed=9)
    prior = np.zeros(wf.shape, dtype=bool)
    prior[40:50, 60:70] = True  # a solid block bigger than the kernel
    fitter = _fitter(sigma_threshold=4.0)
    mask = fitter.fit(wf, kernels=(np.ones((3, 3)),), prior_mask=prior)
    assert mask[40:50, 60:70].all()          # prior stays flagged
    assert np.isfinite(fitter.history["broad_rounds"][0]["sigma_c"])
    assert np.isfinite(fitter.residuals).all()


def test_prior_mask_shape_mismatch_raises(make_waterfall):
    wf = make_waterfall(seed=10)
    with pytest.raises(ValueError):
        _fitter().fit(wf, prior_mask=np.zeros((3, 3), dtype=bool))


def test_dilate_detections_toggle(make_waterfall):
    wf = make_waterfall(seed=2)
    wf[60:70, 90:110] *= 1.10
    kernels = (np.ones((3, 5)),)
    m_dil = _fitter(sigma_threshold=4.0).fit(wf, kernels=kernels)
    m_cen = _fitter(sigma_threshold=4.0, dilate_detections=False).fit(wf, kernels=kernels)
    assert m_dil.sum() > m_cen.sum()          # dilation strictly adds flags here
    assert np.all(m_cen <= m_dil)             # and only ever adds (superset)


# ---------------------------------------------------------------------------
# API: removed kwargs and kernel validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"phase1_degree": 5},
    {"phase2_degree_freq": 10},
    {"phase2_degree_time": 5},
    {"sigma_floor_factor": 1.0},
])
def test_removed_kwargs_raise_typeerror(kw):
    with pytest.raises(TypeError):
        IterativeSurfaceFitter(**kw)


def test_bad_noise_estimator_raises():
    with pytest.raises(ValueError):
        IterativeSurfaceFitter(noise_estimator="bogus")


def test_kernel_1d_ndim_raises(make_waterfall):
    wf = make_waterfall(seed=11)
    with pytest.raises(ValueError):
        _fitter().fit(wf, kernels=(np.ones(5),))  # 1D, must be (1,5)/(5,1)


def test_kernel_too_big_raises(make_waterfall):
    wf = make_waterfall(nt=20, nf=20, seed=12)
    with pytest.raises(ValueError):
        _fitter().fit(wf, kernels=(np.ones((25, 3)),))


def test_negative_kernel_warns(make_waterfall):
    wf = make_waterfall(seed=13)
    k = np.ones((3, 3))
    k[0, 0] = -1.0
    with pytest.warns(UserWarning):
        _fitter(sigma_threshold=4.0).fit(wf, kernels=(k,))


# ---------------------------------------------------------------------------
# Attributes / history structure and value pins
# ---------------------------------------------------------------------------

def test_attributes_and_history_structure(make_waterfall):
    wf = make_waterfall(seed=14)
    wf[5, 5] *= 30.0
    fitter = _fitter(sigma_threshold=4.0)
    fitter.fit(wf, kernels=(np.ones((3, 3)), np.eye(3)))
    assert fitter.mask.shape == wf.shape
    assert fitter.surface.shape == wf.shape
    assert fitter.residuals.shape == wf.shape
    assert set(fitter.history) == {"round0", "broad_rounds"}
    assert len(fitter.history["broad_rounds"]) == 2
    kinds = [b["kernel"]["kind"] for b in fitter.history["broad_rounds"]]
    assert kinds == ["box", "diagonal"]
    # round-0 iteration records carry the expected keys
    it0 = fitter.history["round0"]["iterations"][0]
    assert {"iteration", "sigma_used", "flag_fraction", "changed_fraction"} <= set(it0)


@pytest.mark.parametrize("df,dt", [(10, 5), (0, 3), (3, 0), (8, 8), (1, 1)])
def test_precompute_basis_bit_identical(make_waterfall, df, dt):
    # The cached-basis fast path must produce a BIT-IDENTICAL mask to the
    # MomentEmu fallback, including at extreme degrees. A 1e-16 FP drift near the
    # threshold flips ~tens of pixels, so exact equality is the contract.
    wf = make_waterfall(seed=30)
    wf[20, 40] *= 40.0
    wf[50:60, 80:100] *= 1.05
    kw = dict(sigma_threshold=4.0, degree_freq=df, degree_time=dt, verbose=False)
    kernels = (np.ones((3, 5)),)
    m_fast = IterativeSurfaceFitter(precompute_basis=True, **kw).fit(wf, kernels=kernels)
    m_ref = IterativeSurfaceFitter(precompute_basis=False, **kw).fit(wf, kernels=kernels)
    assert np.array_equal(m_fast, m_ref)


def test_max_basis_gb_fallback_bit_identical(make_waterfall):
    # A tiny max_basis_gb forces the constant-memory fallback even with
    # precompute_basis=True; the result must still be bit-identical.
    wf = make_waterfall(seed=31)
    wf[10, 10] *= 40.0
    kw = dict(sigma_threshold=4.0, degree_freq=6, degree_time=4, verbose=False)
    m_full = IterativeSurfaceFitter(precompute_basis=True, max_basis_gb=1e9, **kw).fit(wf)
    m_guard = IterativeSurfaceFitter(precompute_basis=True, max_basis_gb=1e-9, **kw).fit(wf)
    assert np.array_equal(m_full, m_guard)


def test_clean_data_flags_few(make_waterfall):
    # No injected RFI: round 0 should flag only a small fraction at 4 sigma.
    wf = make_waterfall(nt=150, nf=200, noise=0.02, seed=15)
    fitter = _fitter(sigma_threshold=4.0)
    mask = fitter.fit(wf)
    assert mask.mean() < 0.02
    assert fitter.noise_sigma > 0


# ---------------------------------------------------------------------------
# Coverage for optional code paths
# ---------------------------------------------------------------------------

def test_sigma_value_fixes_round0_sigma(make_waterfall):
    wf = make_waterfall(seed=16)
    fitter = _fitter(sigma_value=0.05)
    fitter.fit(wf)
    assert fitter.noise_sigma == 0.05


def test_sigma_value_with_kernels_still_detects_broad(make_waterfall):
    # Regression: broad rounds must re-estimate sigma on the convolved field even
    # when sigma_value is set — otherwise the threshold is ~sqrt(K) too high and
    # nothing broad would flag.
    wf = make_waterfall(seed=2)
    wf[60:70, 90:110] *= 1.05
    fitter = _fitter(sigma_threshold=4.0, sigma_value=0.0087)  # ~per-pixel log noise
    mask = fitter.fit(wf, kernels=(np.ones((3, 5)),))
    assert mask[60:70, 90:110].mean() > 0.5


def test_lower_tail_estimator_runs(make_waterfall):
    wf = make_waterfall(seed=17)
    wf[20, 40] *= 40.0
    fitter = _fitter(sigma_threshold=4.0, noise_estimator="lower_tail", lower_tail_fraction=0.2)
    mask = fitter.fit(wf)
    assert mask[20, 40]
    assert fitter.noise_sigma > 0


def test_noise_estimator_diff_runs_and_catches_bright(make_waterfall):
    wf = make_waterfall(seed=17)
    wf[20, 40] *= 40.0
    fitter = _fitter(sigma_threshold=4.0, noise_estimator="diff")
    mask = fitter.fit(wf)
    assert mask[20, 40]
    assert fitter.noise_sigma > 0


def test_diff_sigma_is_fit_independent(make_waterfall):
    # noise_sigma from the "diff" estimator equals diff_sigma of the log-waterfall
    # directly (it does not depend on the surface fit), and is held fixed across
    # round-0 iterations.
    from MomentRFI import diff_sigma
    wf = make_waterfall(nt=150, nf=200, noise=0.02, seed=5)
    wf[10, 10] *= 40.0
    fitter = _fitter(sigma_threshold=4.0, noise_estimator="diff")
    fitter.fit(wf)
    bad = ~np.isfinite(wf) | (wf <= 0)
    good = ~bad
    expected = diff_sigma(np.log10(np.where(bad, 1.0, wf)), good, axis=0)
    assert fitter.noise_sigma == expected
    used = [it["sigma_used"] for it in fitter.history["round0"]["iterations"]]
    assert all(s == expected for s in used)         # fixed across iterations


def test_diff_axis_freq(make_waterfall):
    wf = make_waterfall(seed=6)
    wf[20, 40] *= 40.0
    fitter = _fitter(sigma_threshold=4.0, noise_estimator="diff", diff_axis=1)
    mask = fitter.fit(wf)
    assert mask[20, 40] and fitter.noise_sigma > 0


def test_sigma_value_overrides_diff(make_waterfall):
    wf = make_waterfall(seed=7)
    fitter = _fitter(noise_estimator="diff", sigma_value=0.05)
    fitter.fit(wf)
    assert fitter.noise_sigma == 0.05               # sigma_value wins


def test_one_sided_clipping_catches_negative_dropout_via_final_pass(make_waterfall):
    # one_sided flags only positive residuals DURING convergence, so a deep
    # negative dropout is caught only by the final symmetric pass. If that pass
    # were removed, the dropout would escape — so this exercises that branch.
    wf = make_waterfall(seed=18)
    wf[25, 55] *= 50.0     # bright positive spike (caught during convergence)
    wf[40, 90] *= 0.02     # deep negative dropout (caught by final symmetric pass)
    fitter = _fitter(sigma_threshold=4.0, one_sided_clipping=True)
    mask = fitter.fit(wf)
    assert mask[25, 55]    # positive spike
    assert mask[40, 90]    # negative dropout -> requires the final symmetric pass


def test_force_flag_fallback_triggers_and_makes_progress(make_waterfall):
    # Grossly overestimated sigma (via sigma_value) makes normal clipping flag
    # almost nothing, so the fallback must fire to force progress. A low
    # sigma_threshold makes the Gaussian-expected count large enough that the
    # 'expected_n // 4' trigger can engage.
    wf = make_waterfall(nt=60, nf=80, seed=19)
    fitter = _fitter(sigma_threshold=2.0, sigma_value=5.0, force_flag_fallback=True)
    mask = fitter.fit(wf)
    iters = fitter.history["round0"]["iterations"]
    assert any(it["forced_flag"] for it in iters)   # the forced branch executed
    # It made real progress (flagged good pixels) and the safety abort bounded it.
    assert 0.0 < mask.mean() < 0.6


def test_safety_abort_leaves_none_state(make_waterfall):
    # A prior mask covering > (1 - min_good_fraction) of pixels aborts round 0
    # before any fit: surface/residuals None, noise_sigma None, broad rounds
    # skipped, but prior flags preserved in the returned mask.
    wf = make_waterfall(seed=20)
    prior = np.zeros(wf.shape, dtype=bool)
    prior[: int(0.7 * wf.shape[0]), :] = True   # 70% flagged > 1 - 0.5
    fitter = _fitter(sigma_threshold=4.0, min_good_fraction=0.5)
    mask = fitter.fit(wf, kernels=(np.ones((3, 3)),), prior_mask=prior)
    assert fitter.surface is None
    assert fitter.residuals is None
    assert fitter.noise_sigma is None
    assert fitter.history["broad_rounds"] == []
    assert mask[prior].all()                    # prior preserved


# ---------------------------------------------------------------------------
# Post-processing methods
# ---------------------------------------------------------------------------

def test_dilate_mask_dilates_along_axis(make_waterfall):
    wf = make_waterfall(seed=21)
    fitter = _fitter(sigma_threshold=4.0)
    fitter.fit(wf)
    fitter.mask[:] = False
    fitter.mask[10, 20] = True
    out = fitter.dilate_mask(kernel_size=3, axis=1)   # along frequency
    assert out[10, 19] and out[10, 20] and out[10, 21]
    assert not out[9, 20] and not out[11, 20]         # not along time


def test_dilate_mask_validates(make_waterfall):
    fitter = _fitter()
    with pytest.raises(RuntimeError):
        fitter.dilate_mask()                      # before fit()
    fitter.fit(make_waterfall(seed=22))
    with pytest.raises(ValueError):
        fitter.dilate_mask(axis=2)


def test_flag_by_fraction_flags_rows_and_cols(make_waterfall):
    wf = make_waterfall(seed=23)
    fitter = _fitter(sigma_threshold=4.0)
    fitter.fit(wf)
    fitter.mask[:] = False
    fitter.mask[:, 30] = True                     # column 30 is fully flagged
    out = fitter.flag_by_fraction(threshold=0.5, axis=1)  # freq columns
    assert out[:, 30].all()
    # a half-flagged column at exactly the threshold is flagged (>=)
    fitter.mask[:, 40] = False
    fitter.mask[: wf.shape[0] // 2, 40] = True
    out = fitter.flag_by_fraction(threshold=0.5, axis=1)
    assert out[:, 40].all()


def test_flag_by_fraction_validates():
    fitter = _fitter()
    with pytest.raises(RuntimeError):
        fitter.flag_by_fraction(0.5, axis=1)      # before fit()


# ---------------------------------------------------------------------------
# Extended input hardening + input validation
# ---------------------------------------------------------------------------

def test_input_hardening_inf_and_bad_column(make_waterfall):
    wf = make_waterfall(seed=24)
    wf[5, 5] = np.inf
    wf[6, 6] = -np.inf
    wf[:, 70] = 0.0                                # an entire bad frequency column
    fitter = _fitter(sigma_threshold=4.0)
    mask = fitter.fit(wf)
    assert mask[5, 5] and mask[6, 6]
    assert mask[:, 70].all()
    assert np.isfinite(fitter.surface).all()
    assert np.isfinite(fitter.residuals).all()
    assert np.isfinite(fitter.noise_sigma)


def test_fit_rejects_non_2d(make_waterfall):
    with pytest.raises(ValueError):
        _fitter().fit(np.ones(50))                # 1D
    with pytest.raises(ValueError):
        _fitter().fit(np.ones((3, 4, 5)))         # 3D


@pytest.mark.parametrize("kw", [
    {"degree_freq": -1},
    {"degree_time": -2},
    {"sigma_threshold": 0},
    {"broad_sigma_threshold": -1.0},
    {"min_good_fraction": 0},
    {"lower_tail_fraction": 1.0},
    {"max_iterations": 0},
])
def test_constructor_range_validation(kw):
    with pytest.raises(ValueError):
        IterativeSurfaceFitter(**kw)
