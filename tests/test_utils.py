"""Unit tests for the helpers in MomentRFI.utils."""
import numpy as np
import pytest

from MomentRFI.utils import (
    masked_normalized_convolve, dilate_to_footprint,
    mad_sigma, lower_tail_sigma, normalize_to_interval, build_coordinate_grid,
    smooth_mask,
)


def test_masked_convolve_preserves_constant():
    field = np.full((10, 12), 3.0)
    good = np.ones_like(field, dtype=bool)
    conv, weight = masked_normalized_convolve(field, good, np.ones((3, 3)))
    assert np.allclose(conv, 3.0)
    assert np.all(weight > 0)


def test_masked_convolve_ignores_flagged_pixel():
    # A bright, flagged pixel must not leak into its neighbours.
    field = np.zeros((7, 7))
    field[3, 3] = 1000.0
    good = np.ones_like(field, dtype=bool)
    good[3, 3] = False
    conv, _ = masked_normalized_convolve(field, good, np.ones((3, 3)))
    assert conv[2, 2] == 0.0
    assert conv[3, 3] == 0.0
    assert np.all(np.isfinite(conv))


@pytest.mark.parametrize("k", [2, 3, 5])
def test_masked_convolve_sqrtK_noise_reduction(k):
    rng = np.random.default_rng(0)
    n = rng.standard_normal((500, 500))
    good = np.ones_like(n, dtype=bool)
    conv, _ = masked_normalized_convolve(n, good, np.ones((k, k)))
    ratio = n.std() / conv.std()
    assert abs(ratio - k) < 0.4 * k  # ~sqrt(k^2) = k


def test_masked_convolve_edge_unbiased():
    # Reflect mode keeps the local mean unbiased at the borders.
    field = np.full((8, 8), 5.0)
    good = np.ones_like(field, dtype=bool)
    conv, _ = masked_normalized_convolve(field, good, np.ones((3, 3)), mode="reflect")
    assert np.allclose(conv, 5.0)  # including corners/edges


def test_masked_convolve_fully_masked_footprint_no_nan():
    field = np.ones((5, 5))
    good = np.zeros_like(field, dtype=bool)  # nothing good anywhere
    conv, weight = masked_normalized_convolve(field, good, np.ones((3, 3)))
    assert np.all(np.isfinite(conv))
    assert np.allclose(weight, 0.0)


def test_masked_convolve_does_not_mutate_inputs():
    field = np.ones((5, 5)) * 2.0
    good = np.ones((5, 5), dtype=bool)
    field_copy, good_copy = field.copy(), good.copy()
    masked_normalized_convolve(field, good, np.ones((3, 3)))
    assert np.array_equal(field, field_copy)
    assert np.array_equal(good, good_copy)


def test_dilate_box():
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    d = dilate_to_footprint(m, np.ones((3, 3)))
    assert d[2:5, 2:5].all()
    assert d.sum() == 9


def test_dilate_diagonal():
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    d = dilate_to_footprint(m, np.eye(3))
    assert d.sum() == 3  # only the diagonal
    assert d[2, 2] and d[3, 3] and d[4, 4]


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_dilate_1d(shape):
    m = np.zeros((11, 11), dtype=bool)
    m[5, 5] = True
    d = dilate_to_footprint(m, np.ones(shape))
    assert d.sum() == 5  # spreads along one axis only


def test_dilate_asymmetric_kernel_direction():
    # Pin the exact resulting cells for an off-centre support so the dilation
    # direction is locked down (invisible under point-symmetric kernels).
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    kernel = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=float)  # centre + right
    d = dilate_to_footprint(m, kernel)
    assert d.sum() == 2
    assert d[3, 3] and d[3, 4]
    assert not d[3, 2]


# ---------------------------------------------------------------------------
# Noise estimators and coordinate helpers
# ---------------------------------------------------------------------------

def test_mad_sigma_recovers_std():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.3, size=200_000)
    assert abs(mad_sigma(x) - 0.3) < 0.01


def test_lower_tail_sigma_recovers_sigma():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.5, size=100_000)
    assert abs(lower_tail_sigma(x, 0.2) - 0.5) < 0.1


def test_lower_tail_sigma_fallback_on_degenerate_input():
    # A flat/degenerate tail makes the log-linear slope non-negative; the RMS
    # fallback must return a finite positive value rather than crash.
    x = np.concatenate([np.zeros(500), np.ones(500)])
    s = lower_tail_sigma(x, 0.2)
    assert np.isfinite(s) and s >= 0


def test_lower_tail_sigma_extreme_tail_fraction_no_crash():
    # tail_fraction at the boundaries must not raise (k is clamped).
    x = np.random.default_rng(2).normal(size=1000)
    for tf in (1e-6, 1.0):
        assert np.isfinite(lower_tail_sigma(x, tf))


def test_normalize_to_interval_maps_endpoints():
    out = normalize_to_interval(np.array([0.0, 5.0, 10.0]), -1.0, 1.0)
    assert out[0] == -1.0 and out[-1] == 1.0
    assert abs(out[1]) < 1e-12


def test_normalize_to_interval_constant_input():
    out = normalize_to_interval(np.full(4, 7.0), -1.0, 1.0)
    assert np.allclose(out, 0.0)   # midpoint for a constant array


def test_build_coordinate_grid_convention():
    # Column 0 = frequency (fast axis), column 1 = time (slow axis), both in [-1,1].
    coords = build_coordinate_grid(n_time=3, n_freq=4)
    assert coords.shape == (12, 2)
    assert coords[:, 0].min() == -1.0 and coords[:, 0].max() == 1.0
    assert coords[:, 1].min() == -1.0 and coords[:, 1].max() == 1.0
    # First 4 rows (time index 0) sweep all frequencies at the earliest time.
    assert np.allclose(coords[:4, 1], -1.0)          # time fixed
    assert coords[0, 0] == -1.0 and coords[3, 0] == 1.0  # freq varies fastest


def test_smooth_mask_majority_vote():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True                       # isolated single pixel
    out = smooth_mask(m, kernel_size=3)  # majority vote at 0.5 -> erodes it
    assert not out[2, 2]
