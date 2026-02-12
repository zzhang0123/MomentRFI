import numpy as np
from math import erfc, sqrt

from MomentEmu.PolyEmu import (
    generate_multi_indices,
    generate_multi_indices_with_degree_vec,
    compute_moments_vector_output_batched,
    evaluate_emulator_batched,
)
from MomentEmu.MomentEmu import solve_emulator_coefficients

from .utils import mad_sigma, lower_tail_sigma, build_coordinate_grid


def _gaussian_expected_flag_fraction(k):
    """Fraction of a Gaussian outside ±k sigma: erfc(k / sqrt(2))."""
    return erfc(k / sqrt(2))


def _fit_surface(coords, values, multi_indices, batch_size=200_000):
    """Fit polynomial surface via MomentEmu's batched moment method.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
    values : ndarray, shape (N,)
    multi_indices : ndarray
    batch_size : int

    Returns
    -------
    coeffs : ndarray, shape (D, 1)
    """
    Y = values.reshape(-1, 1)
    M, nu = compute_moments_vector_output_batched(
        coords, Y, multi_indices, batch_size=batch_size
    )
    return solve_emulator_coefficients(M, nu)


def _evaluate_surface(coords, coeffs, multi_indices, batch_size=200_000):
    """Evaluate polynomial surface via MomentEmu's batched evaluator.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
    coeffs : ndarray, shape (D, 1)
    multi_indices : ndarray
    batch_size : int

    Returns
    -------
    result : ndarray, shape (N,)
    """
    return evaluate_emulator_batched(
        coords, coeffs, multi_indices, batch_size=batch_size
    ).ravel()


# ---------------------------------------------------------------------------
# IterativeSurfaceFitter
# ---------------------------------------------------------------------------

class IterativeSurfaceFitter:
    """Two-phase iterative sigma-clipping RFI flagger using polynomial surface fitting.

    Phase 1 (Sigma Calibration):
        Fit isotropic low-degree polynomial, iterate to convergence.
        Extract stable sigma as noise floor.

    Phase 2 (Refined Fitting):
        Fit higher-degree anisotropic polynomial with sigma floor from Phase 1.
        Reset mask and re-flag from scratch.

    Parameters
    ----------
    sigma_threshold : float
        Symmetric clipping threshold in units of sigma.
    phase1_degree : int
        Isotropic polynomial degree for Phase 1.
    phase2_degree_freq : int
        Frequency-axis polynomial degree for Phase 2.
    phase2_degree_time : int
        Time-axis polynomial degree for Phase 2.
    sigma_floor_factor : float
        Multiplier on Phase 1 sigma to set the sigma floor for Phase 2.
    convergence_fraction : float
        Fraction of changed pixels below which iteration stops.
    min_good_fraction : float
        Safety abort if fraction of unflagged pixels drops below this.
    max_iterations : int
        Hard cap on iterations per phase.
    batch_size : int
        Batch size for monomial evaluation.
    noise_estimator : str
        ``"mad"`` (default) uses Median Absolute Deviation (robust up to
        ~50% contamination).  ``"lower_tail"`` fits sigma from the lower
        tail of the residuals, which stays valid even when >50% of pixels
        are RFI (since RFI adds power and only inflates the upper tail).
    lower_tail_fraction : float
        Quantile used by the ``"lower_tail"`` estimator (e.g. 0.2 means
        the 20th-percentile).  Smaller values are more conservative but
        noisier.
    force_flag_fallback : bool
        If True, when sigma is overestimated and fewer than 1/4 of the
        Gaussian-expected outliers are flagged, force-flag the top N
        pixels by |residual| (where N is the Gaussian expectation) to
        ensure progress.
    verbose : bool
        Print progress info.
    """

    def __init__(
        self,
        sigma_threshold=4.0,
        phase1_degree=5,
        phase2_degree_freq=10,
        phase2_degree_time=5,
        sigma_floor_factor=1.0,
        convergence_fraction=1e-5,
        min_good_fraction=0.5,
        max_iterations=15,
        batch_size=200_000,
        noise_estimator="mad",
        lower_tail_fraction=0.2,
        force_flag_fallback=False,
        verbose=True,
    ):
        if noise_estimator not in ("mad", "lower_tail"):
            raise ValueError(f"noise_estimator must be 'mad' or 'lower_tail', got '{noise_estimator}'")
        self.sigma_threshold = sigma_threshold
        self.phase1_degree = phase1_degree
        self.phase2_degree_freq = phase2_degree_freq
        self.phase2_degree_time = phase2_degree_time
        self.sigma_floor_factor = sigma_floor_factor
        self.convergence_fraction = convergence_fraction
        self.min_good_fraction = min_good_fraction
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.noise_estimator = noise_estimator
        self.lower_tail_fraction = lower_tail_fraction
        self.force_flag_fallback = force_flag_fallback
        self.verbose = verbose

        # Results (populated after fit)
        self.mask = None
        self.surface = None
        self.residuals = None
        self.sigma_floor = None
        self.history = {"phase1": [], "phase2": []}

    def _estimate_sigma(self, residuals):
        """Estimate sigma using the configured noise estimator."""
        if self.noise_estimator == "lower_tail":
            return lower_tail_sigma(residuals, self.lower_tail_fraction)
        return mad_sigma(residuals)

    def fit(self, waterfall):
        """Run the two-phase RFI flagging algorithm.

        Parameters
        ----------
        waterfall : ndarray, shape (n_time, n_freq)
            Raw waterfall data (positive values, linear scale).

        Returns
        -------
        mask : ndarray of bool, shape (n_time, n_freq)
            True where RFI is flagged.
        """
        n_time, n_freq = waterfall.shape
        n_pixels = n_time * n_freq
        log_data = np.log10(waterfall).ravel()
        coords = build_coordinate_grid(n_time, n_freq)

        # ---- Phase 1: Sigma Calibration ----
        if self.verbose:
            print("=" * 60)
            print("Phase 1: Sigma Calibration (isotropic degree {})".format(self.phase1_degree))
            print(f"  Noise estimator: {self.noise_estimator}"
                  + (f" (tail={self.lower_tail_fraction})" if self.noise_estimator == "lower_tail" else ""))
            print("=" * 60)

        multi_indices_p1 = generate_multi_indices(2, self.phase1_degree)
        if self.verbose:
            print(f"  Number of basis terms: {len(multi_indices_p1)}")

        mask_flat = np.zeros(n_pixels, dtype=bool)
        sigma_floor = None

        for iteration in range(1, self.max_iterations + 1):
            good = ~mask_flat
            n_good = good.sum()
            good_frac = n_good / n_pixels

            if good_frac < self.min_good_fraction:
                if self.verbose:
                    print(f"  [ABORT] Only {good_frac:.1%} unflagged — safety abort.")
                break

            coeffs = _fit_surface(
                coords[good], log_data[good], multi_indices_p1, self.batch_size
            )
            surface_flat = _evaluate_surface(coords, coeffs, multi_indices_p1, self.batch_size)
            residuals_flat = log_data - surface_flat

            # Compute sigma from unflagged pixels only
            sigma = self._estimate_sigma(residuals_flat[good])

            # Flag: symmetric clipping
            new_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma

            # Fallback: if sigma is overestimated, too few pixels get flagged
            # and iteration stalls.  For a Gaussian, we expect a known fraction
            # outside ±k*sigma.  If actual flags < 1/4 of that, force-flag the
            # top N by |residual| to ensure progress.
            forced = False
            if self.force_flag_fallback:
                expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold))
                actual_n = int(new_mask.sum())
                if expected_n > 0 and actual_n < expected_n // 4:
                    abs_res = np.abs(residuals_flat)
                    threshold_val = np.partition(abs_res, -expected_n)[-expected_n]
                    new_mask = abs_res >= threshold_val
                    forced = True

            changed = np.sum(new_mask != mask_flat)
            changed_frac = changed / n_pixels

            self.history["phase1"].append({
                "iteration": iteration,
                "sigma": sigma,
                "n_flagged": int(new_mask.sum()),
                "flag_fraction": new_mask.sum() / n_pixels,
                "changed_fraction": changed_frac,
                "forced_flag": forced,
            })

            if self.verbose:
                force_note = " [forced]" if forced else ""
                print(
                    f"  Iter {iteration:2d}: sigma={sigma:.6f}, "
                    f"flagged={new_mask.sum()} ({new_mask.sum()/n_pixels:.4%}), "
                    f"changed={changed} ({changed_frac:.6%}){force_note}"
                )

            mask_flat = new_mask

            if changed_frac < self.convergence_fraction:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}.")
                break

        sigma_floor = sigma * self.sigma_floor_factor
        self.sigma_floor = sigma_floor
        if self.verbose:
            print(f"  Phase 1 sigma floor: {sigma_floor:.6f}")

        # ---- Phase 2: Refined Fitting ----
        if self.verbose:
            print()
            print("=" * 60)
            print(f"Phase 2: Refined Fitting (degree freq={self.phase2_degree_freq}, time={self.phase2_degree_time})")
            print("=" * 60)

        multi_indices_p2 = generate_multi_indices_with_degree_vec(
            [self.phase2_degree_freq, self.phase2_degree_time]
        )
        if self.verbose:
            print(f"  Number of basis terms: {len(multi_indices_p2)}")

        # Reset mask
        mask_flat = np.zeros(n_pixels, dtype=bool)

        for iteration in range(1, self.max_iterations + 1):
            good = ~mask_flat
            n_good = good.sum()
            good_frac = n_good / n_pixels

            if good_frac < self.min_good_fraction:
                if self.verbose:
                    print(f"  [ABORT] Only {good_frac:.1%} unflagged — safety abort.")
                break

            coeffs = _fit_surface(
                coords[good], log_data[good], multi_indices_p2, self.batch_size
            )
            surface_flat = _evaluate_surface(coords, coeffs, multi_indices_p2, self.batch_size)
            residuals_flat = log_data - surface_flat

            # Compute sigma with floor enforcement
            sigma_raw = self._estimate_sigma(residuals_flat[good])
            sigma = max(sigma_raw, sigma_floor)

            # Flag: symmetric clipping
            new_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma

            # Fallback: force-flag if sigma is overestimated (see Phase 1)
            forced = False
            if self.force_flag_fallback:
                expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold))
                actual_n = int(new_mask.sum())
                if expected_n > 0 and actual_n < expected_n // 4:
                    abs_res = np.abs(residuals_flat)
                    threshold_val = np.partition(abs_res, -expected_n)[-expected_n]
                    new_mask = abs_res >= threshold_val
                    forced = True

            changed = np.sum(new_mask != mask_flat)
            changed_frac = changed / n_pixels

            self.history["phase2"].append({
                "iteration": iteration,
                "sigma_raw": sigma_raw,
                "sigma_used": sigma,
                "sigma_floor": sigma_floor,
                "n_flagged": int(new_mask.sum()),
                "flag_fraction": new_mask.sum() / n_pixels,
                "changed_fraction": changed_frac,
                "forced_flag": forced,
            })

            if self.verbose:
                floor_note = " [floor]" if sigma_raw < sigma_floor else ""
                force_note = " [forced]" if forced else ""
                print(
                    f"  Iter {iteration:2d}: sigma={sigma:.6f}{floor_note}, "
                    f"flagged={new_mask.sum()} ({new_mask.sum()/n_pixels:.4%}), "
                    f"changed={changed} ({changed_frac:.6%}){force_note}"
                )

            mask_flat = new_mask

            if changed_frac < self.convergence_fraction:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}.")
                break

        # Store results
        self.mask = mask_flat.reshape(n_time, n_freq)
        self.surface = (10.0 ** surface_flat).reshape(n_time, n_freq)
        self.residuals = residuals_flat.reshape(n_time, n_freq)

        if self.verbose:
            total_flagged = self.mask.sum()
            actual_frac = total_flagged / n_pixels
            gauss_frac = _gaussian_expected_flag_fraction(self.sigma_threshold)
            ratio = actual_frac / gauss_frac if gauss_frac > 0 else float("inf")
            print()
            print(f"Final: {total_flagged} pixels flagged ({actual_frac:.4%})")
            print(f"  Gaussian expectation at {self.sigma_threshold:.1f}-sigma: {gauss_frac:.4%}")
            print(f"  Actual / expected: {ratio:.1f}x")

        return self.mask
