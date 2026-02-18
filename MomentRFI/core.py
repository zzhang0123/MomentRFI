import numpy as np
from math import erfc, sqrt
from scipy.ndimage import convolve

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
    phase2_degree_freq : int or None
        Frequency-axis polynomial degree for Phase 2. Set to ``None``
        (together with ``phase2_degree_time``) to skip Phase 2 entirely
        and return the Phase 1 mask directly.
    phase2_degree_time : int or None
        Time-axis polynomial degree for Phase 2. Set to ``None``
        (together with ``phase2_degree_freq``) to skip Phase 2.
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
    sigma_value : float or None
        If provided, use this value directly as sigma in every iteration of
        both phases, bypassing the noise estimator and the Phase 2 sigma
        floor. Default ``None`` uses the normal data-driven estimation.
    force_flag_fallback : bool
        If True, when sigma is overestimated and fewer than 1/4 of the
        Gaussian-expected outliers are flagged, force-flag the top N
        pixels by |residual| (where N is the Gaussian expectation) to
        ensure progress.
    one_sided_clipping : bool
        If True, during convergence iterations only pixels with
        ``residual > +k·sigma`` (above the surface) are flagged. RFI always
        adds power, so below-surface pixels are clean noise and should not
        bias the surface fit. After the final converged phase, one additional
        symmetric pass ``|residual| > threshold`` is applied to also flag
        extreme low-noise statistical outliers. Default False (symmetric
        clipping throughout, current behaviour).
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
        sigma_value=None,
        force_flag_fallback=False,
        one_sided_clipping=False,
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
        self.sigma_value = sigma_value
        self.force_flag_fallback = force_flag_fallback
        self.one_sided_clipping = one_sided_clipping
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

    def fit(self, waterfall, prior_mask=None):
        """Run the two-phase RFI flagging algorithm.

        Parameters
        ----------
        waterfall : ndarray, shape (n_time, n_freq)
            Raw waterfall data (positive values, linear scale).
        prior_mask : ndarray of bool, shape (n_time, n_freq), optional
            A priori mask of known-bad pixels. True = flagged. These pixels
            are excluded from surface fitting in both phases and remain
            flagged in the returned mask.

        Returns
        -------
        mask : ndarray of bool, shape (n_time, n_freq)
            True where RFI is flagged. If ``phase2_degree_freq`` or
            ``phase2_degree_time`` is ``None``, Phase 2 is skipped and
            the Phase 1 mask is returned directly.
        """
        n_time, n_freq = waterfall.shape
        n_pixels = n_time * n_freq
        log_data = np.log10(waterfall).ravel()
        coords = build_coordinate_grid(n_time, n_freq)

        # Validate and flatten prior mask
        if prior_mask is not None:
            prior_mask = np.asarray(prior_mask, dtype=bool)
            if prior_mask.shape != (n_time, n_freq):
                raise ValueError(
                    f"prior_mask shape {prior_mask.shape} does not match waterfall shape {(n_time, n_freq)}"
                )
            prior_mask_flat = prior_mask.ravel()
        else:
            prior_mask_flat = np.zeros(n_pixels, dtype=bool)

        # ---- Phase 1: Sigma Calibration ----
        if self.verbose:
            print("=" * 60)
            print("Phase 1: Sigma Calibration (isotropic degree {})".format(self.phase1_degree))
            if self.sigma_value is not None:
                print(f"  Sigma: fixed={self.sigma_value} (noise estimator bypassed)")
            else:
                print(f"  Noise estimator: {self.noise_estimator}"
                      + (f" (tail={self.lower_tail_fraction})" if self.noise_estimator == "lower_tail" else ""))
            if self.one_sided_clipping:
                print("  Clipping: one-sided (positive residuals only during convergence)")
            print("=" * 60)

        multi_indices_p1 = generate_multi_indices(2, self.phase1_degree)
        if self.verbose:
            print(f"  Number of basis terms: {len(multi_indices_p1)}")

        mask_flat = prior_mask_flat.copy()
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
            sigma = (self.sigma_value if self.sigma_value is not None
                     else self._estimate_sigma(residuals_flat[good]))

            # Flag: one-sided or symmetric clipping
            if self.one_sided_clipping:
                new_mask = residuals_flat > self.sigma_threshold * sigma
            else:
                new_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma

            # Fallback: if sigma is overestimated, too few pixels get flagged
            # and iteration stalls.  For a Gaussian, we expect a known fraction
            # outside ±k*sigma.  If actual flags < 1/4 of that, force-flag the
            # top N by |residual| to ensure progress.
            forced = False
            if self.force_flag_fallback:
                if self.one_sided_clipping:
                    expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold) / 2)
                    actual_n = int(new_mask.sum())
                    if expected_n > 0 and actual_n < expected_n // 4:
                        threshold_val = np.partition(residuals_flat, -expected_n)[-expected_n]
                        new_mask = residuals_flat >= threshold_val
                        forced = True
                else:
                    expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold))
                    actual_n = int(new_mask.sum())
                    if expected_n > 0 and actual_n < expected_n // 4:
                        abs_res = np.abs(residuals_flat)
                        threshold_val = np.partition(abs_res, -expected_n)[-expected_n]
                        new_mask = abs_res >= threshold_val
                        forced = True

            new_mask |= prior_mask_flat

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

        # ---- Optional early exit: Phase 2 skipped ----
        if self.phase2_degree_freq is None or self.phase2_degree_time is None:
            if self.verbose:
                print("\n  Phase 2 skipped (phase2_degree_freq or phase2_degree_time is None).")
            if self.one_sided_clipping:
                # Surface is now finalised; apply one symmetric pass to also catch
                # extreme low-noise outliers on both sides.
                final_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma
                final_mask |= prior_mask_flat
                mask_flat = final_mask
                if self.verbose:
                    print("  [one_sided] Final symmetric pass applied.")
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

        # ---- Phase 2: Refined Fitting ----
        if self.verbose:
            print()
            print("=" * 60)
            print(f"Phase 2: Refined Fitting (degree freq={self.phase2_degree_freq}, time={self.phase2_degree_time})")
            if self.one_sided_clipping:
                print("  Clipping: one-sided (positive residuals only during convergence)")
            print("=" * 60)

        multi_indices_p2 = generate_multi_indices_with_degree_vec(
            [self.phase2_degree_freq, self.phase2_degree_time]
        )
        if self.verbose:
            print(f"  Number of basis terms: {len(multi_indices_p2)}")

        # Reset mask
        mask_flat = prior_mask_flat.copy()

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
            sigma_raw = (self.sigma_value if self.sigma_value is not None
                         else self._estimate_sigma(residuals_flat[good]))
            sigma = (self.sigma_value if self.sigma_value is not None
                     else max(sigma_raw, sigma_floor))

            # Flag: one-sided or symmetric clipping
            if self.one_sided_clipping:
                new_mask = residuals_flat > self.sigma_threshold * sigma
            else:
                new_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma

            # Fallback: force-flag if sigma is overestimated (see Phase 1)
            forced = False
            if self.force_flag_fallback:
                if self.one_sided_clipping:
                    expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold) / 2)
                    actual_n = int(new_mask.sum())
                    if expected_n > 0 and actual_n < expected_n // 4:
                        threshold_val = np.partition(residuals_flat, -expected_n)[-expected_n]
                        new_mask = residuals_flat >= threshold_val
                        forced = True
                else:
                    expected_n = int(n_good * _gaussian_expected_flag_fraction(self.sigma_threshold))
                    actual_n = int(new_mask.sum())
                    if expected_n > 0 and actual_n < expected_n // 4:
                        abs_res = np.abs(residuals_flat)
                        threshold_val = np.partition(abs_res, -expected_n)[-expected_n]
                        new_mask = abs_res >= threshold_val
                        forced = True

            new_mask |= prior_mask_flat

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

        if self.one_sided_clipping:
            # Surface is now finalised; apply one symmetric pass to also catch
            # extreme low-noise outliers on both sides.
            final_mask = np.abs(residuals_flat) > self.sigma_threshold * sigma
            final_mask |= prior_mask_flat
            mask_flat = final_mask
            if self.verbose:
                print("  [one_sided] Final symmetric pass applied.")

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
            print(f"Final: {total_flagged} data points flagged ({actual_frac:.4%})")
            print(f"  Gaussian expectation at {self.sigma_threshold:.1f}-sigma: {gauss_frac:.4%}")
            print(f"  Actual / expected: {ratio:.1f}x")

        return self.mask

    def smooth_mask_with_kernel(self, kernel_size=3, axis=1):
        """Dilate self.mask with a 1D kernel along a single axis.

        Any pixel whose 1D convolution result is > 0 — i.e. at least one
        flagged pixel falls within the kernel's reach along the chosen axis —
        is flagged. This is a 1D morphological dilation: flagged regions expand
        by ``(kernel_size - 1) // 2`` pixels in both directions along the axis.

        Parameters
        ----------
        kernel_size : int
            Length of the uniform (all-ones) 1D kernel. A larger value
            produces a wider dilation.
        axis : int
            0 to dilate along the time axis (each flagged pixel spreads to
            neighbouring time samples at the same frequency),
            1 to dilate along the frequency axis (each flagged pixel spreads
            to neighbouring channels at the same time).

        Returns
        -------
        ndarray of bool, shape (n_time, n_freq)
            Dilated mask (True = flagged). Also updates self.mask in place.
        """
        if self.mask is None:
            raise RuntimeError("No mask available. Run fit() first.")
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 (time) or 1 (frequency), got {axis!r}")
        # Build a 2D kernel that is 1D along the chosen axis
        if axis == 1:
            kernel = np.ones((1, kernel_size), dtype=float)
        else:
            kernel = np.ones((kernel_size, 1), dtype=float)
        smoothed = convolve(self.mask.astype(float), kernel, mode='constant', cval=0.0)
        self.mask = smoothed > 0
        return self.mask

    def flag_by_fraction(self, threshold, axis):
        """Flag entire rows or columns where flagged fraction exceeds a threshold.

        Parameters
        ----------
        threshold : float
            Fraction of flagged pixels in a row/column at or above which the
            entire row/column is flagged. E.g. 0.5 flags any row/column that
            is already more than half flagged.
        axis : int
            0 to flag time rows (each time sample checked independently),
            1 to flag frequency columns (each frequency channel checked).

        Returns
        -------
        ndarray of bool, shape (n_time, n_freq)
            Updated mask (True = flagged). Also updates self.mask in place.
        """
        if self.mask is None:
            raise RuntimeError("No mask available. Run fit() first.")
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 (time rows) or 1 (freq columns), got {axis!r}")
        mask = self.mask.copy()
        fractions = mask.mean(axis=1 - axis)  # axis=0 -> mean over cols; axis=1 -> mean over rows
        over = fractions >= threshold
        if axis == 0:
            mask[over, :] = True
        else:
            mask[:, over] = True
        self.mask = mask
        return self.mask
