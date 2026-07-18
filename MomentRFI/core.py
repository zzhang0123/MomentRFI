import warnings

import numpy as np
from math import erfc, sqrt
from scipy.ndimage import convolve

from MomentEmu.PolyEmu import generate_multi_indices_with_degree_vec

from .surface import (
    _fit_surface,
    _evaluate_surface,
    _build_phi,
    _fit_surface_from_phi,
    _evaluate_surface_from_phi,
)
from .utils import (
    mad_sigma,
    lower_tail_sigma,
    build_coordinate_grid,
    masked_normalized_convolve,
    dilate_to_footprint,
    _WEIGHT_FLOOR,
)


# Parameters removed in the round-based redesign, mapped to a migration hint.
_REMOVED_KWARGS = {
    "phase1_degree": "removed — round 0 now uses degree_freq/degree_time",
    "phase2_degree_freq": "renamed to degree_freq",
    "phase2_degree_time": "renamed to degree_time",
    "sigma_floor_factor": "removed — the two-phase sigma floor no longer exists",
}


def _gaussian_expected_flag_fraction(k):
    """Fraction of a Gaussian outside ±k sigma: erfc(k / sqrt(2))."""
    return erfc(k / sqrt(2))


def _describe_kernel(kernel):
    """Return a small dict describing a kernel's shape and geometry kind."""
    k = np.asarray(kernel)
    nt, nf = k.shape
    nz = np.count_nonzero(k)
    if nt == 1 and nf > 1:
        kind = "line-freq"
    elif nf == 1 and nt > 1:
        kind = "line-time"
    elif nt == nf and nz == nt and np.all(np.diag(k) != 0):
        kind = "diagonal"
    elif nz == k.size:
        kind = "box"
    else:
        kind = "custom"
    return {"shape": (int(nt), int(nf)), "kind": kind}


# ---------------------------------------------------------------------------
# IterativeSurfaceFitter
# ---------------------------------------------------------------------------

class IterativeSurfaceFitter:
    """Round-based iterative sigma-clipping RFI flagger via polynomial surfaces.

    The algorithm runs one surface-fitting round plus one cheap detection round
    per convolution kernel:

    Round 0 (surface fit):
        Fit an anisotropic polynomial surface to ``log10(waterfall)`` and
        iterate sigma-clipping to convergence. Catches **bright** RFI and
        produces the baseline surface and residuals.

    Rounds 1..N (per kernel, matched filter):
        For each kernel, convolve the round-0 residuals (mask-aware, excluding
        already-flagged pixels) and threshold the result. Broad RFI is spatially
        or spectrally continuous, so it adds ~linearly under the kernel while
        thermal noise adds in quadrature (~sqrt(K)); the broad-RFI SNR improves
        by ~sqrt(K), making faint broad RFI detectable. Detections are optionally
        dilated to the kernel footprint. No surface is refit — the single round-0
        baseline is reused, so a flexible polynomial can never "absorb" the broad
        RFI it is meant to find.

    Masks accumulate (union) across rounds; each round excludes already-flagged
    pixels from its statistics.

    Parameters
    ----------
    sigma_threshold : float
        Symmetric clipping threshold in units of sigma (round 0 and, unless
        overridden, the broad rounds).
    degree_freq : int
        Frequency-axis polynomial degree for the round-0 surface fit.
    degree_time : int
        Time-axis polynomial degree for the round-0 surface fit.
    broad_sigma_threshold : float or None
        Threshold (in sigma) for the broad-RFI kernel rounds. ``None`` (default)
        reuses ``sigma_threshold``.
    dilate_detections : bool
        If True (default), each broad-round detection is dilated to the kernel's
        footprint, so the full spatial extent (wings) of the broad RFI is flagged
        rather than only the footprint centre.
    convergence_fraction : float
        Fraction of changed pixels below which round-0 iteration stops.
    min_good_fraction : float
        Safety abort if the fraction of unflagged pixels drops below this.
    max_iterations : int
        Hard cap on round-0 iterations.
    batch_size : int
        Batch size for monomial evaluation.
    noise_estimator : str
        ``"mad"`` (default) uses Median Absolute Deviation (robust up to ~50%
        contamination). ``"lower_tail"`` fits sigma from the lower tail of the
        residuals, valid even when >50% of pixels are RFI.
    lower_tail_fraction : float
        Quantile used by the ``"lower_tail"`` estimator.
    sigma_value : float or None
        If provided, use this value directly as the round-0 sigma in every
        iteration, bypassing the noise estimator. Broad rounds always re-estimate
        their own sigma on the convolved field (whose noise level differs from the
        per-pixel value by the kernel's ~sqrt(K) factor).
    force_flag_fallback : bool
        Round-0 only: if sigma is overestimated and fewer than 1/4 of the
        Gaussian-expected outliers are flagged, force-flag the top-N pixels by
        |residual| to ensure progress. Deliberately NOT applied to the broad
        rounds, where convolution correlates neighbouring pixels and the Gaussian
        count would force-flag correlated noise blobs.
    one_sided_clipping : bool
        Round-0 only: if True, during convergence iterations only pixels with
        ``residual > +k·sigma`` are flagged, with one final symmetric pass. The
        broad rounds are always one-sided positive (broad RFI only adds power).
    precompute_basis : bool
        If True (default), build the full polynomial basis (Vandermonde) once and
        reuse it across all round-0 sigma-clip iterations instead of rebuilding it
        each iteration — a large speedup with bit-identical results. Costs
        ``N * D * 8`` bytes (``D = (degree_freq+1)*(degree_time+1)``). Set False to
        use MomentEmu's constant-memory path.
    max_basis_gb : float
        Memory cap (GB) for the precomputed basis. If the full Vandermonde would
        exceed this, ``fit`` automatically falls back to the constant-memory
        MomentEmu path (even with ``precompute_basis=True``), so huge waterfalls
        never OOM by default.
    verbose : bool
        Print progress info.

    Attributes (populated after ``fit``)
    ------------------------------------
    mask : ndarray of bool
        Accumulated RFI mask (True = flagged).
    surface : ndarray
        Round-0 fitted surface (linear scale).
    residuals : ndarray
        Round-0 log-space residuals (the field the kernels convolve).
    noise_sigma : float
        Final round-0 sigma (the estimated noise level).
    history : dict
        ``{"round0": {"sigma": ..., "iterations": [...]}, "broad_rounds": [...]}``.
    """

    def __init__(
        self,
        sigma_threshold=4.0,
        degree_freq=10,
        degree_time=5,
        broad_sigma_threshold=None,
        dilate_detections=True,
        convergence_fraction=1e-5,
        min_good_fraction=0.5,
        max_iterations=15,
        batch_size=200_000,
        noise_estimator="mad",
        lower_tail_fraction=0.2,
        sigma_value=None,
        force_flag_fallback=False,
        one_sided_clipping=False,
        precompute_basis=True,
        max_basis_gb=16.0,
        verbose=True,
        **removed_kwargs,
    ):
        if removed_kwargs:
            hints = "; ".join(
                f"'{k}' {_REMOVED_KWARGS.get(k, 'is not a valid parameter')}"
                for k in removed_kwargs
            )
            raise TypeError(f"IterativeSurfaceFitter: {hints}")
        if noise_estimator not in ("mad", "lower_tail"):
            raise ValueError(f"noise_estimator must be 'mad' or 'lower_tail', got '{noise_estimator}'")
        for name, val in (("degree_freq", degree_freq), ("degree_time", degree_time)):
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValueError(f"{name} must be a non-negative int, got {val!r}")
        if sigma_threshold <= 0:
            raise ValueError(f"sigma_threshold must be > 0, got {sigma_threshold}")
        if broad_sigma_threshold is not None and broad_sigma_threshold <= 0:
            raise ValueError(f"broad_sigma_threshold must be > 0 or None, got {broad_sigma_threshold}")
        if not 0 < min_good_fraction <= 1:
            raise ValueError(f"min_good_fraction must be in (0, 1], got {min_good_fraction}")
        if not 0 < lower_tail_fraction < 1:
            raise ValueError(f"lower_tail_fraction must be in (0, 1), got {lower_tail_fraction}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        self.sigma_threshold = sigma_threshold
        self.degree_freq = degree_freq
        self.degree_time = degree_time
        self.broad_sigma_threshold = broad_sigma_threshold
        self.dilate_detections = dilate_detections
        self.convergence_fraction = convergence_fraction
        self.min_good_fraction = min_good_fraction
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.noise_estimator = noise_estimator
        self.lower_tail_fraction = lower_tail_fraction
        self.sigma_value = sigma_value
        self.force_flag_fallback = force_flag_fallback
        self.one_sided_clipping = one_sided_clipping
        self.precompute_basis = precompute_basis
        self.max_basis_gb = max_basis_gb
        self.verbose = verbose

        # Results (populated after fit)
        self.mask = None
        self.surface = None
        self.residuals = None
        self.noise_sigma = None
        self.history = {"round0": None, "broad_rounds": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _broad_threshold(self):
        """Sigma threshold used by the broad-RFI rounds (resolved in one place)."""
        return (self.broad_sigma_threshold if self.broad_sigma_threshold is not None
                else self.sigma_threshold)

    def _estimate_sigma(self, residuals):
        """Estimate sigma using the configured noise estimator."""
        if self.noise_estimator == "lower_tail":
            return lower_tail_sigma(residuals, self.lower_tail_fraction)
        return mad_sigma(residuals)

    def _clip_mask(self, residuals_flat, sigma, good):
        """Flag pixels beyond the sigma threshold, with an optional fallback.

        ``good`` is the boolean array of currently-unflagged pixels. Returns
        ``(mask, forced)``. When ``force_flag_fallback`` is on and sigma is
        overestimated (fewer than 1/4 of the Gaussian-expected outliers flag among
        the good pixels), force-flag the top-N good pixels by residual magnitude to
        ensure progress. Candidate selection is restricted to good pixels so the
        fallback cannot re-select already-flagged RFI (whose residuals dominate the
        magnitude ranking) and stall.
        """
        n_good = int(good.sum())
        thr = self.sigma_threshold * sigma
        if self.one_sided_clipping:
            new_mask = residuals_flat > thr
        else:
            new_mask = np.abs(residuals_flat) > thr

        if not self.force_flag_fallback:
            return new_mask, False

        frac = _gaussian_expected_flag_fraction(self.sigma_threshold)
        if self.one_sided_clipping:
            expected_n = int(n_good * frac / 2)
            metric = residuals_flat
        else:
            expected_n = int(n_good * frac)
            metric = np.abs(residuals_flat)
        new_good = int((new_mask & good).sum())
        if expected_n > 0 and new_good < expected_n // 4:
            cand = metric[good]
            if cand.size >= expected_n:
                threshold_val = np.partition(cand, -expected_n)[-expected_n]
                return new_mask | (good & (metric >= threshold_val)), True
        return new_mask, False

    def _flatten_prior_mask(self, prior_mask, n_time, n_freq):
        """Validate and flatten a prior mask to a (N,) bool array (a copy)."""
        if prior_mask is None:
            return np.zeros(n_time * n_freq, dtype=bool)
        prior_mask = np.asarray(prior_mask, dtype=bool)
        if prior_mask.shape != (n_time, n_freq):
            raise ValueError(
                f"prior_mask shape {prior_mask.shape} does not match "
                f"waterfall shape {(n_time, n_freq)}"
            )
        return prior_mask.ravel().copy()

    def _validate_kernels(self, kernels, shape):
        """Coerce ``kernels`` to a tuple of validated 2D float arrays."""
        if kernels is None:
            return ()
        validated = []
        for i, k in enumerate(kernels):
            kf = np.asarray(k, dtype=float)
            if kf.ndim != 2:
                raise ValueError(
                    f"kernel {i} must be 2D (use shape (1, k) or (k, 1) for a 1D "
                    f"kernel), got ndim={kf.ndim}"
                )
            if kf.size == 0:
                raise ValueError(f"kernel {i} is empty")
            if kf.shape[0] > shape[0] or kf.shape[1] > shape[1]:
                raise ValueError(
                    f"kernel {i} shape {kf.shape} exceeds waterfall shape {shape}"
                )
            if np.any(kf < 0):
                warnings.warn(
                    f"kernel {i} has negative entries; the broad-RFI SNR argument "
                    f"assumes non-negative kernels",
                    stacklevel=3,
                )
            validated.append(kf)
        return tuple(validated)

    def _prepare_basis(self, coords, multi_indices, values_flat, n_pixels):
        """Precompute the full Vandermonde once, unless disabled or too large.

        ``coords`` are fixed for the whole fit, so building Phi once and reusing it
        across iterations avoids rebuilding the monomials each iteration (~5x). The
        O(N*D) memory is guarded by ``max_basis_gb``; over budget falls back to the
        MomentEmu constant-memory path. Returns ``(use_cache, Phi, Y)``.
        """
        if not self.precompute_basis:
            return False, None, None
        mi = [(int(a), int(b)) for a, b in multi_indices]
        est_gb = n_pixels * len(mi) * 8 / 1e9
        if est_gb > self.max_basis_gb:
            if self.verbose:
                print(f"  [precompute_basis] basis ~{est_gb:.1f} GB > max_basis_gb"
                      f"={self.max_basis_gb} — using MomentEmu constant-memory path.")
            return False, None, None
        Phi = _build_phi(coords, mi, max(a for a, _ in mi), max(b for _, b in mi))
        return True, Phi, values_flat.reshape(-1, 1)

    def _fit_round(self, values_flat, coords, multi_indices, prior_mask_flat, n_pixels):
        """Run one iterative sigma-clip surface fit.

        Returns ``(mask_flat, surface_flat, residuals_flat, final_sigma, records)``.
        ``surface_flat``/``residuals_flat`` are ``None`` if the round aborts on the
        safety threshold before any fit.
        """
        mask_flat = prior_mask_flat.copy()
        records = []
        sigma = None
        surface_flat = residuals_flat = None

        use_cache, Phi, Y = self._prepare_basis(coords, multi_indices, values_flat, n_pixels)

        for iteration in range(1, self.max_iterations + 1):
            good = ~mask_flat
            n_good = int(good.sum())
            if n_good / n_pixels < self.min_good_fraction:
                if self.verbose:
                    print(f"  [ABORT] Only {n_good/n_pixels:.1%} unflagged — safety abort.")
                break

            if use_cache:
                idx = np.flatnonzero(good)
                coeffs = _fit_surface_from_phi(Phi, idx, Y, self.batch_size)
                surface_flat = _evaluate_surface_from_phi(Phi, coeffs, self.batch_size)
            else:
                coeffs = _fit_surface(coords[good], values_flat[good], multi_indices, self.batch_size)
                surface_flat = _evaluate_surface(coords, coeffs, multi_indices, self.batch_size)
            residuals_flat = values_flat - surface_flat

            sigma = (self.sigma_value if self.sigma_value is not None
                     else self._estimate_sigma(residuals_flat[good]))

            new_mask, forced = self._clip_mask(residuals_flat, sigma, good)
            new_mask = new_mask | prior_mask_flat

            n_flagged = int(new_mask.sum())
            changed = int(np.sum(new_mask != mask_flat))
            changed_frac = changed / n_pixels

            records.append({
                "iteration": iteration,
                "sigma_used": sigma,
                "n_flagged": n_flagged,
                "flag_fraction": n_flagged / n_pixels,
                "changed_fraction": changed_frac,
                "forced_flag": forced,
            })

            if self.verbose:
                force_note = " [forced]" if forced else ""
                print(
                    f"  Iter {iteration:2d}: sigma={sigma:.6f}, "
                    f"flagged={n_flagged} ({n_flagged/n_pixels:.4%}), "
                    f"changed={changed} ({changed_frac:.6%}){force_note}"
                )

            mask_flat = new_mask
            if changed_frac < self.convergence_fraction:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}.")
                break

        # One-sided clipping: surface is finalised; apply one symmetric pass to
        # also catch extreme low-noise outliers on both sides.
        if self.one_sided_clipping and surface_flat is not None:
            mask_flat = (np.abs(residuals_flat) > self.sigma_threshold * sigma) | prior_mask_flat
            if self.verbose:
                print("  [one_sided] Final symmetric pass applied.")

        return mask_flat, surface_flat, residuals_flat, sigma, records

    def _detect_broad(self, residuals_2d, good_2d, kernel):
        """Detect broad RFI by matched-filtering the residuals with a kernel.

        Returns ``(detection_mask, sigma_c)``. The kernel round is one-sided
        positive (broad RFI only adds power).
        """
        conv, weight = masked_normalized_convolve(residuals_2d, good_2d, kernel, mode="reflect")
        # Match the division floor in masked_normalized_convolve: a footprint whose
        # good-weight fell below the floor was divided by the floor, not its true
        # weight, so it is not a trustworthy local mean.
        valid = good_2d & (weight > _WEIGHT_FLOOR)
        if not valid.any():
            # Every footprint is fully flagged — nothing to detect.
            return np.zeros(good_2d.shape, dtype=bool), float("nan")
        thr = self._broad_threshold
        # Estimate sigma on the convolved field: this captures the ~sqrt(K) noise
        # reduction empirically, so no hand-coded factor is needed. sigma_value
        # fixes only the round-0 per-pixel sigma — the convolved field has a
        # different noise level, so broad rounds always re-estimate here. Note the
        # convolved pixels are correlated over the footprint, so the false-positive
        # rate is somewhat higher than the independent-pixel erfc(thr) would imply.
        sigma_c = self._estimate_sigma(conv[valid])
        detect = valid & (conv > thr * sigma_c)
        if self.dilate_detections:
            detect = dilate_to_footprint(detect, kernel)
        return detect, sigma_c

    def _run_broad_rounds(self, residuals_2d, accumulated, kernels, n_pixels):
        """Run one matched-filter detection round per kernel, accumulating flags.

        Appends a record per kernel to ``self.history['broad_rounds']`` and
        returns the updated accumulated (flat) mask.
        """
        if kernels and self.verbose:
            print("\n" + "=" * 60)
            print(f"Broad rounds: {len(kernels)} kernel(s), threshold={self._broad_threshold}-sigma, "
                  f"dilate={self.dilate_detections}")
            print("=" * 60)

        for i, kernel in enumerate(kernels):
            good_2d = (~accumulated).reshape(residuals_2d.shape)
            detect, sigma_c = self._detect_broad(residuals_2d, good_2d, kernel)
            before = int(accumulated.sum())
            accumulated = accumulated | detect.ravel()
            n_new = int(accumulated.sum()) - before
            desc = _describe_kernel(kernel)
            self.history["broad_rounds"].append({
                "kernel": desc, "sigma_c": sigma_c, "n_new": n_new,
                "flag_fraction": accumulated.sum() / n_pixels,
            })
            if self.verbose:
                print(f"  Kernel {i} {desc['kind']} {desc['shape']}: "
                      f"sigma_c={sigma_c:.6f}, +{n_new} new "
                      f"(total {accumulated.sum()/n_pixels:.4%})")
        return accumulated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, waterfall, kernels=None, prior_mask=None):
        """Run the round-based RFI flagging algorithm.

        Parameters
        ----------
        waterfall : ndarray, shape (n_time, n_freq)
            Raw waterfall data (positive values, linear scale).
        kernels : sequence of 2D ndarray, optional
            Convolution kernels for the broad-RFI rounds. ``None`` or empty runs
            round 0 only. Each kernel is a 2D array — e.g. a box ``np.ones((3,3))``,
            a diagonal ``np.eye(3)``, or a 1D line ``np.ones((1, k))`` /
            ``np.ones((k, 1))``. One broad round runs per kernel, in order.
        prior_mask : ndarray of bool, shape (n_time, n_freq), optional
            A priori mask of known-bad pixels. True = flagged. Excluded from all
            statistics and unconditionally True in the returned mask.

        Returns
        -------
        mask : ndarray of bool, shape (n_time, n_freq)
            Accumulated RFI mask (True = flagged).

        Notes
        -----
        - Non-finite (NaN/inf) and non-positive (<= 0) pixels are pre-flagged
          before the log10 transform and count toward the returned mask, so e.g.
          zero-valued dropouts come back flagged.
        - If fewer than ``min_good_fraction`` of pixels remain unflagged, round 0
          aborts before producing a surface: the returned mask still preserves the
          prior/bad flags, but ``self.surface`` and ``self.residuals`` stay
          ``None``, ``self.noise_sigma`` may be ``None``, and the broad rounds are
          skipped.
        """
        waterfall = np.asarray(waterfall)
        if waterfall.ndim != 2:
            raise ValueError(f"waterfall must be 2D (n_time, n_freq), got ndim={waterfall.ndim}")
        n_time, n_freq = waterfall.shape
        n_pixels = n_time * n_freq
        shape = (n_time, n_freq)

        # Input hardening: non-finite / non-positive pixels cannot be log-fit.
        bad = ~np.isfinite(waterfall) | (waterfall <= 0)
        n_bad = int(bad.sum())

        prior_mask_flat = self._flatten_prior_mask(prior_mask, n_time, n_freq)
        prior_mask_flat = prior_mask_flat | bad.ravel()

        # Safe log10: bad pixels are masked out; avoid -inf / warnings.
        safe = np.where(bad, 1.0, waterfall)
        log_flat = np.log10(safe).ravel()
        coords = build_coordinate_grid(n_time, n_freq)

        kernels = self._validate_kernels(kernels, shape)
        multi_indices = generate_multi_indices_with_degree_vec([self.degree_freq, self.degree_time])

        # ---- Round 0: surface fit ----
        if self.verbose:
            self._print_round0_header(len(multi_indices), n_bad)

        mask0_flat, surface_flat, residuals_flat, sigma0, records = self._fit_round(
            log_flat, coords, multi_indices, prior_mask_flat, n_pixels
        )
        accumulated = prior_mask_flat | mask0_flat
        self.noise_sigma = sigma0
        self.history = {"round0": {"sigma": sigma0, "iterations": records}, "broad_rounds": []}

        # ---- Rounds 1..N: broad-RFI detection (matched filter on residuals) ----
        if surface_flat is None:
            if kernels and self.verbose:
                print("  [skip] Round 0 produced no surface; broad rounds skipped.")
            self.surface = None
            self.residuals = None
        else:
            residuals_2d = residuals_flat.reshape(n_time, n_freq)
            accumulated = self._run_broad_rounds(residuals_2d, accumulated, kernels, n_pixels)
            self.surface = (10.0 ** surface_flat).reshape(n_time, n_freq)
            self.residuals = residuals_2d

        # Store final mask
        self.mask = accumulated.reshape(n_time, n_freq)

        if self.verbose:
            self._print_final_summary(n_pixels, sigma0, bool(kernels))

        return self.mask

    def _print_round0_header(self, n_basis, n_bad):
        """Print the round-0 configuration banner (verbose mode)."""
        print("=" * 60)
        print(f"Round 0: surface fit (degree freq={self.degree_freq}, time={self.degree_time})")
        if self.sigma_value is not None:
            print(f"  Sigma: fixed={self.sigma_value} (noise estimator bypassed)")
        else:
            print(f"  Noise estimator: {self.noise_estimator}"
                  + (f" (tail={self.lower_tail_fraction})" if self.noise_estimator == "lower_tail" else ""))
        if self.one_sided_clipping:
            print("  Clipping: one-sided (positive residuals only during convergence)")
        if n_bad:
            print(f"  Input hardening: {n_bad} non-finite/non-positive pixels pre-flagged.")
        print("=" * 60)
        print(f"  Number of basis terms: {n_basis}")

    def _print_final_summary(self, n_pixels, sigma0, has_kernels):
        """Print the final flagging summary (verbose mode)."""
        total = int(self.mask.sum())
        actual_frac = total / n_pixels
        gauss_frac = _gaussian_expected_flag_fraction(self.sigma_threshold)
        ratio = actual_frac / gauss_frac if gauss_frac > 0 else float("inf")
        print()
        print(f"Final: {total} pixels flagged ({actual_frac:.4%})")
        print(f"  Round-0 sigma: {sigma0:.6f}" if sigma0 is not None
              else "  Round-0 sigma: n/a (aborted before any fit)")
        print(f"  Gaussian expectation at {self.sigma_threshold:.1f}-sigma: {gauss_frac:.4%}")
        print(f"  Actual / expected: {ratio:.1f}x"
              + ("  (broad rounds add correlated detections; not directly comparable)"
                 if has_kernels else ""))

    def dilate_mask(self, kernel_size=3, axis=1):
        """Dilate self.mask with a 1D kernel along a single axis (in place).

        Any pixel whose 1D convolution result is > 0 — i.e. at least one
        flagged pixel falls within the kernel's reach along the chosen axis —
        is flagged. This is a 1D morphological dilation: flagged regions expand
        by ``(kernel_size - 1) // 2`` pixels in both directions along the axis.

        (Renamed from ``smooth_mask_with_kernel`` to avoid confusion with the
        majority-vote :func:`MomentRFI.utils.smooth_mask`, which can *erode*
        isolated flags — this method only ever *adds* flags.)

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
