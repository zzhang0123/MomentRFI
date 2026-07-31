# Changelog

## [Unreleased] — 2026-07-20

### Packaging (`pyproject.toml`)

The package had no `pyproject.toml` or `setup.py`, so it could only be used by
putting its directory on `sys.path` — and nothing downstream could depend on it
at all. `rheplicant`'s `MomentRFIFlaggingOperator` had a full test suite that
was skipped in every environment, including CI, for exactly this reason; those
tests now run and pass unmodified.

- hatchling backend, flat layout (`packages = ["MomentRFI"]`), version 0.1.0.
- Runtime dependencies: `numpy`, `scipy`, `h5py`, `MomentEmu`.
- `matplotlib` is an extra (`[plot]`) rather than a runtime dependency, because
  `__init__` deliberately does not import `plotting` — so a flagging-only
  install, which is what a pipeline wants, does not drag a plotting stack
  behind it. `[dev]` adds pytest, matplotlib and jupyter for the notebooks.

```bash
pip install -e .
pip install "MomentRFI @ git+https://github.com/zzhang0123/MomentRFI"
```

### Difference-based noise estimator (`noise_estimator="diff"`)
- New `"diff"` option estimates the round-0 per-pixel sigma from the successive
  differences of the **log-waterfall** along `diff_axis` (default: time):
  `σ = MAD(ΔL)/√2`. In log space the multiplicative thermal noise is additive and
  homoscedastic (radiometer equation), so the differenced field is a clean
  √2-scaled draw of the noise. Unlike MAD-on-residuals it is **fit-independent**
  (differencing removes any slow baseline, not just the polynomial) and **immune
  to slowly-varying broad RFI** (it cancels in the difference like the signal).
  Computed once and held fixed across round-0 iterations; broad rounds still
  estimate sigma on the convolved field. New `diff_axis` param (0=time, 1=freq).
- New public helper `MomentRFI.diff_sigma(values_2d, good_2d, axis=0)` (also
  exports `mad_sigma`).

### Tooling
- Registered a distinct `rfi_flagger` Jupyter kernel and pointed every notebook
  at it (was the generic `python3`, which resolved to whatever env launched
  Jupyter — an env lacking MomentEmu produced "No module named MomentEmu").

## [Unreleased] — 2026-07-17

### Redesign: two-phase → round-based flagging (breaking)

The two-phase algorithm (Phase 1 sigma-calibration + Phase 2 refined fit) is
replaced by a **round-based** model. There is now **one** surface fit plus one
cheap detection round per convolution kernel.

- **Round 0** — fit the anisotropic surface to `log10(waterfall)` once and
  iterate sigma-clipping. Catches **bright** RFI; produces `surface`,
  `residuals`, and `noise_sigma`.
- **Rounds 1..N** — `fit(waterfall, kernels=(...))` runs one round per kernel:
  convolve the round-0 residuals (mask-aware normalized convolution,
  `mode='reflect'`), estimate `sigma_c` on the convolved field, threshold
  one-sided positive, and optionally dilate detections to the kernel footprint.
  This is a matched filter for **faint broad** RFI — continuous RFI adds
  ~linearly under the kernel while noise adds as √K, so the broad-RFI SNR
  improves by ~√K (captured empirically via `sigma_c`, no hand-coded factor).
  Kernels are arbitrary 2D arrays: box `np.ones((3,3))`, diagonal `np.eye(3)`,
  or 1D line `np.ones((1,k))` / `np.ones((k,1))`. `kernels=None` runs round 0
  only. No surface is refit, so a flexible polynomial can never absorb the
  broad RFI it is meant to find.

Masks accumulate (union) across rounds; each round excludes already-flagged
pixels from its statistics.

#### New / changed parameters
- **Added:** `degree_freq` (10), `degree_time` (5) — the round-0 surface degrees.
- **Added:** `broad_sigma_threshold` (None → reuse `sigma_threshold`),
  `dilate_detections` (True).
- **Added:** `fit(kernels=...)`.
- **Removed:** `phase1_degree`, `phase2_degree_freq`, `phase2_degree_time`,
  `sigma_floor_factor`. Passing any of them raises a `TypeError` naming the
  replacement.
- `force_flag_fallback` and `one_sided_clipping` now govern round 0 only; broad
  rounds are always one-sided positive and never force-flag (convolution
  correlates neighbours).

#### Output attribute changes
- **Renamed:** `sigma_floor` → `noise_sigma` (final round-0 sigma).
- **Restructured:** `history` is now
  `{"round0": {"sigma", "iterations": [...]}, "broad_rounds": [...]}` instead of
  `{"phase1": [...], "phase2": [...]}`.
- `mask`, `surface`, `residuals` are unchanged in meaning (round-0 fit + the
  accumulated mask), so `plot_summary()` is unaffected. `plot_convergence()` was
  rewritten for the new history.

#### Robustness
- **Input hardening:** non-finite and non-positive waterfall pixels are
  automatically folded into the mask before `log10`, so zeros / NaNs / negatives
  no longer poison the fit, convolution, or noise estimate.
- **New utils:** `masked_normalized_convolve`, `dilate_to_footprint`.
- **Tests:** added a `tests/` pytest suite (utils + core + plotting/io, boundary cases).

#### Review-driven hardening
- **Constructor/`fit` validation:** `degree_freq`/`degree_time` (non-negative
  ints), `sigma_threshold`/`broad_sigma_threshold` (> 0), `min_good_fraction`,
  `lower_tail_fraction`, `max_iterations` ranges, and `waterfall.ndim == 2` now
  raise clear `ValueError`s.
- **`force_flag_fallback`:** the fallback now selects top-N candidates among
  *good* (unflagged) pixels only, so it cannot re-select already-flagged RFI and
  stall.
- **`lower_tail_sigma`:** clamps the partition index (no crash at extreme
  `tail_fraction`) and guards a zero/`nan` regression slope (degenerate tails
  fall back to RMS instead of returning `nan`).
- **Masked convolution:** the division floor and the caller's validity test share
  one `_WEIGHT_FLOOR` constant.
- **Docs:** fixed stale two-phase wording, the `sigma_value` "every round" claim,
  and a non-existent-notebook reference.
- **Rename:** `IterativeSurfaceFitter.smooth_mask_with_kernel` →
  `dilate_mask` (it only ever *adds* flags; the old name collided with the
  majority-vote `utils.smooth_mask`, which can *erode* isolated flags).

#### Performance
- **Basis caching (~5× faster round 0, bit-identical):** the coordinate grid is
  fixed for a whole fit, so the polynomial Vandermonde `Phi` is now built once and
  reused across all sigma-clip iterations instead of being rebuilt every
  iteration (which dominated the runtime). A 1135×8192 fit drops from ~95 s to
  ~19 s, with a **bit-identical** mask (verified across degree configs). The
  full-grid surface is still evaluated each iteration, so wrongly-flagged good
  pixels can still be re-admitted — the optimization changes speed only, not
  results.
- **New params:** `precompute_basis` (default True) and `max_basis_gb` (default
  16.0). The basis costs `N·D·8` bytes; when it would exceed `max_basis_gb`,
  `fit()` automatically falls back to MomentEmu's constant-memory path so huge
  waterfalls never OOM by default. Set `precompute_basis=False` to force it.
- **New module:** `MomentRFI/surface.py` holds the surface-fitting primitives
  (MomentEmu path + cached-basis fast path), keeping `core.py` focused.

## [Unreleased] — 2026-02-18

### New features

#### A priori mask input (`fit(prior_mask=...)`)
`fit()` now accepts an optional `prior_mask` boolean array. Pixels flagged there are excluded from surface fitting in both phases and are unconditionally `True` in the returned mask. Intended for chaining: run a first pass, then feed the result back as the prior for a second, tighter pass.

#### Skippable Phase 2
Setting `phase2_degree_freq=None` or `phase2_degree_time=None` causes `fit()` to return the Phase 1 mask directly without resetting or re-fitting. Useful when a lower-degree isotropic fit is sufficient.

#### Fixed-sigma override (`sigma_value`)
When `sigma_value` is set, that value is used as sigma in every iteration of both phases, bypassing the noise estimator and the Phase 2 sigma floor entirely. Useful for debugging or when the noise level is known a priori.

#### One-sided clipping (`one_sided_clipping=True`)
During convergence iterations, only pixels *above* the surface (`residual > +k·sigma`) are flagged — physically motivated since RFI adds power, never removes it. A single final symmetric pass is applied after convergence to also catch extreme low-noise statistical outliers on both sides.

#### Post-processing methods on `IterativeSurfaceFitter`

- **`dilate_mask(kernel_size=3, axis=1)`** — 1D morphological dilation of `self.mask` along either the time (`axis=0`) or frequency (`axis=1`) axis. Any pixel within `(kernel_size-1)//2` steps of a flagged pixel along that axis is also flagged. (Renamed from `smooth_mask_with_kernel`.)
- **`flag_by_fraction(threshold, axis)`** — flags any time row (`axis=0`) or frequency column (`axis=1`) whose flagged-pixel fraction ≥ `threshold`. Prevents thin slivers of nominally clean data surviving in heavily contaminated rows/columns.

Both methods update `self.mask` in place, return the new mask, and raise `RuntimeError` if called before `fit()`.

### Improvements

#### Robust colour scaling in `plot_summary()`
The three waterfall panels (original, fitted surface, flagged) now share percentile-based colour limits (`vmin` = 1st percentile, `vmax` = 99th percentile), so bright RFI spikes no longer compress the rest of the colour scale into darkness. All three panels use the same limits, making them directly comparable.

`plot_waterfall()` also now accepts a caller-supplied `norm` keyword, forwarded to `imshow`.

### Documentation

- Added `fit()` parameter table to README (previously undocumented).
- Documented `sigma_value`, `one_sided_clipping`, and the two post-processing methods with parameter tables and usage examples.
- Phase 2 now described as optional throughout.
