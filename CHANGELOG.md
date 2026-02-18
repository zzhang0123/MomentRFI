# Changelog

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

- **`smooth_mask_with_kernel(kernel_size=3, axis=1)`** — 1D morphological dilation of `self.mask` along either the time (`axis=0`) or frequency (`axis=1`) axis. Any pixel within `(kernel_size-1)//2` steps of a flagged pixel along that axis is also flagged.
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
