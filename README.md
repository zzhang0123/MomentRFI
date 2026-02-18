# MomentRFI

Iterative two-phase sigma-clipping for flagging Radio Frequency Interference (RFI) in radio cosmology waterfall data, using 2D polynomial surface fitting powered by [MomentEmu](https://github.com/zzhang0123/MomentEmu).

## Installation

Dependencies: `numpy`, `scipy`, `matplotlib`, `h5py`, `jupyter`, `MomentEmu`.

## Quick Start

```python
from MomentRFI import load_waterfall, validate_waterfall, IterativeSurfaceFitter
from MomentRFI.plotting import plot_summary

waterfall, freqs, times = load_waterfall("2025-12-02_16-54-49_obs.hd5f")
validate_waterfall(waterfall)

fitter = IterativeSurfaceFitter()
mask = fitter.fit(waterfall)

plot_summary(waterfall, fitter, freqs, times)
```

See `notebooks/demo_rfi_flagging.ipynb` for a full walkthrough.

## Algorithm

### The Problem

Radio waterfall data (time x frequency) contains a smooth astrophysical signal spanning several orders of magnitude, contaminated by narrow-band or transient RFI. The goal is to fit the smooth background and flag outlier pixels as RFI.

The key challenge is **coupling between polynomial degree and sigma threshold**: a low-degree polynomial leaves large residuals that inflate sigma, hiding real RFI. A high-degree polynomial can overfit RFI features, suppressing sigma and causing runaway flagging. A two-phase approach decouples these.

### Two-Phase Strategy

All fitting is performed in **log10 space** with coordinates normalized to **[-1, 1]**.

#### Phase 1: Sigma Calibration

1. Fit an **isotropic** polynomial of conservative degree (default 5, giving 21 basis terms) to the full waterfall.
2. Compute residuals (log10 data - log10 surface) at all pixels.
3. Estimate noise from unflagged pixels using the chosen estimator (MAD-sigma by default; see [Noise Estimators](#noise-estimators) below).
4. Flag pixels where |residual| > sigma_threshold * sigma.
5. Repeat from step 1 using only unflagged pixels for the fit.
6. Stop when <0.001% of pixels change between iterations (convergence), >50% are flagged (safety abort), or 15 iterations are reached.
7. Record the final sigma as the **sigma floor**.

The low-degree polynomial intentionally underfits fine spectral structure, producing a reliable upper bound on the true noise level.

#### Phase 2: Refined Fitting (optional)

Phase 2 is skipped when `phase2_degree_freq=None` or `phase2_degree_time=None`; in that case the Phase 1 mask, surface, and residuals are returned directly.

1. **Reset the mask** (start from any a priori flags supplied via `prior_mask`; discard Phase 1 sigma-clip flags).
2. Fit an **anisotropic** polynomial with higher frequency-axis degree (default 10) and the same time-axis degree (default 5), giving 66 basis terms. This better captures spectral structure without overfitting time-domain variations.
3. Compute residuals and sigma as before, but enforce: `sigma_used = max(sigma_raw, sigma_floor)`. This prevents the improved polynomial from driving sigma too low and causing runaway flagging.
4. Flag and iterate with the same stopping criteria as Phase 1.

The result is a mask with fewer false positives than Phase 1 alone, because the higher-degree polynomial removes spectral structure that Phase 1 would misidentify as RFI.

### Noise Estimators

Two options are available via the `noise_estimator` parameter:

**`"mad"` (default)** — Median Absolute Deviation: `1.4826 * median(|x - median(x)|)`. Robust up to ~50% contamination. The median remains anchored to the clean population even when many pixels are RFI. This is the right choice for most datasets.

**`"lower_tail"`** — Zero-mean Gaussian fit to the lower tail. RFI adds power, so it only inflates the *upper* tail of the residual distribution. The lower tail should be clean noise. We histogram the bottom `lower_tail_fraction` (default 20%) of residuals and fit `A * exp(-x² / 2σ²)` analytically via linear regression of `log(counts)` vs `x²` — no iterative optimisation, just a closed-form solution.

| | MAD | Lower-tail |
|---|---|---|
| **Robust up to** | ~50% contamination | >50% (only lower tail needs to be clean) |
| **Speed** | Fast (two passes over data) | Fast (O(n) partition + histogram + linear regression) |
| **On clean data** | Tighter sigma, more sensitive | Wider sigma (includes spectral structure), more conservative |
| **Best for** | Moderate RFI (<50%) | Heavy RFI (>50%) where MAD breaks down |

(Note: I found MAD generally works better, at least for Phase 1.)

### Polynomial Basis

For a 2D polynomial (frequency, time):
- **Isotropic degree d**: all monomials `freq^a * time^b` where `a + b <= d`. Number of terms = `(d+1)(d+2)/2`.
- **Anisotropic degrees (d_freq, d_time)**: all monomials where `a <= d_freq` and `b <= d_time`. Number of terms = `(d_freq+1) * (d_time+1)`.

Fitting uses the **moment method**: accumulate `M = Phi^T Phi / N` and `nu = Phi^T y / N` in batches (only building the small D x D matrix, never the full N x D design matrix), then solve `M c = nu`. This keeps memory usage constant regardless of waterfall size.

## Parameters

### `IterativeSurfaceFitter`

| Parameter | Default | Description |
|---|---|---|
| `sigma_threshold` | 4.0 | Clipping threshold in units of sigma. Pixels with \|residual\| > threshold * sigma are flagged. Lower values flag more aggressively. |
| `phase1_degree` | 5 | Isotropic polynomial degree for Phase 1 (sigma calibration). Higher values fit more spectral detail but risk absorbing RFI into the model. |
| `phase2_degree_freq` | 10 | Frequency-axis degree for Phase 2. Frequency structure typically needs higher polynomial order than time. Set to `None` (with `phase2_degree_time=None`) to skip Phase 2 and return the Phase 1 mask directly. |
| `phase2_degree_time` | 5 | Time-axis degree for Phase 2. Time variations are usually smoother. Set to `None` (with `phase2_degree_freq=None`) to skip Phase 2. |
| `sigma_floor_factor` | 1.0 | Multiplier on the Phase 1 sigma to set the floor. Values > 1.0 make Phase 2 more conservative (fewer flags). |
| `convergence_fraction` | 1e-5 | Iteration stops when the fraction of pixels that changed state is below this value. |
| `min_good_fraction` | 0.5 | Safety abort: if the fraction of unflagged pixels drops below this, iteration stops immediately. |
| `max_iterations` | 15 | Hard cap on iterations per phase. |
| `batch_size` | 200,000 | Number of pixels processed per batch during polynomial evaluation. Controls memory vs speed tradeoff. |
| `noise_estimator` | `"mad"` | `"mad"` or `"lower_tail"`. See [Noise Estimators](#noise-estimators). |
| `lower_tail_fraction` | 0.2 | Fraction of lowest residuals used by the `"lower_tail"` estimator. Smaller = more conservative but noisier. |
| `sigma_value` | `None` | Fixed sigma for clipping. If set, bypasses both the noise estimator and the Phase 2 sigma floor. Default `None` estimates sigma from data each iteration. |
| `force_flag_fallback` | False | Force-flag top outliers when sigma is overestimated and flagging stalls (see below). |
| `one_sided_clipping` | False | If True, convergence iterations only flag pixels above the surface (`residual > +k·sigma`). A final symmetric pass is applied after convergence to also flag extreme low-noise outliers. Default False (symmetric clipping throughout). |
| `verbose` | True | Print per-iteration diagnostics. |

### `fit()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `waterfall` | — | 2D ndarray `(n_time, n_freq)`, positive linear-scale power values. |
| `prior_mask` | `None` | Optional bool ndarray `(n_time, n_freq)`. `True` = known-bad pixel. Prior-flagged pixels are excluded from surface fitting in both phases and are always `True` in the returned mask, regardless of their residual. |

### Outputs (after calling `.fit()`)

| Attribute | Type | Description |
|---|---|---|
| `mask` | `ndarray[bool]` (n_time, n_freq) | `True` where RFI is flagged. |
| `surface` | `ndarray[float]` (n_time, n_freq) | Fitted polynomial surface in **linear** scale (10^fitted_log10). |
| `residuals` | `ndarray[float]` (n_time, n_freq) | Residuals in **log10** scale (log10_data - log10_surface). |
| `sigma_floor` | `float` | MAD-sigma from Phase 1 convergence. |
| `history` | `dict` | Per-iteration diagnostics for both phases (sigma, flag count, convergence). |

### Post-processing Methods

After calling `.fit()`, two methods are available to refine the mask:

#### `smooth_mask_with_kernel(kernel_size=3, axis=1)`

Dilate `self.mask` with a 1D kernel along a single axis. Any pixel that lies within `(kernel_size - 1) // 2` steps of a flagged pixel along the chosen axis is also flagged. This is a 1D morphological dilation.

| Parameter | Default | Description |
|---|---|---|
| `kernel_size` | 3 | Length of the uniform 1D kernel. Larger values produce a wider dilation. |
| `axis` | 1 | `0` to dilate along time (flags spread to neighbouring time samples), `1` to dilate along frequency (flags spread to neighbouring channels). |

Updates `self.mask` in place and returns the new mask. Raises `RuntimeError` if called before `fit()`, `ValueError` if `axis` is not 0 or 1.

```python
# Dilate 3 channels wide along frequency
mask = fitter.smooth_mask_with_kernel(kernel_size=3, axis=1)

# Dilate 5 time samples wide along time
mask = fitter.smooth_mask_with_kernel(kernel_size=5, axis=0)
```

#### `flag_by_fraction(threshold, axis)`

Flag entire time rows or frequency columns where the fraction of already-flagged pixels meets or exceeds `threshold`. Avoids retaining thin slivers of nominally "good" data in heavily contaminated rows/columns.

| Parameter | Description |
|---|---|
| `threshold` | Float in [0, 1]. Flag the entire row/column if its flagged fraction ≥ this value. E.g. `0.5` flags any row/column that is already more than half flagged. |
| `axis` | `0` to operate on time rows; `1` to operate on frequency columns. |

Updates `self.mask` in place and returns the new mask. Raises `RuntimeError` if called before `fit()`, `ValueError` if `axis` is not 0 or 1.

```python
# Flag any frequency channel that is >50% flagged
mask = fitter.flag_by_fraction(threshold=0.5, axis=1)

# Flag any time sample that is >80% flagged
mask = fitter.flag_by_fraction(threshold=0.8, axis=0)
```

## Data Format

Expects HDF5 files with the structure:
```
sdr/
  sdr_waterfall   (n_time, n_freq)  float64   -- power values, must be all positive
  sdr_freqs       (n_freq,)         float64   -- frequency axis in MHz
  sdr_times       (n_time,)         float64   -- time axis in seconds
```

## Tuning Guide

- **Too many flags?** Increase `sigma_threshold` (try 4.5 or 5.0) or increase `sigma_floor_factor`.
- **Missing faint RFI?** Decrease `sigma_threshold` (try 3.5), but watch for runaway flagging via the convergence plot.
- **>50% RFI? (BETA)** Switch to `noise_estimator="lower_tail"`. MAD breaks down above ~50% contamination; the lower-tail fit stays valid as long as RFI only adds power.
- **Polynomial ringing at band edges?** Decrease `phase2_degree_freq`.
- **Slow convergence?** Usually not an issue (typically 7-12 iterations), but can lower `max_iterations` to cap runtime.

## Project Structure

```
RFI_flagger/
├── MomentRFI/
│   ├── __init__.py      # Package exports
│   ├── core.py          # IterativeSurfaceFitter (imports polynomial fitting from MomentEmu)
│   ├── io.py            # load_waterfall(), validate_waterfall()
│   ├── utils.py         # mad_sigma(), lower_tail_sigma(), coordinate grid utilities
│   └── plotting.py      # Visualization functions
├── notebooks/
│   └── demo_rfi_flagging.ipynb
└── data/
```
