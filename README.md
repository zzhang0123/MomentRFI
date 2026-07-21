# MomentRFI

Round-based iterative sigma-clipping for flagging Radio Frequency Interference (RFI) in radio cosmology waterfall data, using 2D polynomial surface fitting powered by [MomentEmu](https://github.com/zzhang0123/MomentEmu). One surface-fitting round catches bright RFI; optional per-kernel matched-filter rounds catch faint broad RFI.

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

Radio waterfall data (time × frequency) contains a smooth astrophysical signal spanning several orders of magnitude, contaminated by narrow-band or transient RFI. The goal is to fit the smooth background and flag outlier data points as RFI.

> **Terminology:** throughout this documentation, *pixel* is used as a convenient shorthand for a single data point in the 2D waterfall array — i.e. one (time, frequency) sample. The data are not images, but the term is standard in RFI flagging literature.

RFI comes in two regimes that need different detectors. **Bright** RFI (narrow spikes, streaks) stands out per-pixel and is caught by thresholding the surface-fit residuals directly. **Faint broad** RFI (spatially or spectrally continuous, low amplitude) is buried in per-pixel noise but stands out when neighbouring pixels are summed — because continuous RFI adds ~linearly under a kernel while thermal noise adds in quadrature.

### Round-Based Strategy

All fitting is performed in **log10 space** with coordinates normalized to **[-1, 1]**. The algorithm runs **one surface-fitting round plus one detection round per convolution kernel**. Masks accumulate (union) across rounds.

#### Round 0: Surface Fit (bright RFI)

1. Fit an **anisotropic** polynomial (default degrees: frequency 10, time 5 → 66 basis terms) to `log10(waterfall)`.
2. Compute residuals (log10 data − log10 surface) at all pixels.
3. Estimate noise from unflagged pixels using the chosen estimator (MAD by default; see [Noise Estimators](#noise-estimators)).
4. Flag pixels where |residual| > `sigma_threshold * sigma`.
5. Repeat from step 1 using only unflagged pixels for the fit.
6. Stop on convergence (< `convergence_fraction` of pixels change), safety abort (< `min_good_fraction` unflagged), or `max_iterations`.

Round 0 produces the baseline `surface`, the `residuals`, and the estimated `noise_sigma`. It replaces the old two-phase design: the separate low-degree "sigma calibration" phase is gone — the sigma floor it produced is no longer needed because there is only one fit and the broad rounds re-estimate their own noise.

#### Rounds 1..N: Broad-RFI Detection (per kernel, matched filter)

Pass a tuple of kernels to `fit(waterfall, kernels=(...))`. For each kernel, in order:

1. **Convolve the round-0 residuals** with the kernel, mask-aware (already-flagged pixels are excluded via a normalized convolution `(residuals·good ⊛ K)/(good ⊛ K)`, `mode='reflect'`), so bright RFI cannot leak into its neighbours.
2. Estimate `sigma_c` on the convolved residuals. Because the box averages K pixels, `sigma_c ≈ noise_sigma / √K` — the √K SNR boost is captured automatically, with **no hand-coded factor**.
3. Flag where `convolved_residual > threshold * sigma_c` (one-sided positive — broad RFI only *adds* power). The threshold is `broad_sigma_threshold` if set, else `sigma_threshold`.
4. If `dilate_detections` (default True), dilate each detection to the kernel's footprint so the broad RFI's full extent (including wings) is flagged.
5. Union the detections into the accumulated mask.

No surface is refit — the single round-0 baseline is reused, so a flexible polynomial can never "absorb" the broad RFI it is meant to find. This is the exact matched filter for a footprint-shaped RFI template in additive noise.

Kernels can be any 2D array: a box `np.ones((3, 3))`, a diagonal `np.eye(3)` (drifting emitters), or a 1D line `np.ones((1, k))` / `np.ones((k, 1))` (broad in frequency / time). Passing no kernels runs round 0 only.

### Noise Estimators

Three options are available via the `noise_estimator` parameter:

**`"mad"` (default)** — Median Absolute Deviation of the residuals: `1.4826 * median(|x - median(x)|)`. Robust up to ~50% contamination. The median remains anchored to the clean population even when many pixels are RFI. This is the right choice for most datasets.

**`"lower_tail"`** — Zero-mean Gaussian fit to the lower tail. RFI adds power, so it only inflates the *upper* tail of the residual distribution. The lower tail should be clean noise. We histogram the bottom `lower_tail_fraction` (default 20%) of residuals and fit `A * exp(-x² / 2σ²)` analytically via linear regression of `log(counts)` vs `x²` — no iterative optimisation, just a closed-form solution.

**`"diff"`** — Successive-difference estimator on the **log-waterfall**. In log space the multiplicative thermal noise becomes additive and homoscedastic (radiometer equation). Assuming the signal varies slowly along `diff_axis` (default: time) and per-pixel noise is independent, the first difference `ΔL` cancels the signal and has standard deviation `√2·σ`, so `σ = MAD(ΔL)/√2`. Two properties make this attractive: it is **fit-independent** (differencing removes *any* slowly-varying baseline, not just the fitted polynomial) and **immune to slowly-varying broad RFI** (which cancels in the difference just like the signal — only fast/narrow outliers survive, and MAD rejects those). The `"diff"` sigma is computed once from the raw log-waterfall and held fixed across round-0 iterations (a stable noise floor); the broad-RFI rounds still estimate their own sigma on the convolved field.

| | MAD | Lower-tail | Diff |
|---|---|---|---|
| **Estimates from** | residuals | residuals (lower tail) | log-waterfall differences |
| **Fit-dependent?** | yes | yes | **no** |
| **Robust up to** | ~50% | >50% | ~50% (narrow RFI); broad RFI cancels entirely |
| **Best for** | moderate RFI | heavy RFI (>50%) | data with broad RFI or an imperfect surface fit |

(Note: I found MAD generally works well for round 0; `"diff"` is a good choice when broad RFI or baseline curvature would otherwise inflate the MAD.)

### Polynomial Basis

For a 2D polynomial (frequency, time):
- **Isotropic degree d**: all monomials `freq^a * time^b` where `a + b <= d`. Number of terms = `(d+1)(d+2)/2`.
- **Anisotropic degrees (d_freq, d_time)**: all monomials where `a <= d_freq` and `b <= d_time`. Number of terms = `(d_freq+1) * (d_time+1)`.

Fitting uses the **moment method**: accumulate `M = Phi^T Phi / N` and `nu = Phi^T y / N` in batches, then solve `M c = nu`.

**Basis caching (performance).** The coordinate grid is fixed for a whole fit, so rebuilding the monomial basis every sigma-clip iteration is pure redundancy — it dominates the runtime. With `precompute_basis=True` (default) the Vandermonde `Phi` is built **once** and reused across all iterations, giving a **~5× speedup** (e.g. a 1135×8192 fit drops from ~95 s to ~19 s) with **bit-identical** results. This trades the moment method's constant memory for `N·D·8` bytes of `Phi`; the `max_basis_gb` cap auto-falls-back to the constant-memory MomentEmu path when `Phi` would be too large. The full-grid surface is still evaluated every iteration, so wrongly-flagged good pixels can be re-admitted as the fit improves.

## Parameters

### `IterativeSurfaceFitter`

| Parameter | Default | Description |
|---|---|---|
| `sigma_threshold` | 4.0 | Clipping threshold in units of sigma (round 0, and broad rounds unless overridden). Pixels with \|residual\| > threshold * sigma are flagged. Lower values flag more aggressively. |
| `degree_freq` | 10 | Frequency-axis degree of the round-0 anisotropic surface. Frequency structure typically needs higher polynomial order than time. |
| `degree_time` | 5 | Time-axis degree of the round-0 surface. Time variations are usually smoother. |
| `broad_sigma_threshold` | `None` | Threshold (in sigma) for the broad-RFI kernel rounds. `None` reuses `sigma_threshold`. |
| `dilate_detections` | True | Dilate each broad-round detection to the kernel footprint, flagging the full spatial extent of the broad RFI rather than only the footprint centre. |
| `convergence_fraction` | 1e-5 | Round-0 iteration stops when the fraction of pixels that changed state is below this value. |
| `min_good_fraction` | 0.5 | Safety abort: if the fraction of unflagged pixels drops below this, round-0 iteration stops immediately. |
| `max_iterations` | 15 | Hard cap on round-0 iterations. |
| `batch_size` | 200,000 | Number of pixels processed per batch during polynomial evaluation. Controls memory vs speed tradeoff. |
| `noise_estimator` | `"mad"` | `"mad"`, `"lower_tail"`, or `"diff"`. See [Noise Estimators](#noise-estimators). |
| `lower_tail_fraction` | 0.2 | Fraction of lowest residuals used by the `"lower_tail"` estimator. Smaller = more conservative but noisier. |
| `diff_axis` | 0 | Axis the `"diff"` estimator differences along: `0` = time, `1` = frequency. Pick the axis the signal varies most slowly along. |
| `sigma_value` | `None` | Fixed sigma for round-0 clipping. If set, bypasses the noise estimator in round 0. Broad rounds always re-estimate their own sigma on the convolved field. Default `None` estimates sigma from data. |
| `force_flag_fallback` | False | Round-0 only: force-flag top outliers when sigma is overestimated and flagging stalls. Deliberately not applied to broad rounds (convolution correlates neighbours, so the Gaussian count would force-flag noise blobs). |
| `one_sided_clipping` | False | Round-0 only: if True, convergence iterations only flag pixels above the surface (`residual > +k·sigma`), with a final symmetric pass. Broad rounds are always one-sided positive. Default False. |
| `precompute_basis` | True | Build the polynomial basis (Vandermonde) once and reuse it across round-0 iterations instead of rebuilding it each iteration — ~5× faster with **bit-identical** results. Costs `N·D·8` bytes (`D = (degree_freq+1)(degree_time+1)`). Set False for MomentEmu's constant-memory path. |
| `max_basis_gb` | 16.0 | Memory cap (GB) for the precomputed basis. If the full Vandermonde would exceed this, `fit()` automatically falls back to the constant-memory path, so huge waterfalls never OOM by default. |
| `verbose` | True | Print per-round diagnostics. |

### `fit()` Parameters

| Parameter | Default | Description |
|---|---|---|
| `waterfall` | — | 2D ndarray `(n_time, n_freq)`, positive linear-scale power values. Non-finite or non-positive pixels are automatically pre-flagged. |
| `kernels` | `None` | Sequence of 2D ndarrays. `None`/empty runs round 0 only. One broad-RFI round runs per kernel, in order. Each kernel is 2D: box `np.ones((3,3))`, diagonal `np.eye(3)`, or 1D line `np.ones((1,k))` / `np.ones((k,1))`. |
| `prior_mask` | `None` | Optional bool ndarray `(n_time, n_freq)`. `True` = known-bad pixel. Prior-flagged pixels are excluded from all statistics and are always `True` in the returned mask. |

### Outputs (after calling `.fit()`)

| Attribute | Type | Description |
|---|---|---|
| `mask` | `ndarray[bool]` (n_time, n_freq) | `True` where RFI is flagged (accumulated across all rounds). |
| `surface` | `ndarray[float]` (n_time, n_freq) | Round-0 fitted polynomial surface in **linear** scale (10^fitted_log10). |
| `residuals` | `ndarray[float]` (n_time, n_freq) | Round-0 residuals in **log10** scale (the field the kernels convolve). |
| `noise_sigma` | `float` | Final round-0 sigma (the estimated noise level). |
| `history` | `dict` | `{"round0": {"sigma", "iterations": [...]}, "broad_rounds": [{"kernel", "sigma_c", "n_new", "flag_fraction"}, ...]}`. |

### Post-processing Methods

After calling `.fit()`, two methods are available to refine the mask:

#### `dilate_mask(kernel_size=3, axis=1)`

Dilate `self.mask` with a 1D kernel along a single axis. Any pixel that lies within `(kernel_size - 1) // 2` steps of a flagged pixel along the chosen axis is also flagged. This is a 1D morphological dilation.

| Parameter | Default | Description |
|---|---|---|
| `kernel_size` | 3 | Length of the uniform 1D kernel. Larger values produce a wider dilation. |
| `axis` | 1 | `0` to dilate along time (flags spread to neighbouring time samples), `1` to dilate along frequency (flags spread to neighbouring channels). |

Updates `self.mask` in place and returns the new mask. Raises `RuntimeError` if called before `fit()`, `ValueError` if `axis` is not 0 or 1.

```python
# Dilate 3 channels wide along frequency
mask = fitter.dilate_mask(kernel_size=3, axis=1)

# Dilate 5 time samples wide along time
mask = fitter.dilate_mask(kernel_size=5, axis=0)
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
  sdr_waterfall   (n_time, n_freq)  float64   -- power values, ideally all positive
                                              --   (fit() auto-pre-flags non-finite / <=0 pixels)
  sdr_freqs       (n_freq,)         float64   -- frequency axis in MHz
  sdr_times       (n_time,)         float64   -- time axis in seconds
```

## Tuning Guide

- **Too many flags?** Increase `sigma_threshold` (try 4.5 or 5.0).
- **Missing faint narrow RFI?** Decrease `sigma_threshold` (try 3.5), but watch for runaway flagging via the convergence plot.
- **Missing faint *broad* RFI?** Add kernels matched to the RFI shape: a frequency-broad emitter → `np.ones((1, k))`; a time-persistent one → `np.ones((k, 1))`; a compact blob → `np.ones((3, 3))`; a drifting emitter → `np.eye(k)`. Larger kernels detect fainter, broader RFI (bigger √K) but blur fine structure — keep the kernel smaller than real baseline features. Tune broad sensitivity separately with `broad_sigma_threshold`.
- **>50% RFI? (BETA)** Switch to `noise_estimator="lower_tail"`. MAD breaks down above ~50% contamination; the lower-tail fit stays valid as long as RFI only adds power.
- **Broad RFI or curved baseline inflating sigma?** Switch to `noise_estimator="diff"` — it reads the noise from time-differences of the log-waterfall, so broad RFI and any slow baseline cancel out and don't inflate the estimate.
- **Polynomial ringing at band edges?** Decrease `degree_freq`.
- **Slow convergence?** Usually not an issue (typically 7-12 iterations), but can lower `max_iterations` to cap runtime.

Example with broad-RFI rounds:

```python
import numpy as np
fitter = IterativeSurfaceFitter(sigma_threshold=4.0)
mask = fitter.fit(waterfall, kernels=(np.ones((1, 9)), np.ones((5, 1))))
# round 0 catches bright RFI; the two line kernels catch broad
# frequency- and time-continuous RFI respectively.
```

## Project Structure

```
MomentRFI/
├── MomentRFI/
│   ├── __init__.py      # Package exports
│   ├── core.py          # IterativeSurfaceFitter (imports polynomial fitting from MomentEmu)
│   ├── io.py            # load_waterfall(), validate_waterfall()
│   ├── utils.py         # mad_sigma(), lower_tail_sigma(), diff_sigma(),
│   │                    #   coordinate grid, masked_normalized_convolve(),
│   │                    #   dilate_to_footprint()
│   └── plotting.py      # Visualization functions
├── notebooks/
│   ├── demo_rfi_flagging.ipynb         # walkthrough (source)
│   └── executed_demo_1.ipynb, ...      # pre-executed variants + comparisons
├── tests/               # pytest suite (utils + round-based core, boundary tests)
└── data/
```
