import numpy as np
from scipy.ndimage import uniform_filter, convolve

# Smallest convolved good-weight treated as a usable footprint in
# masked_normalized_convolve. A footprint whose weight falls at or below this is
# divided by the floor (not its true weight), so callers should also treat
# ``weight > _WEIGHT_FLOOR`` as the validity test.
_WEIGHT_FLOOR = 1e-12


def mad_sigma(residuals):
    """Compute robust estimate of standard deviation using Median Absolute Deviation.

    sigma_MAD = 1.4826 * median(|x - median(x)|)

    Parameters
    ----------
    residuals : array_like
        1D array of residual values.

    Returns
    -------
    float
        MAD-based sigma estimate.
    """
    med = np.median(residuals)
    return 1.4826 * np.median(np.abs(residuals - med))


def diff_sigma(values_2d, good_2d, axis=0):
    """Robust noise sigma from successive differences along ``axis``.

    Under the assumptions that the underlying signal varies slowly along
    ``axis`` and the per-pixel noise is independent, the first difference
    ``D = diff(values, axis)`` cancels the signal, leaving noise with standard
    deviation ``sqrt(2) * sigma``. So ``sigma = MAD(D) / sqrt(2)``, with MAD the
    robust (1.4826-scaled) estimator over the *valid* pairs (both endpoints
    good).

    Motivation (radiometer equation): in ``log`` space the multiplicative
    thermal noise becomes additive and homoscedastic, so the differenced field
    is a clean sqrt(2)-scaled draw of that noise. This estimator is therefore
    **fit-independent** — differencing removes any slowly-varying baseline, not
    just a fitted polynomial — and **immune to slowly-varying broad RFI**, which
    cancels in the difference exactly like the signal; only fast/narrow outliers
    survive, and MAD rejects those.

    Parameters
    ----------
    values_2d : ndarray, shape (n_time, n_freq)
        The field to difference (typically ``log10`` of the waterfall).
    good_2d : ndarray of bool, same shape
        True where the pixel is usable. A difference is counted only when both
        of its endpoints are good.
    axis : int
        Axis to difference along (0 = time, default; 1 = frequency). Choose the
        axis along which the signal varies most slowly.

    Returns
    -------
    float
        Estimated per-pixel noise sigma, or ``nan`` if there are no valid
        difference pairs (e.g. a single sample along ``axis``, or everything
        masked). Pure function: inputs are not mutated.
    """
    d = np.diff(values_2d, axis=axis)
    if axis == 0:
        pair_good = good_2d[:-1, :] & good_2d[1:, :]
    else:
        pair_good = good_2d[:, :-1] & good_2d[:, 1:]
    valid = d[pair_good]
    if valid.size == 0:
        return float("nan")
    return mad_sigma(valid) / np.sqrt(2.0)


def lower_tail_sigma(residuals, tail_fraction=0.2, max_samples=20_000):
    """Estimate sigma by fitting a zero-mean Gaussian to the lower tail.

    Residuals are centered at zero by construction (polynomial fit).  RFI
    adds power, so only the upper tail is contaminated.  We histogram the
    bottom *tail_fraction*, then fit  A * exp(-x^2 / 2 sigma^2)  via
    linear regression of  log(counts) vs x^2  — a closed-form solution
    with no iterative optimisation.

    Parameters
    ----------
    residuals : array_like
        1D array of residual values.
    tail_fraction : float
        Fraction of lowest values to use (e.g. 0.2 = bottom 20%).
    max_samples : int
        If the lower tail has more points than this, randomly subsample
        before histogramming for speed.

    Returns
    -------
    float
        Fitted Gaussian sigma.
    """
    n = len(residuals)
    # Clamp into a valid partition index: tail_fraction near 0 or 1 (or n small)
    # would otherwise put k out of [0, n-1] and raise in np.partition.
    k = min(max(int(n * tail_fraction), 1), n - 1)

    # O(n) partial sort to find the threshold
    threshold = np.partition(residuals, k)[k]
    lower = residuals[residuals <= threshold]

    if len(lower) > max_samples:
        rng = np.random.default_rng(42)
        lower = rng.choice(lower, max_samples, replace=False)

    counts, edges = np.histogram(lower, bins=200)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Keep only bins with counts > 0 (log is undefined at 0)
    pos = counts > 0
    z = centers[pos] ** 2          # x_i^2
    y = np.log(counts[pos].astype(float))  # log(c_i)

    # Linear regression:  y = a + b*z,  where b = -1/(2*sigma^2)
    z_mean = z.mean()
    y_mean = y.mean()
    denom = np.dot(z - z_mean, z - z_mean)

    # sigma = sqrt(-1 / (2b));  b must be negative for a valid Gaussian.
    # Degenerate tails (a single occupied bin -> denom == 0) give b = nan, which
    # `b >= 0` would NOT catch — guard denom explicitly and fall back to RMS.
    if denom == 0 or not np.isfinite(denom):
        return float(np.sqrt(np.mean(lower ** 2)))
    b = np.dot(z - z_mean, y - y_mean) / denom
    if not (b < 0):  # covers b >= 0 and b == nan
        # Fallback: use RMS of the lower-tail data as rough sigma
        return float(np.sqrt(np.mean(lower ** 2)))

    return float(np.sqrt(-0.5 / b))


def normalize_to_interval(values, low=-1.0, high=1.0):
    """Linearly map values to [low, high].

    Parameters
    ----------
    values : array_like
        Input values.
    low, high : float
        Target interval bounds.

    Returns
    -------
    ndarray
        Normalized values.
    """
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.full_like(values, (low + high) / 2.0, dtype=float)
    return low + (high - low) * (values - vmin) / (vmax - vmin)


def build_coordinate_grid(n_time, n_freq):
    """Build a flattened (N, 2) coordinate array normalized to [-1, 1].

    Column 0 = frequency axis (varies fast), column 1 = time axis (varies slow).

    Parameters
    ----------
    n_time, n_freq : int
        Dimensions of the waterfall.

    Returns
    -------
    coords : ndarray, shape (n_time * n_freq, 2)
    """
    freq_norm = normalize_to_interval(np.arange(n_freq, dtype=float))
    time_norm = normalize_to_interval(np.arange(n_time, dtype=float))
    tt, ff = np.meshgrid(time_norm, freq_norm, indexing="ij")
    coords = np.column_stack([ff.ravel(), tt.ravel()])
    return coords


def smooth_mask(mask, kernel_size=3):
    """Smooth a boolean mask and round to nearest integer.

    This performs morphological smoothing by:
    1. Converting the boolean mask to float (0.0 or 1.0)
    2. Applying a uniform (box) filter of given size
    3. Rounding to nearest integer (0 or 1)
    4. Converting back to boolean

    Use cases:
    - Fill small unflagged gaps within flagged regions (acts like closing)
    - Expand flagged regions slightly to create a more conservative mask
    - Remove isolated single-pixel flags (acts like opening)

    Parameters
    ----------
    mask : ndarray of bool, shape (n_time, n_freq)
        Input boolean mask (True = flagged).
    kernel_size : int or tuple of int
        Size of the uniform filter kernel. If int, uses the same size
        for both dimensions. If tuple (size_time, size_freq), applies
        different smoothing along each axis.
        - kernel_size=3: minimal smoothing (3x3 box)
        - kernel_size=5: moderate smoothing (5x5 box)
        - kernel_size=(1, 5): smooth only along frequency axis

    Returns
    -------
    ndarray of bool, shape (n_time, n_freq)
        Smoothed mask (True = flagged).

    Examples
    --------
    >>> mask = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool)
    >>> smooth_mask(mask, kernel_size=3)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]])

    Notes
    -----
    The smoothing threshold is 0.5 after convolution, meaning a pixel
    becomes flagged if more than half its neighbors (within the kernel)
    are flagged. Adjust kernel_size to control the degree of dilation/erosion.
    """
    # Convert bool to float for filtering
    mask_float = mask.astype(float)

    # Apply uniform filter (box average)
    smoothed = uniform_filter(mask_float, size=kernel_size, mode='constant', cval=0.0)

    # Round to nearest integer: >= 0.5 -> 1, < 0.5 -> 0
    return np.round(smoothed).astype(bool)


def masked_normalized_convolve(field, good, kernel, mode="reflect"):
    """Mask-aware local weighted average of ``field`` under ``kernel``.

    Computes ``(field * good) ⊛ kernel  /  good ⊛ kernel`` — i.e. a normalized
    convolution that ignores pixels where ``good`` is False (already-flagged or
    invalid data). This prevents bright RFI from leaking its power into the
    smoothed field, which a plain convolution would do.

    The normalization by the convolved good-indicator also makes the result a
    true local mean regardless of how many pixels in each footprint are masked,
    so partially-masked footprints near the array edges or near flagged regions
    stay unbiased. Memory stays O(N) — ``scipy.ndimage.convolve`` streams.

    Parameters
    ----------
    field : ndarray, shape (n_time, n_freq)
        The 2D field to smooth (e.g. surface-fit residuals).
    good : ndarray of bool, shape (n_time, n_freq)
        True where the pixel is usable. False pixels contribute nothing.
    kernel : array_like, 2D
        Convolution kernel. Normalization is handled here, so the kernel scale
        does not matter (a box, a diagonal ``np.eye(k)``, or a 1D ``(1, k)`` /
        ``(k, 1)`` line all work).
    mode : str
        Boundary mode forwarded to ``scipy.ndimage.convolve``. Use ``'reflect'``
        (default) so the local mean stays unbiased at the array borders. Do NOT
        use ``'constant'`` with ``cval=0`` here — that biases the mean toward
        zero at the edges (that mode is only appropriate for mask *dilation*).

    Returns
    -------
    convolved : ndarray, shape (n_time, n_freq)
        The normalized local average. Footprints with no good pixels are 0.
    weight : ndarray, shape (n_time, n_freq)
        The convolved good-indicator (``good ⊛ kernel``). ``weight <= 0`` marks
        footprints that contain no usable pixels; the caller should exclude
        those from any threshold test.

    Notes
    -----
    Pure function: neither ``field`` nor ``good`` is mutated.
    """
    kf = np.asarray(kernel, dtype=float)
    field = np.asarray(field, dtype=float)
    good_f = np.asarray(good, dtype=float)
    num = convolve(np.where(good, field, 0.0), kf, mode=mode)
    weight = convolve(good_f, kf, mode=mode)
    convolved = num / np.maximum(weight, _WEIGHT_FLOOR)
    return convolved, weight


def dilate_to_footprint(mask, kernel):
    """Dilate a boolean mask to the geometric support of ``kernel``.

    Any pixel within the kernel's non-zero footprint of a flagged pixel becomes
    flagged. This maps a detection in a convolved image back to the native
    resolution the broad RFI physically occupies: a box kernel dilates to a box,
    a diagonal ``np.eye(k)`` dilates along the diagonal, a 1D line dilates along
    its axis.

    Parameters
    ----------
    mask : ndarray of bool, shape (n_time, n_freq)
        Detection mask (True = flagged).
    kernel : array_like, 2D
        The same kernel used for the convolution; only its non-zero pattern
        (support) is used, so kernel weights are irrelevant.

    Returns
    -------
    ndarray of bool, shape (n_time, n_freq)
        Dilated mask. Pure function: ``mask`` is not mutated.
    """
    support = (np.asarray(kernel) != 0).astype(float)
    dilated = convolve(mask.astype(float), support, mode="constant", cval=0.0)
    return dilated > 0
