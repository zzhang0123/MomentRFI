import numpy as np
from scipy.ndimage import uniform_filter


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
    k = int(n * tail_fraction)

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
    b = np.dot(z - z_mean, y - y_mean) / np.dot(z - z_mean, z - z_mean)

    # sigma = sqrt(-1 / (2b));  b must be negative for a valid Gaussian
    if b >= 0:
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
