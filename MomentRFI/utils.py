import numpy as np


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
