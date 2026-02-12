import numpy as np
import h5py


def load_waterfall(filepath, group="sdr"):
    """Load waterfall data from an HDF5 observation file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    group : str
        HDF5 group containing the data.

    Returns
    -------
    waterfall : ndarray, shape (n_time, n_freq)
    freqs : ndarray, shape (n_freq,)
        Frequency axis in MHz.
    times : ndarray, shape (n_time,)
        Time axis in seconds.
    """
    with h5py.File(filepath, "r") as f:
        g = f[group]
        waterfall = g[f"{group}_waterfall"][:]
        freqs = g[f"{group}_freqs"][:]
        times = g[f"{group}_times"][:]
    return waterfall, freqs, times


def validate_waterfall(waterfall):
    """Check that waterfall data is suitable for log-space fitting.

    Raises ValueError if problems are found.

    Parameters
    ----------
    waterfall : ndarray

    Returns
    -------
    dict
        Summary statistics: shape, min, max, has_nan, has_zero, has_negative.
    """
    info = {
        "shape": waterfall.shape,
        "min": float(np.nanmin(waterfall)),
        "max": float(np.nanmax(waterfall)),
        "has_nan": bool(np.any(np.isnan(waterfall))),
        "has_zero": bool(np.any(waterfall == 0)),
        "has_negative": bool(np.any(waterfall < 0)),
    }

    problems = []
    if info["has_nan"]:
        problems.append("Data contains NaN values")
    if info["has_zero"]:
        problems.append("Data contains zero values (log10 undefined)")
    if info["has_negative"]:
        problems.append("Data contains negative values (log10 undefined)")
    if waterfall.ndim != 2:
        problems.append(f"Expected 2D array, got {waterfall.ndim}D")

    if problems:
        raise ValueError("Waterfall validation failed:\n  " + "\n  ".join(problems))

    return info
