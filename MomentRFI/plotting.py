import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_waterfall(waterfall, freqs=None, times=None, ax=None, title="Waterfall",
                   cmap="viridis", log=True, **kwargs):
    """Plot a waterfall image.

    Parameters
    ----------
    waterfall : ndarray, shape (n_time, n_freq)
    freqs : ndarray, optional
    times : ndarray, optional
    ax : matplotlib Axes, optional
    title : str
    cmap : str
    log : bool
        Use LogNorm for colorscale.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    extent = None
    if freqs is not None and times is not None:
        extent = [freqs[0], freqs[-1], times[-1], times[0]]

    norm = kwargs.pop("norm", LogNorm() if log else None)
    im = ax.imshow(waterfall, aspect="auto", extent=extent, cmap=cmap, norm=norm,
                   interpolation="none", **kwargs)
    ax.set_title(title)
    if freqs is not None:
        ax.set_xlabel("Frequency [MHz]")
    if times is not None:
        ax.set_ylabel("Time [s]")
    plt.colorbar(im, ax=ax, label="Power")
    return ax


def plot_mask(mask, freqs=None, times=None, ax=None, title="RFI Mask"):
    """Plot the boolean RFI mask.

    Parameters
    ----------
    mask : ndarray of bool, shape (n_time, n_freq)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    extent = None
    if freqs is not None and times is not None:
        extent = [freqs[0], freqs[-1], times[-1], times[0]]

    ax.imshow(mask.astype(float), aspect="auto", extent=extent, cmap="Reds",
              vmin=0, vmax=1, interpolation="none")
    n_flagged = mask.sum()
    frac = n_flagged / mask.size
    ax.set_title(f"{title}  ({n_flagged} pixels, {frac:.2%})")
    if freqs is not None:
        ax.set_xlabel("Frequency [MHz]")
    if times is not None:
        ax.set_ylabel("Time [s]")
    return ax


def plot_residuals(residuals, freqs=None, times=None, ax=None, title="Residuals (log10)",
                   vmin=None, vmax=None):
    """Plot residuals (data - surface) in log10 space.

    Parameters
    ----------
    residuals : ndarray, shape (n_time, n_freq)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    extent = None
    if freqs is not None and times is not None:
        extent = [freqs[0], freqs[-1], times[-1], times[0]]

    if vmin is None:
        vmin = np.percentile(residuals, 1)
    if vmax is None:
        vmax = np.percentile(residuals, 99)

    im = ax.imshow(residuals, aspect="auto", extent=extent, cmap="RdBu_r",
                   vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title)
    if freqs is not None:
        ax.set_xlabel("Frequency [MHz]")
    if times is not None:
        ax.set_ylabel("Time [s]")
    plt.colorbar(im, ax=ax, label="log10 residual")
    return ax


def plot_convergence(history, ax=None):
    """Plot convergence diagnostics for both phases.

    Parameters
    ----------
    history : dict
        The .history attribute from IterativeSurfaceFitter.
    """
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = ax

    # Sigma evolution
    ax0 = axes[0]
    if history["phase1"]:
        iters1 = [h["iteration"] for h in history["phase1"]]
        sigmas1 = [h["sigma"] for h in history["phase1"]]
        ax0.plot(iters1, sigmas1, "o-", label="Phase 1")
    if history["phase2"]:
        iters2 = [h["iteration"] for h in history["phase2"]]
        sigmas2 = [h["sigma_used"] for h in history["phase2"]]
        ax0.plot(iters2, sigmas2, "s-", label="Phase 2")
        if history["phase2"][0].get("sigma_floor"):
            ax0.axhline(history["phase2"][0]["sigma_floor"], ls="--", color="gray",
                        label="Sigma floor")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Sigma (MAD)")
    ax0.set_title("Sigma Evolution")
    ax0.legend()

    # Flag fraction
    ax1 = axes[1]
    if history["phase1"]:
        fracs1 = [h["flag_fraction"] * 100 for h in history["phase1"]]
        ax1.plot(iters1, fracs1, "o-", label="Phase 1")
    if history["phase2"]:
        fracs2 = [h["flag_fraction"] * 100 for h in history["phase2"]]
        ax1.plot(iters2, fracs2, "s-", label="Phase 2")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Flagged (%)")
    ax1.set_title("Flag Fraction")
    ax1.legend()

    # Changed fraction (convergence)
    ax2 = axes[2]
    if history["phase1"]:
        changed1 = [h["changed_fraction"] * 100 for h in history["phase1"]]
        ax2.semilogy(iters1, changed1, "o-", label="Phase 1")
    if history["phase2"]:
        changed2 = [h["changed_fraction"] * 100 for h in history["phase2"]]
        ax2.semilogy(iters2, changed2, "s-", label="Phase 2")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Changed pixels (%)")
    ax2.set_title("Convergence")
    ax2.legend()

    plt.tight_layout()
    return axes


def plot_time_averaged_spectrum(waterfall, freqs=None, mask=None, ax=None,
                                 title="Time-Averaged Spectrum", log=True):
    """Plot time-averaged spectrum, optionally comparing original vs cleaned.

    Parameters
    ----------
    waterfall : ndarray, shape (n_time, n_freq)
        Original waterfall data.
    freqs : ndarray, optional
        Frequency axis in MHz.
    mask : ndarray of bool, optional, shape (n_time, n_freq)
        RFI mask. If provided, plots both original and cleaned spectra.
    ax : matplotlib Axes, optional
    title : str
    log : bool
        Use log scale for y-axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Compute time-averaged spectrum
    spectrum_original = np.mean(waterfall, axis=0)

    if freqs is None:
        freqs = np.arange(len(spectrum_original))
        xlabel = "Channel"
    else:
        xlabel = "Frequency [MHz]"

    # Plot original spectrum
    ax.plot(freqs, spectrum_original, '-', alpha=0.7, label="Original", linewidth=1.5)

    # If mask provided, compute and plot cleaned spectrum
    if mask is not None:
        waterfall_cleaned = waterfall.astype(float).copy()
        waterfall_cleaned[mask] = np.nan
        spectrum_cleaned = np.nanmean(waterfall_cleaned, axis=0)
        ax.plot(freqs, spectrum_cleaned, '-', alpha=0.7, label="Cleaned", linewidth=1.5)
        ax.legend()

        # Add statistics
        n_flagged = mask.sum()
        frac = n_flagged / mask.size
        ax.text(0.02, 0.98, f"Flagged: {frac:.2%}", transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if log:
        ax.set_yscale('log')

    return ax


def plot_summary(waterfall, fitter, freqs=None, times=None):
    """Five-panel summary: original, surface, flagged, residuals, mask.

    Parameters
    ----------
    waterfall : ndarray, shape (n_time, n_freq)
    fitter : IterativeSurfaceFitter (after .fit())
    freqs, times : ndarray, optional
    """
    fig = plt.figure(figsize=(18, 11))
    # Top row: 3 panels
    ax0 = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2)
    ax2 = fig.add_subplot(2, 3, 3)
    # Bottom row: 2 panels, centered
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5)
    fig.delaxes(fig.add_subplot(2, 3, 6))

    # Use percentile-based colour limits so bright RFI spikes don't
    # anchor vmax and compress the rest of the data into darkness.
    vmin = np.nanpercentile(waterfall, 1)
    vmax = np.nanpercentile(waterfall, 99)
    robust_norm = LogNorm(vmin=vmin, vmax=vmax)

    plot_waterfall(waterfall, freqs, times, ax=ax0, title="Original Waterfall",
                   norm=robust_norm)
    plot_waterfall(fitter.surface, freqs, times, ax=ax1, title="Fitted Surface",
                   norm=LogNorm(vmin=vmin, vmax=vmax))

    # Flagged waterfall: flagged pixels blanked (NaN) to show missing data.
    # Re-use the same colour limits so all three waterfall panels are comparable.
    flagged = waterfall.astype(float).copy()
    flagged[fitter.mask] = np.nan
    plot_waterfall(flagged, freqs, times, ax=ax2, title="Flagged Waterfall",
                   norm=LogNorm(vmin=vmin, vmax=vmax))

    plot_residuals(fitter.residuals, freqs, times, ax=ax3)
    plot_mask(fitter.mask, freqs, times, ax=ax4)

    plt.tight_layout()
    return fig
