"""Polynomial surface-fitting primitives.

Two equivalent backends, both fitting/evaluating a 2D polynomial surface over
coordinates normalized to [-1, 1]:

- **MomentEmu path** (`_fit_surface` / `_evaluate_surface`): constant-memory,
  rebuilds the monomial basis on every call.
- **Cached-basis fast path** (`_build_phi` + `_fit_surface_from_phi` /
  `_evaluate_surface_from_phi`): the coordinate grid is fixed for a whole fit, so
  the Vandermonde `Phi` is built once and reused across all sigma-clip iterations.
  This eliminates the redundant monomial recomputation (~5x faster) while
  preserving the exact batched accumulation order — and therefore **bit-identical
  results** — and full-grid evaluation each iteration (so previously-flagged good
  pixels can still be re-admitted). Costs O(N*D) memory; the caller guards it.
"""
import numpy as np

from MomentEmu.PolyEmu import (
    compute_moments_vector_output_batched,
    evaluate_emulator_batched,
)
from MomentEmu.MomentEmu import solve_emulator_coefficients


def _fit_surface(coords, values, multi_indices, batch_size=200_000):
    """Fit polynomial surface via MomentEmu's batched moment method.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
    values : ndarray, shape (N,)
    multi_indices : ndarray
    batch_size : int

    Returns
    -------
    coeffs : ndarray, shape (D, 1)
    """
    Y = values.reshape(-1, 1)
    M, nu = compute_moments_vector_output_batched(
        coords, Y, multi_indices, batch_size=batch_size
    )
    return solve_emulator_coefficients(M, nu)


def _evaluate_surface(coords, coeffs, multi_indices, batch_size=200_000):
    """Evaluate polynomial surface via MomentEmu's batched evaluator.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
    coeffs : ndarray, shape (D, 1)
    multi_indices : ndarray
    batch_size : int

    Returns
    -------
    result : ndarray, shape (N,)
    """
    return evaluate_emulator_batched(
        coords, coeffs, multi_indices, batch_size=batch_size
    ).ravel()


def _power_tables(coords, degree_freq, degree_time):
    """Per-axis monomial power tables, matching ``X[:, i] ** deg`` exactly.

    Returns ``(freq_pow, time_pow)`` where ``freq_pow[a] = coords[:, 0] ** a`` and
    ``time_pow[b] = coords[:, 1] ** b``. Index 0 is a shared ones-array. Uses the
    same ``**`` op as MomentEmu (not a cumulative product) to stay bit-identical.
    """
    n = coords.shape[0]
    ones = np.ones(n, dtype=coords.dtype)
    f = coords[:, 0]
    t = coords[:, 1]
    freq_pow = [ones] + [f ** d for d in range(1, degree_freq + 1)]
    time_pow = [ones] + [t ** d for d in range(1, degree_time + 1)]
    return freq_pow, time_pow


def _phi_block(freq_pow, time_pow, mi, sel, dtype):
    """Build a Phi block (len(sel) x D) for rows ``sel`` from the power tables.

    Column ``j = (a, b)`` is ``freq_pow[a][sel] * time_pow[b][sel]`` — identical to
    MomentEmu's ``ones * freq^a * time^b`` (``1 * x == x`` in IEEE-754).
    """
    fp = [p[sel] for p in freq_pow]
    tp = [p[sel] for p in time_pow]
    phi = np.empty((fp[0].shape[0], len(mi)), dtype=dtype)
    for j, (a, b) in enumerate(mi):
        phi[:, j] = fp[a] * tp[b]
    return phi


def _build_phi(coords, mi, degree_freq, degree_time):
    """Build the full Vandermonde ``Phi`` (N x D) once, bit-identical to MomentEmu.

    Column ``j = (a, b)`` is ``coords[:,0]**a * coords[:,1]**b``, matching
    ``evaluate_monomials_lazy``'s ``ones * freq^a * time^b`` build order.
    """
    freq_pow, time_pow = _power_tables(coords, degree_freq, degree_time)
    return _phi_block(freq_pow, time_pow, mi, slice(None), coords.dtype)


def _fit_surface_from_phi(Phi, idx, Y, batch_size):
    """Fit coeffs from a prebuilt Phi, matching MomentEmu's batched moments.

    Accumulates ``M += Phi[good].T @ Phi[good]`` and ``nu += Phi[good].T @ Y`` over
    the good rows ``idx`` in the same ``batch_size`` chunks and order as MomentEmu,
    then divides by the good count — so the result is bit-identical to
    ``_fit_surface``.
    """
    D = Phi.shape[1]
    dtype = Phi.dtype
    M = np.zeros((D, D), dtype=dtype)
    nu = np.zeros((D, 1), dtype=dtype)
    n = idx.shape[0]
    for s in range(0, n, batch_size):
        gi = idx[s:s + batch_size]
        Pb = Phi[gi]
        M += Pb.T @ Pb
        nu += Pb.T @ Y[gi]
    M /= n
    nu /= n
    return solve_emulator_coefficients(M, nu)


def _evaluate_surface_from_phi(Phi, coeffs, batch_size):
    """Evaluate the surface over the full grid from a prebuilt Phi.

    Bit-identical to ``_evaluate_surface`` (same contiguous ``batch_size`` chunks
    and ``Phi @ coeffs`` per chunk).
    """
    n = Phi.shape[0]
    out = np.empty(n, dtype=Phi.dtype)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out[s:e] = (Phi[s:e] @ coeffs)[:, 0]
    return out
