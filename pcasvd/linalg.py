"""From-scratch numerical kernels behind the PCA solvers.

None of these functions call ``np.linalg.svd``, ``np.linalg.eig`` or
``np.linalg.eigh``. The only LAPACK routine they use is the Householder QR
factorisation (``np.linalg.qr``), as an orthonormalisation primitive; every
eigen/singular decomposition is computed here with matrix products and plane
rotations.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IndexArray = NDArray[np.intp]

EPS = float(np.finfo(np.float64).eps)


def sign_flip(components: Array) -> Array:
    """Return ``components`` with each row's largest-magnitude entry made positive.

    Singular/eigen vectors are only defined up to sign; this is the same
    deterministic convention as scikit-learn's ``svd_flip(u_based_decision=False)``.
    """
    rows = np.arange(components.shape[0])
    signs = np.sign(components[rows, np.argmax(np.abs(components), axis=1)])
    signs[signs == 0] = 1.0
    return components * signs[:, None]


@lru_cache(maxsize=32)
def _round_robin(n: int) -> tuple[tuple[IndexArray, IndexArray], ...]:
    """Split all column pairs of ``0..n-1`` into ``n-1`` rounds of disjoint pairs.

    Circle method of a round-robin tournament: within a round no index appears
    twice, so all rotations of a round commute and can be applied at once.
    """
    players = list(range(n)) + ([-1] if n % 2 else [])  # -1 is a "bye"
    m = len(players)
    rounds = []
    for _ in range(m - 1):
        pairs = [
            (min(a, b), max(a, b))
            for a, b in zip(players[: m // 2], players[::-1][: m // 2])
            if a >= 0 and b >= 0
        ]
        if pairs:
            p, q = np.array(pairs, dtype=np.intp).T
            rounds.append((p, q))
        players = [players[0], players[-1], *players[1:-1]]
    return tuple(rounds)


def _one_sided_jacobi(a: Array, tol: float, max_sweeps: int) -> tuple[Array, Array]:
    """Orthogonalise the columns of ``a`` with Jacobi plane rotations (Hestenes).

    Returns ``(w, v)`` with ``a @ v = w``, ``v`` orthogonal and the columns of
    ``w`` mutually orthogonal, i.e. ``a = (w / |w|) diag(|w|) v.T`` is an SVD.
    """
    # Columns are stored as contiguous rows (wt = w.T, vt = v.T): ~3x faster gathers.
    wt = np.array(a.T, dtype=np.float64, order="C")
    vt = np.eye(a.shape[1])
    for _ in range(max_sweeps):
        rotated = False
        for p, q in _round_robin(a.shape[1]):
            wp, wq = wt[p], wt[q]
            alpha = np.einsum("ij,ij->i", wp, wp)
            beta = np.einsum("ij,ij->i", wq, wq)
            gamma = np.einsum("ij,ij->i", wp, wq)
            # Rotate only pairs whose cosine |<wp, wq>| / (|wp| |wq|) exceeds tol.
            active = np.abs(gamma) > tol * np.sqrt(alpha * beta)
            if not active.any():
                continue
            rotated = True
            p, q, wp, wq = p[active], q[active], wp[active], wq[active]
            zeta = (beta[active] - alpha[active]) / (2.0 * gamma[active])
            t = np.where(zeta >= 0, 1.0, -1.0) / (np.abs(zeta) + np.hypot(1.0, zeta))
            c = 1.0 / np.hypot(1.0, t)[:, None]  # rotation angle that zeroes <wp, wq>
            s = c * t[:, None]
            wt[p], wt[q] = c * wp - s * wq, s * wp + c * wq
            vp, vq = vt[p], vt[q]
            vt[p], vt[q] = c * vp - s * vq, s * vp + c * vq
        if not rotated:
            return wt.T, vt.T
    warnings.warn(f"Jacobi SVD did not converge in {max_sweeps} sweeps.", RuntimeWarning, stacklevel=3)
    return wt.T, vt.T


def jacobi_svd(a: Array, *, tol: float | None = None, max_sweeps: int = 50) -> tuple[Array, Array, Array]:
    """Thin SVD ``a = u @ diag(s) @ vt`` by QR-preconditioned one-sided Jacobi.

    The matrix is first reduced to a square ``min(m, n)`` triangular factor by a
    QR factorisation, whose columns are then orthogonalised by cyclic Jacobi
    rotations applied in parallel rounds. ``vt`` is always orthonormal; columns
    of ``u`` belonging to exactly-zero singular values are set to zero.

    Returns ``(u, s, vt)`` with ``s`` sorted in decreasing order.
    """
    a = np.asarray(a, dtype=np.float64)
    m, n = a.shape
    tol = min(m, n) * EPS if tol is None else tol
    if m >= n:
        q, r = np.linalg.qr(a)  # a = q r
        w, v = _one_sided_jacobi(r, tol, max_sweeps)  # r v = w
        s = np.linalg.norm(w, axis=0)
        u, vt = q @ _normalise_columns(w, s), v.T
    else:
        q, r = np.linalg.qr(a.T)  # a = r.T q.T
        w, v = _one_sided_jacobi(r.T, tol, max_sweeps)  # r.T v = w
        s = np.linalg.norm(w, axis=0)
        u, vt = _normalise_columns(w, s), (q @ v).T
    order = np.argsort(-s, kind="stable")
    return u[:, order], s[order], vt[order]


def _normalise_columns(w: Array, norms: Array) -> Array:
    return np.divide(w, norms, out=np.zeros_like(w), where=norms > 0)


def power_iteration_eigh(
    a: Array, k: int, *, tol: float = 1e-10, max_iter: int = 10_000, seed: int = 0
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Top-``k`` eigenpairs of a symmetric positive semi-definite matrix.

    Power iteration with deflation by orthogonal projection: eigenpair ``i`` is
    the dominant eigenpair of ``P A P`` with ``P = I - V_i V_i^T`` (``V_i`` the
    eigenvectors already found). Eigenpair ``i`` stops when the residual
    ``||A v - lambda_i v||`` drops below ``tol * lambda_i`` (relative accuracy,
    even for small eigenvalues) or below the rounding floor ``n * eps * lambda_1``
    (numerically null eigenvalues).

    Returns eigenvalues (decreasing), eigenvectors as columns, and the number of
    iterations spent on each eigenpair.
    """
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.zeros(k)
    vecs = np.zeros((n, k))
    iters = np.zeros(k, dtype=np.int64)
    n_unconverged = 0
    for i in range(k):
        basis = vecs[:, :i]
        v = rng.standard_normal(n)
        v -= basis @ (basis.T @ v)
        v /= np.linalg.norm(v)
        lam, converged = 0.0, False
        for it in range(1, max_iter + 1):
            w = a @ v
            w -= basis @ (basis.T @ w)  # deflation: stay orthogonal to found eigenvectors
            lam = float(v @ w)  # Rayleigh quotient
            lam_max = vals[0] if i else abs(lam)
            if np.linalg.norm(w - lam * v) <= tol * abs(lam) + n * EPS * lam_max:
                converged = True
                break
            v = w / np.linalg.norm(w)
        n_unconverged += not converged
        vals[i], vecs[:, i], iters[i] = lam, v, it
    if n_unconverged:
        warnings.warn(
            f"Power iteration: {n_unconverged}/{k} eigenpairs hit max_iter={max_iter}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return vals, vecs, iters


def randomized_svd(
    a: Array, k: int, *, n_oversamples: int = 10, n_power_iter: int = 4, seed: int = 0
) -> tuple[Array, Array, Array]:
    """Rank-``k`` truncated SVD by randomized subspace iteration.

    Halko, Martinsson & Tropp (2011): Gaussian sketch ``Y = A Omega`` of
    ``k + n_oversamples`` columns, ``n_power_iter`` QR-stabilised subspace
    iterations (Alg. 4.4), then the exact SVD of the small projected matrix
    ``B = Q^T A`` (Alg. 5.1), computed with :func:`jacobi_svd`.
    """
    m, n = a.shape
    sketch = min(k + n_oversamples, m, n)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(a @ rng.standard_normal((n, sketch)))
    for _ in range(n_power_iter):
        z, _ = np.linalg.qr(a.T @ q)
        q, _ = np.linalg.qr(a @ z)
    u_b, s, vt = jacobi_svd(q.T @ a)
    return (q @ u_b)[:, :k], s[:k], vt[:k]
