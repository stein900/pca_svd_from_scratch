"""Truncated-SVD image compression and the Eckart-Young optimality bound."""

from __future__ import annotations

import numpy as np

from .linalg import Array


def low_rank_approx(u: Array, s: Array, vt: Array, k: int) -> Array:
    """Rank-``k`` reconstruction ``U_k diag(s_k) V_k^T`` from (possibly longer) SVD factors."""
    return (u[:, :k] * s[:k]) @ vt[:k]


def optimal_squared_error(s: Array, k: int) -> float:
    """Eckart-Young: ``min_{rank(B)<=k} ||A - B||_F^2 = sum_{i>k} sigma_i^2``."""
    return float(np.sum(s[k:] ** 2))


def storage_ratio(shape: tuple[int, int], k: int) -> float:
    """Numbers stored by a rank-``k`` factorisation, ``k (m + n + 1)``, over ``m n`` pixels."""
    m, n = shape
    return k * (m + n + 1) / (m * n)


def psnr(reference: Array, approx: Array, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (``approx`` is clipped to the valid range first)."""
    mse = np.mean((np.asarray(reference, dtype=np.float64) - np.clip(approx, 0.0, data_range)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(data_range**2 / mse))
