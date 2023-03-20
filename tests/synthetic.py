"""Random matrices with a prescribed spectrum, shared by the tests."""

from __future__ import annotations

import numpy as np


def matrix_with_spectrum(m: int, n: int, singular_values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random ``m x n`` matrix ``U diag(s) V^T`` with Haar-distributed orthonormal ``U`` and ``V``."""
    rng = np.random.default_rng(seed)
    r = len(singular_values)
    u, _ = np.linalg.qr(rng.standard_normal((m, r)))
    v, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return (u * singular_values) @ v.T


def low_rank_dataset(n: int, p: int, seed: int = 0) -> np.ndarray:
    """Data whose covariance has a geometrically decaying, well-separated spectrum."""
    s = 100.0 * 0.8 ** np.arange(min(n, p))
    return matrix_with_spectrum(n, p, s, seed) + 5.0
