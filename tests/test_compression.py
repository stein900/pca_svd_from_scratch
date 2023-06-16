"""Truncated-SVD compression: Eckart-Young optimality, storage and PSNR."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import matrix_with_spectrum

from pcasvd.compression import low_rank_approx, optimal_squared_error, psnr, storage_ratio
from pcasvd.linalg import randomized_svd


@pytest.mark.parametrize("k", [1, 5, 29, 30])
def test_truncation_error_equals_tail_energy(k: int) -> None:
    a = np.random.default_rng(0).standard_normal((40, 30))
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    error = np.sum((a - low_rank_approx(u, s, vt, k)) ** 2)
    np.testing.assert_allclose(error, optimal_squared_error(s, k), rtol=1e-10, atol=1e-20)


def test_no_rank_k_projection_beats_truncated_svd() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 40))
    k = 8
    optimum = optimal_squared_error(np.linalg.svd(a, compute_uv=False), k)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.standard_normal((40, k)))
        assert np.sum((a - a @ q @ q.T) ** 2) >= optimum


def test_randomized_low_rank_approx_is_near_optimal() -> None:
    s_true = 200.0 * np.arange(1, 121, dtype=float) ** -1.5  # power-law decay, like natural images
    a = matrix_with_spectrum(160, 120, s_true, seed=2)
    k = 15
    u, s, vt = randomized_svd(a, k, n_power_iter=4)
    error = np.sum((a - low_rank_approx(u, s, vt, k)) ** 2)
    assert error <= 1.001 * optimal_squared_error(s_true, k)


def test_storage_ratio() -> None:
    assert storage_ratio((768, 1024), 20) == pytest.approx(20 * (768 + 1024 + 1) / (768 * 1024))
    assert storage_ratio((10, 10), 10) > 1.0  # full rank costs more than the raw image


def test_psnr() -> None:
    ref = np.full((4, 4), 100.0)
    assert psnr(ref, ref) == float("inf")
    assert psnr(np.zeros((2, 2)), np.full((2, 2), 255.0)) == pytest.approx(0.0)
    assert psnr(ref, ref + 10.0) == pytest.approx(10 * np.log10(255.0**2 / 100.0))
    assert psnr(np.full((2, 2), 255.0), np.full((2, 2), 300.0)) == float("inf")  # clipped to range
