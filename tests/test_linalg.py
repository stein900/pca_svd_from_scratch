"""Accuracy of the from-scratch kernels against LAPACK and known spectra."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import matrix_with_spectrum

from pcasvd.linalg import _round_robin, jacobi_svd, power_iteration_eigh, randomized_svd, sign_flip
from pcasvd.metrics import max_abs_diff_up_to_sign


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_round_robin_visits_every_pair_once_in_disjoint_rounds(n: int) -> None:
    rounds = _round_robin(n)
    pairs = sorted((int(i), int(j)) for p, q in rounds for i, j in zip(p, q))
    assert pairs == [(i, j) for i in range(n) for j in range(i + 1, n)]
    for p, q in rounds:
        assert len(set(p) | set(q)) == 2 * len(p)


@pytest.mark.parametrize("shape", [(40, 40), (70, 30), (30, 70)])
def test_jacobi_svd_matches_lapack(shape: tuple[int, int]) -> None:
    a = np.random.default_rng(1).standard_normal(shape)
    u, s, vt = jacobi_svd(a)
    k = min(shape)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-12)
    np.testing.assert_allclose((u * s) @ vt, a, atol=1e-12)
    np.testing.assert_allclose(u.T @ u, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(vt @ vt.T, np.eye(k), atol=1e-12)


def test_jacobi_svd_rank_deficient_keeps_orthonormal_right_factor() -> None:
    a = matrix_with_spectrum(30, 20, np.array([5.0, 3.0, 1.0]))
    u, s, vt = jacobi_svd(a)
    np.testing.assert_allclose(s[:3], [5.0, 3.0, 1.0], rtol=1e-12)
    assert np.all(s[3:] < 1e-13)
    np.testing.assert_allclose(vt @ vt.T, np.eye(20), atol=1e-12)
    np.testing.assert_allclose((u * s) @ vt, a, atol=1e-12)


def test_power_iteration_recovers_known_spectrum() -> None:
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.standard_normal((50, 50)))
    lam = 10.0 * 0.7 ** np.arange(50)
    vals, vecs, iters = power_iteration_eigh((q * lam) @ q.T, 8, tol=1e-12)
    np.testing.assert_allclose(vals, lam[:8], rtol=1e-11)
    np.testing.assert_allclose(np.abs(np.sum(vecs * q[:, :8], axis=0)), 1.0, atol=1e-10)
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(8), atol=1e-12)
    assert np.all(iters < 10_000)


def test_power_iteration_warns_when_budget_is_exhausted() -> None:
    lam = np.array([1.0, 1.0 - 1e-6, 0.5])  # tiny gap: convergence needs ~10^7 iterations
    q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))
    with pytest.warns(RuntimeWarning, match="max_iter"):
        power_iteration_eigh((q * lam) @ q.T, 1, tol=1e-14, max_iter=50)


@pytest.mark.parametrize("shape", [(200, 80), (80, 200)])
def test_randomized_svd_on_known_fast_decaying_spectrum(shape: tuple[int, int]) -> None:
    s_true = 0.5 ** np.arange(min(shape))
    a = matrix_with_spectrum(*shape, s_true, seed=4)
    u, s, vt = randomized_svd(a, 10, n_oversamples=10, n_power_iter=2)
    np.testing.assert_allclose(s, s_true[:10], rtol=1e-10)
    vt_ref = np.linalg.svd(a, full_matrices=False)[2][:10]
    assert max_abs_diff_up_to_sign(vt, vt_ref) < 1e-8
    np.testing.assert_allclose(u.T @ u, np.eye(10), atol=1e-12)


def test_randomized_svd_is_near_optimal_on_slowly_decaying_spectrum() -> None:
    s_true = 1.0 / (1.0 + np.arange(150))
    a = matrix_with_spectrum(300, 150, s_true, seed=5)
    k = 20
    u, s, vt = randomized_svd(a, k, n_power_iter=6)
    error = np.sum((a - (u * s) @ vt) ** 2)
    optimum = np.sum(s_true[k:] ** 2)  # Eckart-Young lower bound
    assert optimum <= error <= 1.01 * optimum


def test_sign_flip_is_deterministic() -> None:
    rng = np.random.default_rng(6)
    v = rng.standard_normal((5, 9))
    flipped = sign_flip(v)
    np.testing.assert_array_equal(sign_flip(v * rng.choice([-1.0, 1.0], size=(5, 1))), flipped)
    assert np.all(flipped[np.arange(5), np.argmax(np.abs(flipped), axis=1)] > 0)
