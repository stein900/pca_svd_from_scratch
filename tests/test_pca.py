"""The PCA estimator against scikit-learn and against its own invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from pcasvd import PCA, SOLVERS
from pcasvd.datasets import load_olivetti

# Absolute tolerance on component entries / scores: exact solvers agree with
# LAPACK to rounding error; the iterative ones to their convergence tolerance.
TOL = {"svd": 1e-12, "covariance_eigh": 1e-10, "jacobi": 1e-12, "power": 1e-8, "randomized": 1e-8}
# Orthonormality: rounding level for direct solvers, ~tol of the power solver.
ORTHO_TOL = {"svd": 1e-12, "covariance_eigh": 1e-10, "jacobi": 1e-12, "power": 1e-8, "randomized": 1e-12}


@pytest.mark.parametrize("solver", SOLVERS)
def test_matches_sklearn(dataset: np.ndarray, solver: str) -> None:
    ours = PCA(10, solver=solver).fit(dataset)
    ref = SkPCA(10, svd_solver="full").fit(dataset)
    atol = TOL[solver]
    np.testing.assert_allclose(ours.mean_, ref.mean_, atol=1e-12)
    # Same sign convention as scikit-learn, so components are compared directly.
    np.testing.assert_allclose(ours.components_, ref.components_, atol=atol)
    np.testing.assert_allclose(ours.singular_values_, ref.singular_values_, rtol=atol)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=atol)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=atol)
    np.testing.assert_allclose(ours.transform(dataset), ref.transform(dataset), atol=100 * atol)


@pytest.mark.parametrize("solver", SOLVERS)
def test_components_are_orthonormal(dataset: np.ndarray, solver: str) -> None:
    pca = PCA(10, solver=solver).fit(dataset)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(10), atol=ORTHO_TOL[solver])


@pytest.mark.parametrize("solver", SOLVERS)
def test_full_rank_round_trip_and_variance_sums(dataset: np.ndarray, solver: str) -> None:
    pca = PCA(solver=solver).fit(dataset)
    k = min(dataset.shape)
    assert pca.components_.shape == (k, dataset.shape[1])
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(k), atol=ORTHO_TOL[solver])
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(dataset)), dataset, atol=1e-9)
    total = dataset.var(axis=0, ddof=1).sum()
    np.testing.assert_allclose(pca.explained_variance_.sum(), total, rtol=1e-10)
    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, rtol=1e-10)
    assert np.all(np.diff(pca.explained_variance_) <= 1e-10 * total)


def test_whitening_matches_sklearn(dataset: np.ndarray) -> None:
    ours = PCA(5, whiten=True).fit(dataset)
    ref = SkPCA(5, whiten=True, svd_solver="full").fit(dataset)
    z = ours.transform(dataset)
    np.testing.assert_allclose(z, ref.transform(dataset), atol=1e-10)
    np.testing.assert_allclose(z.var(axis=0, ddof=1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(ours.inverse_transform(z), ref.inverse_transform(z), atol=1e-10)


def test_fit_transform_equals_fit_then_transform(dataset: np.ndarray) -> None:
    pca = PCA(4)
    np.testing.assert_allclose(pca.fit_transform(dataset), pca.transform(dataset), atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, X",
    [
        ({"n_components": 0}, np.ones((5, 3))),
        ({"n_components": 4}, np.ones((5, 3))),
        ({"solver": "qr"}, np.ones((5, 3))),
        ({}, np.ones((1, 3))),
        ({}, np.ones(5)),
    ],
)
def test_invalid_input_raises(kwargs: dict, X: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PCA(**kwargs).fit(X)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        PCA().transform(np.ones((3, 2)))


FACES_CACHE = Path(__file__).resolve().parents[1] / "data" / "olivetti_py3.pkz"


@pytest.mark.skipif(not FACES_CACHE.exists(), reason="Olivetti faces not cached yet (run demo.py once)")
@pytest.mark.parametrize("solver", ["svd", "covariance_eigh", "jacobi"])
def test_eigenfaces_match_sklearn_on_all_nonzero_components(solver: str) -> None:
    X = load_olivetti(FACES_CACHE.parent).data  # 400 x 4096: the Gram-trick regime
    k = 399  # centred data has rank n - 1
    ours = PCA(k, solver=solver).fit(X)
    ref = SkPCA(k, svd_solver="full").fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-9)
    np.testing.assert_allclose(ours.components_, ref.components_, atol=1e-9)
