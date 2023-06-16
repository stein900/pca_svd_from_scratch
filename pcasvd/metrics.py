"""Evaluation helpers: reconstructions, agreement with a reference PCA, 1-NN accuracy."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import Array


class FittedPCA(Protocol):
    """Fitted attributes shared by :class:`pcasvd.PCA` and ``sklearn.decomposition.PCA``."""

    components_: Array
    explained_variance_: Array
    mean_: Array
    n_components_: int


def reconstruct(pca: FittedPCA, X: ArrayLike, k: int) -> Array:
    """Project ``X`` on the first ``k`` principal axes of a fitted model and map it back."""
    vk = pca.components_[:k]
    return ((np.asarray(X, dtype=np.float64) - pca.mean_) @ vk.T) @ vk + pca.mean_


def reconstruction_mse(pca: FittedPCA, X: ArrayLike, k: int) -> float:
    """Mean squared reconstruction error per feature with ``k`` components."""
    X = np.asarray(X, dtype=np.float64)
    return float(np.mean((X - reconstruct(pca, X, k)) ** 2))


def components_for_variance(ratios: Array, threshold: float) -> int:
    """Smallest ``k`` whose cumulative explained-variance ratio reaches ``threshold``."""
    return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)


def max_abs_diff_up_to_sign(a: Array, b: Array) -> float:
    """Largest entry-wise difference between matching rows of ``a`` and ``b``, each row compared up to sign."""
    plus = np.max(np.abs(a - b), axis=1)
    minus = np.max(np.abs(a + b), axis=1)
    return float(np.max(np.minimum(plus, minus)))


def agreement(model: FittedPCA, reference: FittedPCA, X: Array) -> dict[str, float]:
    """Accuracy of a fitted model against an exact reference fitted on the same ``X``."""
    k = model.n_components_
    ref_variance = reference.explained_variance_[:k]
    return {
        "max_abs_component_diff": max_abs_diff_up_to_sign(model.components_, reference.components_[:k]),
        "max_rel_explained_variance_err": float(np.max(np.abs(model.explained_variance_ - ref_variance) / ref_variance)),
        "reconstruction_mse": reconstruction_mse(model, X, k),
    }


def nearest_neighbour_accuracy(
    z_train: Array, y_train: NDArray[np.int64], z_test: Array, y_test: NDArray[np.int64]
) -> float:
    """Accuracy of a 1-nearest-neighbour classifier (Euclidean distance)."""
    d2 = (
        np.einsum("ij,ij->i", z_test, z_test)[:, None]
        - 2.0 * z_test @ z_train.T
        + np.einsum("ij,ij->i", z_train, z_train)[None, :]
    )
    return float(np.mean(y_train[np.argmin(d2, axis=1)] == y_test))
