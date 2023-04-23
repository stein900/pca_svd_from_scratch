"""Principal Component Analysis with a scikit-learn-like API and five solvers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .linalg import EPS, Array, jacobi_svd, power_iteration_eigh, randomized_svd, sign_flip

Solver = Literal["svd", "covariance_eigh", "jacobi", "power", "randomized"]
SOLVERS: tuple[Solver, ...] = ("svd", "covariance_eigh", "jacobi", "power", "randomized")


class PCA:
    """Principal Component Analysis implemented with NumPy.

    Parameters
    ----------
    n_components:
        Number of components to keep; ``None`` keeps ``min(n_samples, n_features)``.
    solver:
        ``"svd"``             LAPACK SVD of the centred data (reference).
        ``"covariance_eigh"`` LAPACK eigendecomposition of the scatter matrix
                              ``Xc^T Xc``, or of the Gram matrix ``Xc Xc^T`` when
                              ``n_samples < n_features`` (eigenfaces trick).
        ``"jacobi"``          from scratch: exact SVD by one-sided Jacobi rotations.
        ``"power"``           from scratch: power iteration with deflation on the
                              smaller of the scatter / Gram matrices.
        ``"randomized"``      from scratch: Halko et al. randomized SVD.
    whiten:
        If True, :meth:`transform` rescales every component to unit variance.
    tol, max_iter:
        Stopping rule of the ``"power"`` solver (residual relative to ``lambda_1``).
    n_oversamples, n_power_iter:
        Sketch oversampling and subspace iterations of the ``"randomized"`` solver.
    random_state:
        Seed of the iterative solvers' random starting points.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        solver: Solver = "svd",
        whiten: bool = False,
        tol: float = 1e-10,
        max_iter: int = 10_000,
        n_oversamples: int = 10,
        n_power_iter: int = 4,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.solver = solver
        self.whiten = whiten
        self.tol = tol
        self.max_iter = max_iter
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X: ArrayLike) -> PCA:
        """Estimate the mean and the principal axes of ``X`` (n_samples, n_features)."""
        X = _as_float_matrix(X)
        n, p = X.shape
        if n < 2:
            raise ValueError("PCA needs at least 2 samples.")
        k = min(n, p) if self.n_components is None else int(self.n_components)
        if not 1 <= k <= min(n, p):
            raise ValueError(f"n_components must be in [1, {min(n, p)}], got {self.n_components}.")
        if self.solver not in SOLVERS:
            raise ValueError(f"Unknown solver {self.solver!r}; choose one of {SOLVERS}.")

        self.mean_ = X.mean(axis=0)
        xc = X - self.mean_
        self.n_iter_: np.ndarray | None = None
        singular_values, components = self._decompose(xc, k)

        self.components_ = sign_flip(components)
        self.singular_values_ = singular_values
        self.explained_variance_ = singular_values**2 / (n - 1)
        total_variance = np.einsum("ij,ij->", xc, xc) / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.n_components_, self.n_samples_, self.n_features_in_ = k, n, p
        return self

    def _decompose(self, xc: Array, k: int) -> tuple[Array, Array]:
        """Return the top-``k`` singular values and right singular vectors (rows) of ``xc``."""
        n, p = xc.shape
        if self.solver == "svd":
            _, s, vt = np.linalg.svd(xc, full_matrices=False)
            return s[:k], vt[:k]
        if self.solver == "jacobi":
            _, s, vt = jacobi_svd(xc)
            return s[:k], vt[:k]
        if self.solver == "randomized":
            _, s, vt = randomized_svd(
                xc, k, n_oversamples=self.n_oversamples, n_power_iter=self.n_power_iter, seed=self.random_state
            )
            return s, vt

        # Eigen-solvers work on the smaller of the two cross-product matrices.
        gram = n < p
        cross = xc @ xc.T if gram else xc.T @ xc  # eigenvalues are sigma_i^2
        if self.solver == "covariance_eigh":
            evals, evecs = np.linalg.eigh(cross)  # ascending order
            evals, evecs = evals[::-1][:k], evecs[:, ::-1][:, :k]
        else:  # power
            evals, evecs, self.n_iter_ = power_iteration_eigh(
                cross, k, tol=self.tol, max_iter=self.max_iter, seed=self.random_state
            )
        s = np.sqrt(np.clip(evals, 0.0, None))
        return _from_gram(xc, evecs, s, self.random_state) if gram else (s, evecs.T)

    # ------------------------------------------------------------ transform
    def transform(self, X: ArrayLike) -> Array:
        """Project ``X`` onto the principal axes: ``Z = (X - mean) V^T``."""
        self._check_fitted()
        Z = (_as_float_matrix(X) - self.mean_) @ self.components_.T
        if self.whiten:
            Z /= np.maximum(np.sqrt(self.explained_variance_), EPS)
        return Z

    def fit_transform(self, X: ArrayLike) -> Array:
        """Fit the model on ``X`` and return its projection."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> Array:
        """Map scores back to feature space: ``X_hat = Z V + mean``."""
        self._check_fitted()
        components = self.components_
        if self.whiten:
            components = np.sqrt(self.explained_variance_)[:, None] * components
        return _as_float_matrix(Z) @ components + self.mean_

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("This PCA instance is not fitted yet; call fit() first.")


def _from_gram(xc: Array, u: Array, s: Array, seed: int) -> tuple[Array, Array]:
    """Map Gram eigenvectors to principal axes: ``v_i = Xc^T u_i / sigma_i``.

    Gram eigenvalues carry an absolute error of order ``n eps sigma_1^2``, so axes
    with ``sigma_i^2 <= n eps sigma_1^2`` are numerically null and cannot be
    recovered this way; they are completed with an orthonormal basis of the
    remaining subspace, and their singular values recomputed as ``||Xc v||``.
    """
    keep = s**2 > xc.shape[0] * EPS * s[0] ** 2
    vt = (xc.T @ u[:, keep] / s[keep]).T
    n_missing = int((~keep).sum())
    if n_missing:
        rng = np.random.default_rng(seed)
        extra = rng.standard_normal((xc.shape[1], n_missing))
        for _ in range(2):  # project out the recovered axes ("twice is enough")
            extra -= vt.T @ (vt @ extra)
        completion = np.linalg.qr(extra)[0].T
        vt = np.vstack([vt, completion])
        s = np.concatenate([s[keep], np.linalg.norm(xc @ completion.T, axis=0)])
    return s, vt


def _as_float_matrix(X: ArrayLike) -> Array:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {X.shape}.")
    return X
