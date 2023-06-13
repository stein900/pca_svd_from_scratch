"""Datasets: Olivetti faces, synthetic Gaussian clouds and a natural test image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .linalg import Array

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FACE_SHAPE = (64, 64)


@dataclass(frozen=True)
class Faces:
    """Olivetti faces: 400 images (64x64, grey levels in [0, 1]) of 40 subjects."""

    data: Array  # (400, 4096) float64, one flattened image per row
    target: NDArray[np.int64]  # (400,) subject id in 0..39


def load_olivetti(data_dir: Path = DEFAULT_DATA_DIR) -> Faces:
    """Load the Olivetti faces (downloaded once by scikit-learn, cached in ``data_dir``)."""
    from sklearn.datasets import fetch_olivetti_faces

    data_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_olivetti_faces(data_home=str(data_dir), shuffle=False)
    return Faces(data=bunch.data.astype(np.float64), target=bunch.target.astype(np.int64))


def split_per_subject(
    target: NDArray[np.int64], n_test_per_subject: int = 2, seed: int = 0
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Stratified split: hold out ``n_test_per_subject`` random images of every subject."""
    rng = np.random.default_rng(seed)
    test = np.concatenate(
        [rng.choice(np.flatnonzero(target == c), n_test_per_subject, replace=False) for c in np.unique(target)]
    )
    test.sort()
    train = np.setdiff1d(np.arange(target.size), test)
    return train, test


def gaussian_cloud(n_samples: int, mean: Array, cov: Array, seed: int = 0) -> Array:
    """Draw ``n_samples`` points from the multivariate normal ``N(mean, cov)``."""
    return np.random.default_rng(seed).multivariate_normal(mean, cov, size=n_samples)


def rotation_covariance(stds: Array, angles_deg: Array) -> Array:
    """Covariance ``R diag(stds^2) R^T`` whose eigenvectors are rotated axes.

    In 2-D, ``angles_deg`` holds a single rotation angle; in 3-D, three Euler
    angles (rotations about x, then y, then z).
    """
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    if len(stds) == 2:
        rot = np.array([[np.cos(a[0]), -np.sin(a[0])], [np.sin(a[0]), np.cos(a[0])]])
    else:
        cx, cy, cz = np.cos(a)
        sx, sy, sz = np.sin(a)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rot = rz @ ry @ rx
    return rot @ np.diag(np.asarray(stds, dtype=np.float64) ** 2) @ rot.T


def load_raccoon(data_dir: Path = DEFAULT_DATA_DIR) -> NDArray[np.uint8]:
    """Grey-scale 768x1024 raccoon photo from ``scipy.datasets.face`` (CC0, pixnio.com).

    The first call downloads the image through ``pooch``; a copy is cached as
    ``data_dir/raccoon_gray.npy``.
    """
    cache = data_dir / "raccoon_gray.npy"
    if cache.exists():
        return np.load(cache)
    from scipy.datasets import face

    data_dir.mkdir(parents=True, exist_ok=True)
    image = face(gray=True)
    np.save(cache, image)
    return image
