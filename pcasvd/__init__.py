"""PCA / SVD from scratch: NumPy PCA with exact, iterative and randomized solvers."""

from .linalg import jacobi_svd, power_iteration_eigh, randomized_svd, sign_flip
from .pca import PCA, SOLVERS

__all__ = ["PCA", "SOLVERS", "jacobi_svd", "power_iteration_eigh", "randomized_svd", "sign_flip"]
__version__ = "0.3.0"
