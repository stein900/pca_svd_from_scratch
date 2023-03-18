"""PCA / SVD from scratch: NumPy PCA with exact and iterative solvers."""

from .linalg import jacobi_svd, power_iteration_eigh, sign_flip
from .pca import PCA, SOLVERS

__all__ = ["PCA", "SOLVERS", "jacobi_svd", "power_iteration_eigh", "sign_flip"]
__version__ = "0.2.0"
