# PCA / SVD from scratch

NumPy PCA with a scikit-learn-style `fit` / `transform` API. Solvers so far:
LAPACK SVD, the Gram-matrix eigen-decomposition, a one-sided Jacobi SVD and
power iteration with deflation.

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## License

MIT. See LICENSE.
