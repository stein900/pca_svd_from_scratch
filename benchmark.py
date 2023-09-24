"""Timing benchmark of every solver against scikit-learn, and randomized-SVD accuracy vs power iterations.

Usage:  python benchmark.py [--repeats 7] [--k 50]
Writes assets/solver_timing.png, assets/randomized_power_iterations.png and the
"benchmark" section of results/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA as SkPCA  # noqa: E402

from pcasvd import PCA, plots  # noqa: E402
from pcasvd.datasets import load_olivetti  # noqa: E402
from pcasvd.metrics import reconstruction_mse  # noqa: E402
from pcasvd.timing import median_fit_time, system_info  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED = 0
ROWS = (  # (label, our solver, scikit-learn counterpart)
    ("Full SVD (LAPACK)", "svd", "full"),
    ("Covariance / Gram eigh", "covariance_eigh", "covariance_eigh"),
    ("Randomized SVD", "randomized", "randomized"),
    ("Power iteration | ARPACK", "power", "arpack"),
    ("One-sided Jacobi SVD", "jacobi", None),
)
POWER_ITERATIONS = (0, 1, 2, 3, 4, 6, 8, 10, 14)
N_SEEDS = 5


def tall_dataset(n: int = 20_000, p: int = 300, rank: int = 40, seed: int = SEED) -> np.ndarray:
    """Many samples, few features: a decaying low-rank signal plus isotropic noise."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((p, rank)))
    scales = 10.0 * 0.85 ** np.arange(rank)
    return (rng.standard_normal((n, rank)) * scales) @ basis.T + 0.3 * rng.standard_normal((n, p))


def time_solvers(X: np.ndarray, k: int, repeats: int) -> list[dict[str, Any]]:
    rows = []
    for label, ours, theirs in ROWS:
        row: dict[str, Any] = {"label": label, "ours_solver": ours, "sklearn_solver": theirs}
        t = median_fit_time(lambda: PCA(k, solver=ours, random_state=SEED), X, repeats)
        row["ours_s"], row["ours_min_s"] = t["median_s"], t["min_s"]
        row["sklearn_s"] = row["sklearn_min_s"] = None
        if theirs is not None:
            t = median_fit_time(lambda: SkPCA(k, svd_solver=theirs, random_state=SEED), X, repeats)
            row["sklearn_s"], row["sklearn_min_s"] = t["median_s"], t["min_s"]
        print(f"  {label:<28} ours {row['ours_s']:9.4f} s   sklearn " + (f"{row['sklearn_s']:9.4f} s" if theirs else "      n/a"))
        rows.append(row)
    return rows


def randomized_accuracy(X: np.ndarray, k: int, n_iters: Sequence[int], n_seeds: int) -> dict[str, Any]:
    """Median accuracy (over seeds) of both randomized solvers against the exact SVD, per number of power iterations."""
    exact = PCA(solver="svd").fit(X)
    optimum = reconstruction_mse(exact, X, k)
    curves: dict[str, dict[str, list[float]]] = {
        name: {"max_rel_sv_err": [], "recon_excess": []} for name in ("ours", "sklearn")
    }
    for q in n_iters:
        runs: dict[str, dict[str, list[float]]] = {name: {"max_rel_sv_err": [], "recon_excess": []} for name in curves}
        for seed in range(n_seeds):
            models = {
                "ours": PCA(k, solver="randomized", n_power_iter=q, random_state=seed).fit(X),
                "sklearn": SkPCA(k, svd_solver="randomized", iterated_power=q, random_state=seed).fit(X),
            }
            for name, model in models.items():
                rel = np.abs(model.singular_values_ - exact.singular_values_[:k]) / exact.singular_values_[:k]
                runs[name]["max_rel_sv_err"].append(float(rel.max()))
                runs[name]["recon_excess"].append(reconstruction_mse(model, X, k) / optimum - 1.0)
        for name, metrics in runs.items():
            for key, values in metrics.items():
                curves[name][key].append(float(np.median(values)))
    s = exact.singular_values_
    sketch = k + PCA().n_oversamples
    return {
        "k": k,
        "power_iterations": list(n_iters),
        "n_seeds": n_seeds,
        "sigma_ratio_sketch_over_k": float(s[sketch] / s[k - 1]),  # sigma_{k+l+1} / sigma_k
        **curves,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "metrics.json")
    parser.add_argument("--repeats", type=int, default=7, help="timed runs per solver (after one warm-up run)")
    parser.add_argument("--k", type=int, default=50, help="components kept on the faces")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    info = system_info()
    print(f"Hardware: {info['cpu']} ({info['logical_cores']} logical cores), {info['os']}, "
          f"NumPy {info['numpy']} ({info['blas']}), scikit-learn {info['scikit_learn']}")
    faces = load_olivetti(args.data_dir).data
    tall = tall_dataset()
    datasets = [("Olivetti faces", faces, args.k), ("Tall synthetic", tall, 20)]
    panels = []
    for name, X, k in datasets:
        print(f"\n{name}: {X.shape[0]} x {X.shape[1]}, k = {k}, median of {args.repeats} runs")
        rows = time_solvers(X, k, args.repeats)
        panels.append({"name": name, "shape": list(X.shape), "k": k, "rows": rows,
                       "title": f"{name}  ({X.shape[0]:,} × {X.shape[1]:,}, k = {k})"})
    plots.plot_timing(panels, args.assets_dir / "solver_timing.png")

    accuracy = randomized_accuracy(faces, args.k, POWER_ITERATIONS, N_SEEDS)
    plots.plot_randomized_accuracy(
        accuracy["power_iterations"], accuracy["ours"], accuracy["sklearn"], args.k, N_SEEDS,
        args.assets_dir / "randomized_power_iterations.png",
    )
    print(f"\nRandomized SVD on the faces, k = {args.k}: max relative singular-value error (median of {N_SEEDS} seeds)")
    for q, a, b in zip(accuracy["power_iterations"], accuracy["ours"]["max_rel_sv_err"], accuracy["sklearn"]["max_rel_sv_err"]):
        print(f"  q = {q:>2}: ours {a:.2e}   sklearn {b:.2e}")

    benchmark = {"system": info, "repeats": args.repeats, "warmup": 1, "datasets": panels,
                 "randomized_accuracy": accuracy, "runtime_s": time.perf_counter() - start}
    metrics = json.loads(args.results.read_text(encoding="utf-8")) if args.results.exists() else {}
    metrics["benchmark"] = benchmark
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nBenchmark finished in {benchmark['runtime_s']:.1f} s; results in {args.results}")


if __name__ == "__main__":
    main()
