"""Regenerate every figure in ``assets/`` and the metrics in ``results/metrics.json``.

Usage:  python demo.py [--data-dir data] [--assets-dir assets] [--results results/metrics.json]
The timing benchmark lives in ``benchmark.py``; ``python demo.py --benchmark`` runs it too.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA as SkPCA  # noqa: E402

from pcasvd import PCA, SOLVERS, plots  # noqa: E402
from pcasvd.animation import reconstruction_gif  # noqa: E402
from pcasvd.compression import low_rank_approx, optimal_squared_error, psnr, storage_ratio  # noqa: E402
from pcasvd.datasets import (  # noqa: E402
    FACE_SHAPE,
    Faces,
    gaussian_cloud,
    load_olivetti,
    load_raccoon,
    rotation_covariance,
    split_per_subject,
)
from pcasvd.linalg import randomized_svd  # noqa: E402
from pcasvd.metrics import (  # noqa: E402
    agreement,
    components_for_variance,
    nearest_neighbour_accuracy,
    reconstruct,
    reconstruction_mse,
)
from pcasvd.timing import system_info  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED = 0
K_COMPARE = 50
RECON_KS = (5, 10, 25, 50, 100, 200)
GIF_KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 399)
GIF_FACE = 103
COMPRESSION_RANKS = (5, 20, 50, 100)
SKLEARN_COUNTERPART = {
    "svd": "full",
    "covariance_eigh": "covariance_eigh",
    "jacobi": "full",
    "power": "arpack",
    "randomized": "randomized",
}


def _psnr01(reference: np.ndarray, approx: np.ndarray) -> float:
    return psnr(reference, approx, data_range=1.0)


def tightest_subjects(scores: np.ndarray, target: np.ndarray, n: int) -> list[int]:
    """Subjects whose images are most compact in the (PC1, PC2) plane (smallest mean spread)."""
    spread = [np.mean(np.linalg.norm(scores[target == c] - scores[target == c].mean(0), axis=1)) for c in np.unique(target)]
    return sorted(int(c) for c in np.argsort(spread)[:n])


def eigenfaces_experiment(faces: Faces, assets: Path) -> dict[str, Any]:
    """PCA of all 400 faces: eigenfaces, scree plot, 2-D projection and the reconstruction GIF."""
    X = faces.data
    pca = PCA(solver="svd").fit(X)
    ratios = pca.explained_variance_ratio_
    thresholds = {t: components_for_variance(ratios, t) for t in (0.5, 0.8, 0.9, 0.95, 0.99)}
    plots.plot_eigenfaces(pca.mean_, pca.components_, ratios, assets / "eigenfaces.png")
    plots.plot_scree(ratios, {t: thresholds[t] for t in (0.8, 0.9, 0.95, 0.99)}, assets / "scree.png")

    scores = pca.transform(X)[:, :2]
    images = X.reshape(-1, *FACE_SHAPE)
    half = FACE_SHAPE[1] // 2
    brightness = X.mean(axis=1)
    left_right = images[:, :, :half].mean(axis=(1, 2)) - images[:, :, half:].mean(axis=(1, 2))
    highlight = tightest_subjects(scores, faces.target, 3)
    plots.plot_projection(scores, faces.target, X, highlight, ratios, assets / "projection_2d.png")

    face = X[GIF_FACE : GIF_FACE + 1]
    frames = [reconstruct(pca, face, k)[0] for k in GIF_KS]
    reconstruction_gif(face[0], frames, GIF_KS, np.cumsum(ratios), assets / "face_reconstruction.gif")
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "explained_variance_ratio_top10": ratios[:10].tolist(),
        "components_for_variance": {f"{t:.2f}": k for t, k in thresholds.items()},
        "corr_pc1_mean_brightness": float(np.corrcoef(scores[:, 0], brightness)[0, 1]),
        "corr_pc2_left_right_brightness": float(np.corrcoef(scores[:, 1], left_right)[0, 1]),
        "highlighted_subjects": highlight,
        "gif_face_index": GIF_FACE,
    }


def generalization_experiment(faces: Faces, assets: Path) -> dict[str, Any]:
    """Fit on 8 images per subject, evaluate reconstruction and 1-NN identification on the other 2."""
    train, test = split_per_subject(faces.target, n_test_per_subject=2, seed=SEED)
    X_train, X_test = faces.data[train], faces.data[test]
    y_train, y_test = faces.target[train], faces.target[test]
    pca = PCA(solver="svd").fit(X_train)
    k_max = pca.n_components_ - 1  # centred training data has rank n_train - 1
    ks = np.unique(np.round(np.geomspace(1, k_max, 30)).astype(int))
    z_train, z_test = pca.transform(X_train), pca.transform(X_test)
    mse_train = np.array([reconstruction_mse(pca, X_train, k) for k in ks])
    mse_test = np.array([reconstruction_mse(pca, X_test, k) for k in ks])
    accuracy = np.array([nearest_neighbour_accuracy(z_train[:, :k], y_train, z_test[:, :k], y_test) for k in ks])
    accuracy_raw = nearest_neighbour_accuracy(X_train, y_train, X_test, y_test)
    plots.plot_generalization(ks, mse_train, mse_test, accuracy, accuracy_raw, assets / "generalization.png")

    rows = np.array([train[0], train[45], test[7], test[52]])  # two training and two held-out faces
    labels = ["train", "train", "held-out", "held-out"]
    recon_ks = (*RECON_KS, k_max)
    recons = np.stack([[reconstruct(pca, faces.data[r : r + 1], k)[0] for k in recon_ks] for r in rows])
    psnrs = np.array([[_psnr01(faces.data[r], recons[i, j]) for j in range(len(recon_ks))] for i, r in enumerate(rows)])
    row_labels = [f"{lab} · subject {faces.target[r]}" for lab, r in zip(labels, rows)]
    plots.plot_reconstructions(faces.data[rows], recons, recon_ks, row_labels, psnrs, assets / "reconstructions.png")

    best = int(np.argmax(accuracy))
    return {
        "n_train": int(train.size),
        "n_test": int(test.size),
        "k_max": int(k_max),
        "ks": ks.tolist(),
        "mse_train": mse_train.tolist(),
        "mse_test": mse_test.tolist(),
        "nn_accuracy": accuracy.tolist(),
        "nn_accuracy_raw_pixels": accuracy_raw,
        "best_nn_accuracy": float(accuracy[best]),
        "best_nn_k": int(ks[best]),
        "reconstruction_grid": {
            "ks": list(recon_ks),
            "rows": [{"index": int(r), "split": lab, "psnr_db": p.tolist()} for r, lab, p in zip(rows, labels, psnrs)],
        },
    }


def sklearn_comparison(faces: Faces) -> dict[str, Any]:
    """Accuracy of every solver (ours and its sklearn counterpart) against the exact LAPACK reference."""
    X = faces.data
    exact = SkPCA(K_COMPARE, svd_solver="full").fit(X)
    optimum = reconstruction_mse(exact, X, K_COMPARE)
    rows = []
    for solver in SOLVERS:
        ours = PCA(K_COMPARE, solver=solver, random_state=SEED).fit(X)
        counterpart = SKLEARN_COUNTERPART[solver]
        theirs = SkPCA(K_COMPARE, svd_solver=counterpart, random_state=SEED).fit(X)
        row: dict[str, Any] = {"solver": solver, "sklearn_solver": counterpart}
        row["ours"] = agreement(ours, exact, X)
        row["sklearn"] = agreement(theirs, exact, X)
        for side in ("ours", "sklearn"):
            row[side]["reconstruction_excess"] = row[side]["reconstruction_mse"] / optimum - 1.0
        if ours.n_iter_ is not None:
            row["ours"]["power_iterations_total"] = int(ours.n_iter_.sum())
        rows.append(row)

    full_rank = {}  # all 399 non-null components of the exact solvers
    exact_full = SkPCA(399, svd_solver="full").fit(X)
    for solver in ("svd", "covariance_eigh", "jacobi"):
        full_rank[solver] = agreement(PCA(399, solver=solver).fit(X), exact_full, X)
    return {
        "k": K_COMPARE,
        "reference": "sklearn PCA(svd_solver='full')",
        "optimal_reconstruction_mse": optimum,
        "rows": rows,
        "full_rank_k399": full_rank,
    }


def print_comparison(comparison: dict[str, Any]) -> None:
    k = comparison["k"]
    print(f"\nAgreement with the exact reference ({comparison['reference']}), Olivetti 400 x 4096, k = {k}")
    header = (
        f"{'solver':<16}{'max|dV| ours':>14}{'max|dV| sklearn':>17}{'rel.err var ours':>18}"
        f"{'MSE excess ours':>17}{'MSE excess sklearn':>20}  sklearn solver"
    )
    print(header)
    print("-" * len(header))
    for row in comparison["rows"]:
        o, s = row["ours"], row["sklearn"]
        print(
            f"{row['solver']:<16}{o['max_abs_component_diff']:>14.2e}{s['max_abs_component_diff']:>17.2e}"
            f"{o['max_rel_explained_variance_err']:>18.2e}{o['reconstruction_excess']:>17.2e}{s['reconstruction_excess']:>20.2e}"
            f"  {row['sklearn_solver']}"
        )
    print(f"MSE excess = MSE / optimal MSE - 1, optimal (Eckart-Young) MSE = {comparison['optimal_reconstruction_mse']:.6e}")


def cloud_experiment(assets: Path) -> dict[str, Any]:
    """Synthetic Gaussian clouds: estimated principal axes vs the true covariance eigenvectors."""
    cov2 = rotation_covariance(np.array([3.0, 1.0]), np.array([30.0]))
    x2 = gaussian_cloud(500, np.array([2.0, 1.0]), cov2, seed=SEED)
    cov3 = rotation_covariance(np.array([3.0, 1.5, 0.5]), np.array([20.0, -30.0, 45.0]))
    x3 = gaussian_cloud(800, np.zeros(3), cov3, seed=SEED + 1)
    results: dict[str, Any] = {}
    fitted = {}
    for name, x, cov in (("2d", x2, cov2), ("3d", x3, cov3)):
        pca = PCA(solver="power", random_state=SEED).fit(x)  # from-scratch solver on a tiny problem
        true_vals, true_vecs = np.linalg.eigh(cov)
        true_vals, true_vecs = true_vals[::-1], true_vecs[:, ::-1]
        cosines = np.abs(np.sum(pca.components_ * true_vecs.T, axis=1)).clip(max=1.0)
        results[name] = {
            "n_samples": int(x.shape[0]),
            "true_variances": true_vals.tolist(),
            "estimated_variances": pca.explained_variance_.tolist(),
            "axis_angle_error_deg": np.degrees(np.arccos(cosines)).tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
        fitted[name] = pca
    p2, p3 = fitted["2d"], fitted["3d"]
    plots.plot_gaussian_cloud(
        x2, p2.mean_, p2.components_, p2.explained_variance_,
        x3, p3.mean_, p3.components_, p3.explained_variance_, assets / "gaussian_cloud.png",
    )
    return results


def compression_experiment(data_dir: Path, assets: Path) -> dict[str, Any]:
    """Truncated SVD of a 768 x 1024 photo: storage, PSNR, Eckart-Young check, randomized SVD."""
    image = load_raccoon(data_dir).astype(np.float64)
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    ranks = np.unique(np.round(np.geomspace(1, s.size, 60)).astype(int))
    ranks = np.union1d(ranks, COMPRESSION_RANKS)
    psnr_curve = np.array([psnr(image, low_rank_approx(u, s, vt, k)) for k in ranks])
    ratio_curve = np.array([storage_ratio(image.shape, k) for k in ranks])

    approximations, stats, per_rank = {}, {}, []
    for k in COMPRESSION_RANKS:
        approx = low_rank_approx(u, s, vt, k)
        err = float(np.sum((image - approx) ** 2))
        optimum = optimal_squared_error(s, k)
        ur, sr, vtr = randomized_svd(image, k, n_power_iter=4, seed=SEED)
        approx_rand = low_rank_approx(ur, sr, vtr, k)
        err_rand = float(np.sum((image - approx_rand) ** 2))
        approximations[k] = approx
        stats[k] = (storage_ratio(image.shape, k), psnr(image, approx))
        per_rank.append({
            "k": k,
            "storage_ratio": stats[k][0],
            "psnr_db": stats[k][1],
            "energy_retained": float(np.sum(s[:k] ** 2) / np.sum(s**2)),
            "eckart_young_rel_gap": abs(err - optimum) / optimum,
            "randomized_psnr_db": psnr(image, approx_rand),
            "randomized_error_over_optimum": err_rand / optimum,
        })
    plots.plot_compression(image, approximations, stats, assets / "svd_compression.png")
    plots.plot_compression_curves(
        s, image.shape, ranks, psnr_curve, ratio_curve, COMPRESSION_RANKS, assets / "svd_spectrum.png"
    )
    return {
        "image": "scipy.datasets.face(gray=True), raccoon photo derived from pixnio.com (CC0)",
        "shape": list(image.shape),
        "ranks": per_rank,
        "curve": {"ranks": ranks.tolist(), "psnr_db": psnr_curve.tolist(), "storage_ratio": ratio_curve.tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "metrics.json")
    parser.add_argument("--benchmark", action="store_true", help="also run the timing benchmark (benchmark.py)")
    args = parser.parse_args()

    start = time.perf_counter()
    faces = load_olivetti(args.data_dir)
    metrics: dict[str, Any] = {"generated_by": "demo.py", "seed": SEED, "system": system_info()}
    metrics["eigenfaces"] = eigenfaces_experiment(faces, args.assets_dir)
    metrics["generalization"] = generalization_experiment(faces, args.assets_dir)
    metrics["sklearn_comparison"] = sklearn_comparison(faces)
    print_comparison(metrics["sklearn_comparison"])
    metrics["gaussian_cloud"] = cloud_experiment(args.assets_dir)
    metrics["compression"] = compression_experiment(args.data_dir, args.assets_dir)
    metrics["demo_runtime_s"] = time.perf_counter() - start

    if args.results.exists():  # keep the benchmark section written by benchmark.py
        previous = json.loads(args.results.read_text(encoding="utf-8"))
        if "benchmark" in previous:
            metrics["benchmark"] = previous["benchmark"]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nFigures written to {args.assets_dir}, metrics to {args.results} ({metrics['demo_runtime_s']:.1f} s)")

    if args.benchmark:
        import benchmark

        benchmark.main([f"--data-dir={args.data_dir}", f"--assets-dir={args.assets_dir}", f"--results={args.results}"])


if __name__ == "__main__":
    main()
