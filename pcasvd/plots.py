"""Static figures (matplotlib). Every function writes one PNG and closes its figure."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from PIL import Image

from .datasets import FACE_SHAPE
from .linalg import Array

# Palette: categorical slots validated for colour-vision deficiencies; recessive chrome.
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, CONTEXT = "#e1e0d9", "#c3c2b7", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list("blue_neutral_red", ["#184f95", BLUE, "#f0efec", "#e34948", "#a32f2e"])

STYLE: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 10,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def styled(func: Callable[..., None]) -> Callable[..., None]:
    """Run a plotting function inside the project's matplotlib style."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        with plt.rc_context(STYLE):
            func(*args, **kwargs)

    return wrapper


def save(fig: plt.Figure, path: Path, *, quantize: bool = False, dpi: int | None = None) -> None:
    """Save ``fig`` as PNG; ``quantize`` stores it with a 256-colour palette (much smaller files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or STYLE["savefig.dpi"])
    plt.close(fig)
    if quantize:
        with Image.open(path) as img:
            small = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        small.save(path, optimize=True)


def _face(vector: Array) -> Array:
    return vector.reshape(FACE_SHAPE)


def _log_k_axis(ax: plt.Axes, k_max: int) -> None:
    """Log-scaled k axis with plain-number ticks."""
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 400, 1000) if t <= k_max]
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    ax.minorticks_off()


def _marker(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=5)


# --------------------------------------------------------------------------- faces
@styled
def plot_eigenfaces(mean_face: Array, components: Array, ratios: Array, path: Path, n_components: int = 15) -> None:
    """Mean face followed by the leading eigenfaces (diverging colour map centred on zero)."""
    fig, axes = plt.subplots(2, (n_components + 1) // 2, figsize=(13, 4.3))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i == 0:
            ax.imshow(_face(mean_face), cmap="gray")
            ax.set_title("Mean face", fontsize=9.5, loc="center")
            continue
        img = _face(components[i - 1])
        lim = np.abs(img).max()
        ax.imshow(img, cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.set_title(f"PC{i}  ·  {100 * ratios[i - 1]:.1f}%", fontsize=9.5, loc="center", fontweight="normal")
    bar = fig.colorbar(
        ScalarMappable(Normalize(-1, 1), DIVERGING), ax=axes, orientation="horizontal", fraction=0.035, pad=0.04, aspect=60
    )
    bar.set_ticks([-1, 0, 1], labels=["negative loading", "0", "positive loading"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8.5, labelcolor=INK_2)
    save(fig, path, quantize=True)


@styled
def plot_scree(ratios: Array, thresholds: dict[float, int], path: Path, n_bars: int = 40) -> None:
    """Explained variance per component (bars) and cumulative explained variance (log-k axis)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    left.bar(np.arange(1, n_bars + 1), 100 * ratios[:n_bars], width=0.65, color=BLUE)
    left.set(xlabel="Principal component", ylabel="Explained variance (%)", xlim=(0, n_bars + 1))
    left.set_title(f"Scree plot: first {n_bars} components")
    left.grid(axis="x", visible=False)

    k = np.arange(1, ratios.size + 1)
    cumulative = 100 * np.cumsum(ratios)
    right.plot(k, cumulative, color=BLUE)
    _log_k_axis(right, k[-1])
    for threshold, kt in thresholds.items():
        _marker(right, kt, cumulative[kt - 1], BLUE)
        right.annotate(
            f"{threshold:.0%} at k = {kt}", (kt, cumulative[kt - 1]), xytext=(9, -13), textcoords="offset points",
            color=INK_2, fontsize=9,
        )
    right.set(xlabel="Number of components k (log scale)", ylabel="Cumulative explained variance (%)", ylim=(0, 102))
    right.set_title("Cumulative explained variance")
    save(fig, path)


@styled
def plot_reconstructions(
    originals: Array, reconstructions: Array, ks: Sequence[int], row_labels: Sequence[str], psnrs: Array, path: Path
) -> None:
    """Grid of faces reconstructed from ``k`` components; the last column is the original."""
    rows, cols = reconstructions.shape[0], len(ks) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.95 * rows))
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            original = c == len(ks)
            ax.imshow(_face(originals[r] if original else reconstructions[r, c]), cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title("Original" if original else f"k = {ks[c]}", fontsize=10, loc="center")
            if not original:  # > 100 dB means the reconstruction is exact up to rounding
                caption = "exact" if psnrs[r, c] > 100 else f"{psnrs[r, c]:.1f} dB"
                ax.set_xlabel(caption, fontsize=8.5, color=INK_2, labelpad=2)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, color=INK_2)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, path)


@styled
def plot_projection(
    scores: Array, target: NDArray[np.int64], images: Array, highlight: Sequence[int], ratios: Array, path: Path
) -> None:
    """Faces in the (PC1, PC2) plane: three subjects highlighted, and image thumbnails."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    others = ~np.isin(target, highlight)
    left.scatter(scores[others, 0], scores[others, 1], s=16, color=CONTEXT, alpha=0.7, linewidths=0, label="Other subjects")
    for subject, color in zip(highlight, (BLUE, ORANGE, AQUA)):
        mask = target == subject
        left.scatter(
            scores[mask, 0], scores[mask, 1], s=46, color=color, edgecolors=SURFACE, linewidths=1.2, label=f"Subject {subject}"
        )
    labels = (f"PC1 ({100 * ratios[0]:.1f}% of variance)", f"PC2 ({100 * ratios[1]:.1f}% of variance)")
    left.set(xlabel=labels[0], ylabel=labels[1])
    left.set_title("All 400 faces projected on the first two components")
    left.legend(loc="upper right", handletextpad=0.3)

    rng = np.random.default_rng(0)
    span = np.ptp(scores, axis=0)
    chosen: list[int] = []
    for i in rng.permutation(len(scores)):  # greedy non-overlapping subset of thumbnails
        if all(np.max(np.abs(scores[i] - scores[j]) / span) > 0.075 for j in chosen):
            chosen.append(i)
    for i in chosen:
        thumb = OffsetImage(_face(images[i]), zoom=0.42, cmap="gray", norm=Normalize(0, 1))
        right.add_artist(AnnotationBbox(thumb, scores[i], frameon=False, pad=0))
    pad = 0.06 * span
    right.set(xlim=(scores[:, 0].min() - pad[0], scores[:, 0].max() + pad[0]), xlabel=labels[0])
    right.set(ylim=(scores[:, 1].min() - pad[1], scores[:, 1].max() + pad[1]), ylabel=labels[1])
    right.set_title(f"{len(chosen)} faces drawn at their (PC1, PC2) coordinates")
    save(fig, path, quantize=True)


@styled
def plot_generalization(
    ks: Array, mse_train: Array, mse_test: Array, accuracy: Array, accuracy_raw: float, path: Path
) -> None:
    """Reconstruction error on training vs held-out faces, and 1-NN identification accuracy, against k."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    shown = ks < ks[-1]  # at full rank the training error is exactly 0 (off the log scale)
    for values, color, label in ((mse_train, BLUE, "Training faces"), (mse_test, ORANGE, "Held-out faces")):
        left.plot(ks[shown], values[shown], color=color, label=label)
    left.set(yscale="log", ylabel="Reconstruction MSE per pixel")
    _log_k_axis(left, ks[-1])
    left.set_xlabel(f"Number of components k (log scale; training error = 0 at k = {ks[-1]})")
    left.set_title("Reconstruction error: training vs held-out faces")
    left.legend(loc="lower left")

    right.plot(ks, 100 * accuracy, color=BLUE)
    first = int(np.argmax(accuracy >= accuracy_raw))  # first k matching the raw-pixel baseline
    _marker(right, ks[first], 100 * accuracy[first], BLUE)
    right.annotate(
        f"reaches raw-pixel accuracy at k = {ks[first]}", (ks[first], 100 * accuracy[first]), xytext=(8, -18),
        textcoords="offset points", color=INK_2, fontsize=9,
    )
    right.axhline(100 * accuracy_raw, color=MUTED, lw=1)
    right.annotate(
        f"raw pixels (4096-D): {100 * accuracy_raw:.1f}%", (ks[0], 100 * accuracy_raw), xytext=(2, 5),
        textcoords="offset points", color=INK_2, fontsize=9,
    )
    right.set(xlabel="Number of components k (log scale)", ylabel="Held-out accuracy (%)", ylim=(0, 104))
    _log_k_axis(right, ks[-1])
    right.set_title("1-nearest-neighbour identification in eigenface space")
    save(fig, path)


# ----------------------------------------------------------------------- geometry
def _axis_arrows_2d(ax: plt.Axes, mean: Array, components: Array, variances: Array) -> None:
    for i, (v, lam, color) in enumerate(zip(components, variances, (BLUE, ORANGE))):
        tip = mean + 2 * np.sqrt(lam) * v
        ax.annotate("", tip, mean, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2, "mutation_scale": 14})
        ax.annotate(f"PC{i + 1}", tip, xytext=(6, 6), textcoords="offset points", color=INK, fontsize=10, fontweight="semibold")


@styled
def plot_gaussian_cloud(
    x2: Array, mean2: Array, components2: Array, variances2: Array,
    x3: Array, mean3: Array, components3: Array, variances3: Array, path: Path,
) -> None:
    """2-D cloud with principal axes, rank-1 projection, and 3-D cloud with its three axes."""
    fig = plt.figure(figsize=(15.5, 5.0))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.1), wspace=0.18)
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set(xlabel="$x_1$", ylabel="$x_2$")

    ax1.scatter(*x2.T, s=9, color=CONTEXT, alpha=0.8, linewidths=0)
    angle = np.degrees(np.arctan2(components2[0, 1], components2[0, 0]))
    for n_std in (1, 2):
        width, height = 2 * n_std * np.sqrt(variances2)
        ax1.add_patch(Ellipse(mean2, width, height, angle=angle, fill=False, color=AXIS, lw=1))
    _axis_arrows_2d(ax1, mean2, components2, variances2)
    ax1.set_title("Principal axes (arrows: 2 standard deviations)")

    v1 = components2[0]
    projected = mean2 + np.outer((x2 - mean2) @ v1, v1)
    shown = slice(0, 120)
    for p, q in zip(x2[shown], projected[shown]):
        ax2.plot(*np.c_[p, q], color=AXIS, lw=0.8, zorder=1)
    ax2.scatter(*x2[shown].T, s=12, color=CONTEXT, linewidths=0, zorder=2, label="Data")
    ax2.scatter(*projected[shown].T, s=14, color=BLUE, linewidths=0, zorder=3, label="Projection on PC1")
    ax2.set(xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
    ax2.set_title(f"Rank-1 reconstruction keeps {100 * variances2[0] / variances2.sum():.1f}% of the variance")
    ax2.legend(loc="lower right")

    ax3 = fig.add_subplot(grid[2], projection="3d")
    ax3.scatter(*x3.T, s=4, color=CONTEXT, alpha=0.45, linewidths=0)
    for i, (v, lam, color) in enumerate(zip(components3, variances3, (BLUE, ORANGE, AQUA))):
        tip = mean3 + 2.5 * np.sqrt(lam) * v
        ax3.quiver(*mean3, *(tip - mean3), color=color, lw=2.6, arrow_length_ratio=0.15)
        ax3.text(*(mean3 + 2.9 * np.sqrt(lam) * v), f"PC{i + 1}", color=INK, fontsize=10, fontweight="semibold")
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
        axis.line.set_color(AXIS)
    ax3.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_3$")
    ax3.tick_params(labelsize=8, colors=INK_2)
    box = np.abs(x3 - mean3).max() * np.array([-1.0, 1.0])
    ax3.set(xlim=mean3[0] + box, ylim=mean3[1] + box, zlim=mean3[2] + box)
    ax3.set_box_aspect((1, 1, 1), zoom=0.95)
    ax3.view_init(elev=18, azim=-62)
    # A 3-D axes box is taller than its 2-D neighbours: align its title with theirs by hand.
    top = ax1.get_position().y1 + 10 / 72 / fig.get_figheight()
    fig.text(ax3.get_position().x0 + 0.01, top, "3-D cloud: three orthogonal principal axes",
             fontsize=11, fontweight="semibold", color=INK, va="bottom")
    save(fig, path)


# -------------------------------------------------------------------- compression
@styled
def plot_compression(image: Array, approximations: dict[int, Array], stats: dict[int, tuple[float, float]], path: Path) -> None:
    """Original image and its rank-k truncated SVDs, annotated with storage ratio and PSNR."""
    fig, axes = plt.subplots(1, len(approximations) + 1, figsize=(3.2 * (len(approximations) + 1), 3.0))
    panels = [("Original (rank 768)", image, "100% storage")]
    panels += [(f"Rank k = {k}", approx, f"{100 * stats[k][0]:.1f}% storage  ·  {stats[k][1]:.1f} dB") for k, approx in approximations.items()]
    for ax, (title, img, caption) in zip(axes, panels):
        ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="center", fontsize=10.5)
        ax.set_xlabel(caption, fontsize=9, color=INK_2)
    fig.tight_layout(w_pad=0.8)
    save(fig, path, quantize=True, dpi=115)


@styled
def plot_compression_curves(
    singular_values: Array,
    shape: tuple[int, int],
    ranks: Array,
    psnr_values: Array,
    ratios: Array,
    highlight: Sequence[int],
    path: Path,
) -> None:
    """Singular value spectrum of the image, and PSNR against storage."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    left.plot(np.arange(1, singular_values.size + 1), singular_values, color=BLUE)
    left.set(yscale="log", xlabel="Index i (log scale)", ylabel="Singular value $\\sigma_i$ (log scale)")
    _log_k_axis(left, singular_values.size)
    left.set_title(f"Singular value spectrum of the {shape[0]} × {shape[1]} image")

    useful = ratios <= 1.0  # beyond k = mn / (m + n + 1) the factors cost more than the pixels
    right.plot(100 * ratios[useful], psnr_values[useful], color=BLUE)
    for k in highlight:
        i = int(np.flatnonzero(ranks == k)[0])
        _marker(right, 100 * ratios[i], psnr_values[i], BLUE)
        right.annotate(f"k = {k}", (100 * ratios[i], psnr_values[i]), xytext=(-8, 8), textcoords="offset points",
                       ha="right", color=INK_2, fontsize=9)
    right.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    right.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    right.minorticks_off()
    right.set(xlabel="Storage: k (m + n + 1) / (m n)  (%, log scale)", ylabel="PSNR (dB)")
    break_even = shape[0] * shape[1] // (shape[0] + shape[1] + 1)
    right.set_title(f"Rate-distortion of truncated SVD (break-even at k = {break_even})")
    save(fig, path)


# ------------------------------------------------------------------------ solvers
@styled
def plot_timing(panels: Sequence[dict[str, Any]], path: Path) -> None:
    """Dot plot of median fit times, one panel per dataset, ours vs scikit-learn on each row."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        rows = panel["rows"]
        y = np.arange(len(rows))[::-1]
        for yi, row in zip(y, rows):
            times = [t for t in (row["ours_s"], row["sklearn_s"]) if t is not None]
            ax.plot([min(times), max(times)], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.plot(row["ours_s"], yi, "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
            if row["sklearn_s"] is not None:
                ax.plot(row["sklearn_s"], yi, "o", ms=8, color=ORANGE, mec=SURFACE, mew=1.5, zorder=3)
        ax.set_yticks(y, labels=[row["label"] for row in rows])
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
        ax.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Median fit time (s, log scale)")
        ax.set_title(panel["title"])
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE, mec=SURFACE, label="pcasvd (this repo)"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=ORANGE, mec=SURFACE, label="scikit-learn"),
    ]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, path)


@styled
def plot_randomized_accuracy(
    n_iters: Sequence[int],
    ours: dict[str, Sequence[float]],
    sklearn: dict[str, Sequence[float]],
    k: int,
    n_seeds: int,
    path: Path,
) -> None:
    """Randomized SVD accuracy (median over seeds) against the number of power (subspace) iterations."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.0))
    keys = (("max_rel_sv_err", left, f"Max relative error of the top-{k} singular values"),
            ("recon_excess", right, "Excess reconstruction error over the optimum"))
    for key, ax, title in keys:
        ax.plot(n_iters, ours[key], color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.2, label="pcasvd randomized (QR + Jacobi)")
        ax.plot(n_iters, sklearn[key], color=ORANGE, marker="o", ms=6, mec=SURFACE, mew=1.2, label="scikit-learn randomized")
        ax.set(yscale="log", xlabel=f"Power iterations q  (median over {n_seeds} seeds)")
        ax.set_xticks(list(n_iters))
        ax.set_title(title)
    left.set_ylabel(r"$\max_i\ |\hat\sigma_i - \sigma_i| \,/\, \sigma_i$")
    right.set_ylabel(r"$\|X - \hat X_k\|_F^2 \,/\, \|X - X_k\|_F^2 - 1$")
    left.legend(loc="lower left")
    save(fig, path)
