"""Animated GIF of a face progressively reconstructed from more and more eigenfaces."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .datasets import FACE_SHAPE
from .linalg import Array
from .plots import BLUE, GRID, INK, INK_2, STYLE


def _frame(original: Array, approx: Array, k: int, explained: float) -> Image.Image:
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(4.8, 3.05), dpi=100)
        label = f"k = {k} eigenface" + ("s" if k > 1 else "")
        for i, (img, title) in enumerate(((approx, label), (original, "Original"))):
            ax = fig.add_axes((0.04 + 0.49 * i, 0.17, 0.43, 0.68))
            ax.imshow(img.reshape(FACE_SHAPE), cmap="gray", vmin=0, vmax=1, interpolation="bicubic")
            ax.set_axis_off()
            ax.set_title(title, loc="center", fontsize=11, color=INK)
        bar = fig.add_axes((0.04, 0.07, 0.92, 0.035))
        bar.set_axis_off()
        bar.barh(0, 1, color=GRID, height=1)
        bar.barh(0, explained, color=BLUE, height=1)
        bar.set_xlim(0, 1)
        fig.text(0.04, 0.115, f"{100 * explained:.1f}% of the variance", fontsize=9, color=INK_2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches=None)
        plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def reconstruction_gif(
    original: Array,
    reconstructions: Sequence[Array],
    ks: Sequence[int],
    cumulative_ratio: Array,
    path: Path,
    frame_ms: int = 380,
    hold_ms: int = 2200,
) -> None:
    """Write a GIF where frame ``i`` shows the reconstruction from ``ks[i]`` components."""
    frames = [_frame(original, approx, k, float(cumulative_ratio[k - 1])) for approx, k in zip(reconstructions, ks)]
    palette = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames_p = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [frame_ms] * (len(frames_p) - 1) + [hold_ms]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=durations, loop=0, optimize=True)
