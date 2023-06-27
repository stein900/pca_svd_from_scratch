"""Wall-clock timing and a description of the machine the numbers come from."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def median_fit_time(make_model: Callable[[], Any], X: np.ndarray, repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    """Median and best wall-clock time of ``make_model().fit(X)`` over ``repeats`` runs."""
    for _ in range(warmup):
        make_model().fit(X)
    times = []
    for _ in range(repeats):
        model = make_model()
        start = time.perf_counter()
        model.fit(X)
        times.append(time.perf_counter() - start)
    return {"median_s": float(np.median(times)), "min_s": float(np.min(times)), "repeats": repeats}


def cpu_name() -> str:
    """Human-readable CPU model (Windows registry, /proc/cpuinfo or sysctl)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            return out.stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def system_info() -> dict[str, str | int | None]:
    """Hardware and library versions relevant to the benchmark."""
    import sklearn

    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": f"{blas.get('name', '?')} {blas.get('version', '')}".strip(),
        "scikit_learn": sklearn.__version__,
    }
