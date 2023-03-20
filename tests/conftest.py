"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import low_rank_dataset


@pytest.fixture(params=[(60, 25), (30, 80)], ids=["tall", "wide"])
def dataset(request: pytest.FixtureRequest) -> np.ndarray:
    """A tall (n > p) and a wide (n < p, Gram-trick path) dataset."""
    n, p = request.param
    return low_rank_dataset(n, p, seed=n + p)
