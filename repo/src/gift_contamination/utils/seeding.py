"""Seed sabitleme. CLAUDE.md §8, kural 12: seed sabitlemeden deney koşulmaz."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy her ortamda var, yine de zorunlu değil
        pass
    return seed
