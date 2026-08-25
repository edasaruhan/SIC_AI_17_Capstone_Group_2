"""Tek biçimli logging.

CLAUDE.md §7: her komut çıktı yolunu ve satır sayısını loglamalı. `log_output` bunu
tek bir yerde standartlaştırır, böylece her modül aynı biçimde rapor verir.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = "gift_contamination"
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    root = logging.getLogger(_ROOT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(handler)
        root.propagate = False
    root.setLevel(level)
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def human_bytes(n: int | float) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= step
    return f"{n:.1f} PB"


def log_output(
    logger: logging.Logger,
    path: Path,
    *,
    n_rows: int | None = None,
    note: str | None = None,
) -> None:
    """Çıktı yolunu, satır sayısını ve dosya boyutunu tek satırda raporlar."""
    parts = [f"yazıldı: {path}"]
    if n_rows is not None:
        parts.append(f"{n_rows:,} satır")
    if path.exists():
        parts.append(human_bytes(path.stat().st_size))
    if note:
        parts.append(note)
    logger.info(" | ".join(parts))
