"""Dosya yazma/okuma yardımcıları ve idempotency kontrolü.

CLAUDE.md §7: her komut idempotent olmalı; çıktı varsa `--force` olmadan yeniden
hesaplamamalı.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def should_skip(path: Path, force: bool, logger: logging.Logger) -> bool:
    """Çıktı zaten üretilmişse True döner ve nedenini loglar."""
    if path.exists() and not force:
        logger.info("atlanıyor (çıktı mevcut, --force ile ezebilirsiniz): %s", path)
        return True
    return False


def write_json(obj: Any, path: Path, logger: logging.Logger | None = None) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    if logger:
        logger.info("yazıldı: %s", path)
    return path


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
