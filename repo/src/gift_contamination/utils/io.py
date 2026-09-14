"""Dosya yazma/okuma yardımcıları ve idempotency kontrolü.

CLAUDE.md §7: her komut idempotent olmalı; çıktı varsa `--force` olmadan yeniden
hesaplamamalı.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT


def relative_to_repo(path: Path) -> str:
    """Repo koku altindaki yolu goreli, disaridakini dosya adi olarak dondurur.

    `reports/results/` altindaki JSON'lar commit ediliyor. Icine `str(path)`
    yazmak isletim sistemi kullanici adini ve tam dizin agacini depoya sokar.
    Yol yazan HER modul bunu kullanmali - bir modulde duzeltip digerlerini
    taramamak bu hatayi bir kez daha uretir.
    """
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def code_version() -> str | None:
    """Kosan kodun git surumu; belirlenemezse None.

    Neden rapora giriyor: Kaggle notebook'u depoyu bir kez klonlayip bir daha
    guncellemiyordu ve kosu sessizce eski kodla devam ediyordu (2026-08-28).
    Fark ancak cikti formatindan sezilebiliyordu. Artik ciktilar bir kod
    surumune baglanabiliyor - notebook ciktisi kaybolsa bile.

    Sessizce None doner: git yoksa veya depo degilse kosu durmamali, bu bir
    kayit alani, bir kapi degil. Burada durmasinin sebebi RecBole ortami:
    `llm_annotate` pydantic cekiyor, deney kosucusu cekemez.
    """
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


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
