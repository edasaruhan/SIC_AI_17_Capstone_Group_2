"""Config yükleme ve yol çözümleme.

Projenin tek doğruluk kaynağı (CLAUDE.md §2). Kodda hardcode path veya parametre yok;
her şey `configs/base.yaml` üzerinden okunur (CLAUDE.md §8, kural 11).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/gift_contamination/config.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = "configs/base.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Config eksik, bulunamadı veya tutarsız."""


class Config:
    """`configs/base.yaml` üzerine ince bir sarmalayıcı.

    Erişim noktalı anahtarla yapılır (`cfg.get("preprocess.min_words")`); eksik anahtar
    sessizce None dönmez, `ConfigError` atar. Amaç, yazım hatasının deneyi sessizce
    bozmasını engellemek.
    """

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    # ------------------------------------------------------------------ yükleme
    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        p = Path(path or DEFAULT_CONFIG)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise ConfigError(f"Config bulunamadı: {p}")
        with p.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError(f"Config bir sözlük değil: {p}")
        load_dotenv(REPO_ROOT / ".env")
        return cls(data, p)

    # ------------------------------------------------------------------- erişim
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Config anahtarı yok: '{dotted}' ({self.source})")
                return default
            node = node[part]
        return node

    def path(self, key: str, *parts: str) -> Path:
        """`paths.<key>` değerini mutlak yola çevirir; göreliyse repo köküne bağlar."""
        raw = Path(self.get(f"paths.{key}"))
        base = raw if raw.is_absolute() else REPO_ROOT / raw
        return base.joinpath(*parts)

    # -------------------------------------------------------- kategori yardımcıları
    def category_roles(self) -> list[str]:
        """Config'te tanımlı rol adları: pilot, high, mid, low."""
        return list(self.get("dataset.categories").keys())

    def category_config_name(self, role: str) -> str:
        """'pilot' -> 'raw_review_All_Beauty' (HuggingFace config adı)."""
        cats = self.get("dataset.categories")
        if role not in cats:
            raise ConfigError(
                f"Bilinmeyen kategori rolü: '{role}'. Tanımlı olanlar: {sorted(cats)}"
            )
        return cats[role]

    def category_slug(self, role: str) -> str:
        """'pilot' -> 'All_Beauty' (HuggingFace dosya adı, uzantısız)."""
        name = self.category_config_name(role)
        return name.split("raw_review_", 1)[-1]

    def __repr__(self) -> str:  # pragma: no cover - hata ayıklama kolaylığı
        return f"Config(source={self.source})"


def add_standard_args(parser: argparse.ArgumentParser, *, category: bool = True,
                      force: bool = True) -> None:
    """CLAUDE.md §7'deki komut sözleşmesinin ortak bayrakları.

    `force=False`: komut zaten her çağrıldığında yeniden hesaplıyorsa bayrak
    EKLENMEZ. `eda` ve `deep_eda` onu kabul edip sessizce yok sayıyordu —
    kullanıcıya olmayan bir davranış vaat eden bir bayrak, hiç olmayandan
    kötüdür (denetim 2026-09-20).
    """
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config yolu")
    if category:
        parser.add_argument(
            "--category",
            default="pilot",
            help="Kategori rolü (pilot/high/mid/low) veya tümü için 'all'",
        )
    if force:
        parser.add_argument(
            "--force",
            action="store_true",
            help="Çıktı zaten varsa bile yeniden hesapla",
        )


def resolve_roles(cfg: Config, category: str) -> list[str]:
    """'all' -> tüm roller; aksi halde tek elemanlı liste (doğrulanmış)."""
    if category == "all":
        return cfg.category_roles()
    cfg.category_config_name(category)  # doğrulama; bilinmeyen rolde ConfigError atar
    return [category]
