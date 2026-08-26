"""Urun metadata'sini jsonl'den parquet'e cevirir.

Neden gerekiyor: LLM bir review'i etiketlerken "she loved it" cumlesindeki
"it"in ne oldugunu bilmiyordu - yalnizca dataset kategorisini (`Toys_and_Games`)
goruyordu. Urun adi prompt'a girince belirsiz vakalarin bir kismi cozulur.

`preprocess.build_raw_parquet` ile ayni strateji: acik sema + `sink_parquet`.
SADECE ihtiyac duyulan alanlar okunur; `description`, `features`, `images`,
`videos`, `details` gibi buyuk/ic ice alanlar hic ayristirilmaz. Toys
metadata'si 2,5 GB jsonl; hepsini ayristirmak gereksiz ve pahali.

Join anahtari `parent_asin` - `asin` DEGIL (CLAUDE.md bolum 8, kural 8).

Kullanim:
    python -m gift_contamination.data.download --category all --meta
    python -m gift_contamination.data.metadata --config configs/base.yaml --category all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..utils.io import should_skip, write_json
from ..utils.logging import get_logger, log_output
from .download import raw_meta_path

log = get_logger("data.metadata")

# Metadata jsonl semasi. `probe_schema` ciktisiyla dogrulandi
# (reports/results/schema_probe_meta_<slug>.json).
#
# `price` ilk 1000 satirda hep null geliyor ama alan string tutuyor ("$12.99"
# bicimi) - String olarak okunuyor, sayiya cevirmek analiz katmaninin isi.
META_SCHEMA: dict[str, object] = {
    "parent_asin": pl.String,
    "title": pl.String,
    "main_category": pl.String,
    "store": pl.String,
    "price": pl.String,
    "average_rating": pl.Float64,
    "rating_number": pl.Int64,
    "categories": pl.List(pl.String),
}

# Ornekleme join'lenirken kullanilan adlar. `main_category` -> `product_category`:
# dataset kategorisi (`Toys_and_Games`) ile karismasin.
JOIN_COLUMNS = {
    "title": "product_title",
    "main_category": "product_category",
}


def meta_parquet_path(cfg: Config, role: str) -> Path:
    return cfg.path("interim", f"{cfg.category_slug(role)}_meta.parquet")


def meta_stats_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"metadata_{cfg.category_slug(role)}.json")


def build_meta_parquet(cfg: Config, role: str, *, force: bool = False) -> Path:
    """jsonl -> parquet, `parent_asin` benzersiz olacak sekilde."""
    src = raw_meta_path(cfg, role)
    dest = meta_parquet_path(cfg, role)
    if should_skip(dest, force, log):
        return dest
    if not src.exists():
        raise FileNotFoundError(
            f"Metadata dosyasi yok: {src}\n"
            "Once indirin: python -m gift_contamination.data.download "
            f"--category {role} --meta"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("metadata jsonl -> parquet: %s", src.name)

    df = (
        pl.scan_ndjson(src, schema=META_SCHEMA, low_memory=True)
        .filter(pl.col("parent_asin").is_not_null())
        .collect()
    )
    n_raw = df.height

    # Ayni parent_asin birden fazla kayit tasiyabiliyor (varyant birlesmesi).
    # Ilk kayit tutulur ve fark loglanir - sessizce cogaltmak join'i patlatir.
    df = df.unique(subset=["parent_asin"], keep="first", maintain_order=True)
    if df.height != n_raw:
        log.warning(
            "%s: %s tekrar eden parent_asin birlestirildi (%s -> %s)",
            role, f"{n_raw - df.height:,}", f"{n_raw:,}", f"{df.height:,}",
        )

    df.write_parquet(dest)
    log_output(log, dest, n_rows=df.height)

    stats = {
        "category": cfg.category_slug(role),
        "n_products_raw": n_raw,
        "n_products": df.height,
        "n_duplicate_parent_asin": n_raw - df.height,
        "columns": df.columns,
        "null_rate": {
            c: round(df[c].null_count() / df.height, 6) if df.height else None
            for c in ("title", "main_category", "store", "price", "average_rating")
        },
    }
    write_json(stats, meta_stats_path(cfg, role), log)
    return dest


def load_join_frame(cfg: Config, role: str) -> pl.DataFrame:
    """Ornekleme join'lenecek ince cerceve: `parent_asin` + yeniden adlandirilmis alanlar."""
    return (
        pl.read_parquet(meta_parquet_path(cfg, role), columns=["parent_asin", *JOIN_COLUMNS])
        .rename(JOIN_COLUMNS)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        build_meta_parquet(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
