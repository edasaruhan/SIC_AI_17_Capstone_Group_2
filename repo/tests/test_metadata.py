"""Urun metadata testleri.

Metadata annotation kalitesi icin eklendi: LLM "she loved it" cumlesindeki
"it"in ne oldugunu bilmeden etiketliyordu. Buradaki risk sessiz kayip - join
kacirirsa urun adi bos kalir, model eskisi gibi calisir ve kimse fark etmez.
O yuzden kacirma orani OLCULUP raporlaniyor, "kapsam tamdir" varsayilmiyor.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.data.metadata import (
    JOIN_COLUMNS,
    META_SCHEMA,
    build_meta_parquet,
    load_join_frame,
    meta_parquet_path,
    meta_stats_path,
)


def test_build_meta_parquet_reads_only_the_needed_fields(cfg: Config):
    """`description`, `images`, `videos` gibi buyuk alanlar hic ayristirilmamali.

    Toys metadata'si 2,5 GB jsonl; hepsini okumak gereksiz ve pahali.
    """
    dest = build_meta_parquet(cfg, "pilot")
    df = pl.read_parquet(dest)

    assert set(df.columns) == set(META_SCHEMA)
    for heavy in ("description", "images", "videos", "features", "details"):
        assert heavy not in df.columns


def test_parent_asin_is_unique(cfg: Config):
    """Join anahtari benzersiz olmali; degilse ornekleme satir cogaltir."""
    df = pl.read_parquet(build_meta_parquet(cfg, "pilot"))

    assert df["parent_asin"].n_unique() == df.height


def test_duplicate_parent_asin_is_collapsed_and_counted(cfg: Config, tmp_path):
    """Tekrar eden kayit sessizce cogaltilmamali, birlestirilip sayilmali."""
    src = cfg.path("raw") / "raw" / "meta_categories" / "meta_Test_Cat.jsonl"
    extra = {
        "parent_asin": "i1", "title": "Wooden Shape Sorter Cube (2nd listing)",
        "main_category": "Toys & Games", "store": "TestBrand", "price": None,
        "average_rating": 4.4, "rating_number": 10, "categories": [],
    }
    with src.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(extra) + "\n")

    df = pl.read_parquet(build_meta_parquet(cfg, "pilot", force=True))
    stats = json.loads(meta_stats_path(cfg, "pilot").read_text(encoding="utf-8"))

    assert df["parent_asin"].n_unique() == df.height
    assert stats["n_duplicate_parent_asin"] == 1
    # Ilk kayit tutulur: sonradan eklenen liste kazanmamali.
    assert "2nd listing" not in df.filter(pl.col("parent_asin") == "i1")["title"].item()


def test_join_frame_renames_to_avoid_collision(cfg: Config):
    """`main_category` -> `product_category`: dataset kategorisiyle karismasin.

    Ornekte zaten `category` (dataset kategorisi) ve `title` (review basligi)
    var; ayni adlar iki farkli seyi gosterirse analiz sessizce yanlis olur.
    """
    build_meta_parquet(cfg, "pilot")

    frame = load_join_frame(cfg, "pilot")

    assert set(frame.columns) == {"parent_asin", *JOIN_COLUMNS.values()}
    assert "main_category" not in frame.columns
    assert "title" not in frame.columns


def test_missing_metadata_file_says_how_to_get_it(cfg: Config):
    (cfg.path("raw") / "raw" / "meta_categories" / "meta_Test_Cat.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="--meta"):
        build_meta_parquet(cfg, "pilot", force=True)


def test_build_is_idempotent(cfg: Config):
    first = pl.read_parquet(build_meta_parquet(cfg, "pilot"))

    second = pl.read_parquet(build_meta_parquet(cfg, "pilot"))

    assert first.equals(second)


def test_recompute_is_deterministic(cfg: Config):
    first = pl.read_parquet(build_meta_parquet(cfg, "pilot", force=True))

    second = pl.read_parquet(build_meta_parquet(cfg, "pilot", force=True))

    assert first.equals(second)
