"""Ortak test fixture'lari.

Gercek veri testte kullanilmaz (CLAUDE.md bolum 10). Butun testler
`tests/fixtures/mini_reviews.jsonl` uzerindeki 9 satirlik sentetik korpusla kosar.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gift_contamination.config import Config

FIXTURES = Path(__file__).parent / "fixtures"

# Fixture k=2 icin tasarlandi: tek gecislik bir k-core implementasyonunun
# yakalayamayacagi bir kaskad iceriyor (bkz. test_preprocess.py).
TEST_K = 2

EXPECTED_STAGES = {
    "01_raw": 9,
    "02_verified_purchase": 8,
    "03_min_words_5": 7,
    "04_dedup": 6,
    f"05_k_core_{TEST_K}": 4,
}


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """tmp_path altina yonlendirilmis, fixture korpusunu isaret eden config."""
    data = {
        "seed": 42,
        "paths": {
            "raw": str(tmp_path / "raw"),
            "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "processed"),
            "annotations": str(tmp_path / "annotations"),
            "human": str(tmp_path / "annotations" / "human"),
            "figures": str(tmp_path / "figures"),
            "results": str(tmp_path / "results"),
        },
        "dataset": {
            "hf_repo": "McAuley-Lab/Amazon-Reviews-2023",
            "categories": {"pilot": "raw_review_Test_Cat"},
            "join_key": "parent_asin",
            "review_file_template": "raw/review_categories/{slug}.jsonl",
            "required_fields": [
                "text",
                "title",
                "rating",
                "timestamp",
                "user_id",
                "parent_asin",
                "verified_purchase",
            ],
            "schema_probe_lines": 1000,
        },
        "preprocess": {
            "verified_only": True,
            "min_words": 5,
            "k_core": TEST_K,
            "dedup": True,
            "timestamp_unit": "auto",
        },
    }
    conf = Config(data, tmp_path / "test.yaml")

    dest = conf.path("raw") / "raw" / "review_categories" / "Test_Cat.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "mini_reviews.jsonl", dest)
    return conf
