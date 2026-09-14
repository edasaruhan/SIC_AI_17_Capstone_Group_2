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
            "models": str(tmp_path / "models"),
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
            "meta_file_template": "raw/meta_categories/meta_{slug}.jsonl",
            "meta_required_fields": ["parent_asin", "title"],
            "schema_probe_lines": 1000,
        },
        "preprocess": {
            "verified_only": True,
            "min_words": 5,
            "k_core": TEST_K,
            "dedup": True,
            "timestamp_unit": "auto",
        },
        # Desen dosyasi gercek olan; sentetik bir kopya semantigi kaydirir ve
        # test_keyword_scan'in kilitledigi ayrimlari dogrulamaz hale getirir.
        "keywords": {
            "path": "configs/gift_keywords.yaml",
            "precision_sample_n": 4,
            "precision_seed": 42,
        },
        # Fixture 6 satirlik temiz korpus birakiyor; n_annotate onun USTUNDE
        # secildi ki tahsis "havuzda olandan fazlasini isteme" yolunu da gecsin.
        "sampling": {
            "n_annotate": 8,
            "strata": ["month", "rating", "text_length_bucket"],
            "text_length_buckets": [15, 40],
            "keyword_boost_fraction": 0.25,
            "received_boost_fraction": 0.25,
        },
        # Prompt ve sema dosyalari GERCEK olanlar - keywords.path ile ayni
        # gerekce. Sentetik bir prompt, `llm_annotate`in dogrulamasi gereken
        # sozlesmeyi (yer tutucular, enum'lar, versiyon satiri) sinamaz.
        "detection": {
            "primary_model": "stub-model",   # stub backend model indirmez
            "dtype": "float16",
            "max_new_tokens": 220,
            "temperature": 0.0,
            "retry_on_parse_fail": 1,
            "schema_path": "configs/annotation_schema.json",
            "prompt_path": "prompts/gift_detection_v3.md",
            "enable_prefix_caching": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            # Testte GPU yok; stub backend zaten `gpus`i 1'e zorluyor.
            "gpus": "auto",
            "tensor_parallel_size": 1,
            # Kasitli kucuk: fixture ornekleminde birden fazla parca olussun ki
            # devam-ettirme yolu (bitmis satirlari atlama) gercekten sinansin.
            "shard_rows": 3,
        },
        # gate1 figuru (F17) bu iki anahtari okuyor.
        "eda": {"figure_dpi": 72, "figure_format": "png"},
        # Hafta 5. Esikler configs/base.yaml ile AYNI (gate1 gerekcesi). Testler
        # yalnizca `stub` ogrenciyle kosar - model indirilmez, GPU gerekmez.
        "distill": {
            "base_model": "answerdotai/ModernBERT-base",
            "fallback_model": "microsoft/deberta-v3-base",
            "max_length": 1024,
            "epochs": 3,
            "batch_size": 32,
            "lr": 3.0e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_checkpointing": True,
            "val_fraction": 0.25,   # fixture kucuk: katmanlarda ayirma gercekten olsun
            "fidelity": {
                "c1_axis_f1_min": 0.85,
                "c1b_axis_f1_min": 0.85,
                "max_drop_vs_teacher_on_human": 0.05,
            },
        },
        # Esikler configs/base.yaml ile AYNI. Testin isi esik degerini degil,
        # esigin baglayici olup olmadigini sinamak; farkli bir deger koymak
        # "gercek config'te de boyle mi" sorusunu cevapsiz birakirdi.
        "gate1": {
            "seasonality_ratio_min": 1.25,
            "seasonality_bootstrap_n": 400,   # testte 2000 gereksiz yavas
            "beyond_keyword_rate_min": 0.01,
            "parse_fail_rate_max": 0.01,
            "span_downgrade_rate_max": 0.10,
            "trial_agreement_min": 0.70,
            "peak_months": [12, 1],
            "trough_months": [6, 7, 8, 9],
        },
        # Hafta 4. `n` ve `strata` fixture'in kucuklugune gore olceklendi;
        # `kappa_min` ve `min_class_n_for_kappa` ise configs/base.yaml ile
        # AYNI - gate1'deki gerekce: testin isi esigin baglayici olup
        # olmadigini sinamak, farkli bir deger uydurmak degil.
        "validation": {
            "n": 5,
            # Testler ÖZGÜN üç kişilik yolu sınıyor; tek etiketleyici yolu
            # (production: [A] + none) testlerde config ezilerek sınanıyor.
            "annotators": ["A", "B", "C"],
            "reliability": "fleiss",
            "min_cell_n": 5,
            "bootstrap_n": 200,   # testte 2000 gereksiz yavaş
            "strata": {
                "gift_given": 2, "household": 1,
                "self": 1, "unclear": 1, "received": 0,
            },
            "role_weights": {"pilot": 1},
            "kappa_min": 0.60,
            "min_class_n_for_kappa": 20,
        },
    }
    conf = Config(data, tmp_path / "test.yaml")

    dest = conf.path("raw") / "raw" / "review_categories" / "Test_Cat.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "mini_reviews.jsonl", dest)

    # Urun metadata'si. `i5`in basligi kasitli olarak BOS: ornekleme "urun adi
    # yok" yolunu da gecsin ve n_missing_product_title sayaci sinansin.
    meta = conf.path("raw") / "raw" / "meta_categories" / "meta_Test_Cat.jsonl"
    meta.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "mini_meta.jsonl", meta)
    return conf
