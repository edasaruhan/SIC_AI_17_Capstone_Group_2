"""Deneye hangi etiket girer - ve yanlisi girerse ne olur.

Buradaki arizalarin hepsi SESSIZ: deney koşar, sayilar makul gorunur ama
etiketler vekilden, stub'dan ya da kapiyi gecmemis bir ogrenciden gelmistir.
Uc kilit:
  1. `--labels distilled` yalnizca gercek ve kapisi PASS olan ogrenciyi kabul eder.
  2. Farkli etiketle uretilmis atomic dosya "zaten var" diye atlanmaz.
  3. Atomic dosya degistiyse eski kosul dosyasi "zaten var" diye atlanmaz.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.data.preprocess import build_clean, kcore_parquet_path
from gift_contamination.detection.distill import report_path as distill_report_path
from gift_contamination.detection.inference import inference_report_path, inferred_path
from gift_contamination.recsys.atomic import build_atomic, load_distilled_labels, split_path
from gift_contamination.recsys.conditions import build_condition, condition_report_path


@pytest.fixture
def kcore(cfg: Config) -> Config:
    build_clean(cfg, "pilot")
    return cfg


def _ogrenci_ciktisi(cfg: Config, *, kaynak: str = "distilled", kapi: str = "PASS",
                     rapor_kapisi: str = "PASS") -> None:
    """Kaggle'dan indirilmis gibi: etiket parquet'i + cikarim raporu + damitma raporu."""
    k = pl.read_parquet(kcore_parquet_path(cfg, "pilot"), columns=["row_id"])
    etiket = k.with_columns(
        pl.lit("self").alias("purchase_type"),
        pl.lit(kaynak).alias("label_source"),
    )
    yol = inferred_path(cfg, "pilot")
    yol.parent.mkdir(parents=True, exist_ok=True)
    etiket.write_parquet(yol)
    r = inference_report_path(cfg, "pilot")
    r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text(json.dumps({"n_rows": etiket.height, "model_key": "base",
                             "distill_gate_verdict": rapor_kapisi}), encoding="utf-8")
    distill_report_path(cfg, "base").write_text(
        json.dumps({"meta": {"model_key": "base"}, "verdict": kapi}), encoding="utf-8")


# ------------------------------------------------------------ 1. kapi
def test_distilled_labels_from_a_passing_student_are_accepted(kcore: Config):
    _ogrenci_ciktisi(kcore)

    etiket = load_distilled_labels(kcore, "pilot")
    build_atomic(kcore, "pilot", etiket, label_source="distilled")

    assert json.loads(split_path(kcore, "pilot").read_text(encoding="utf-8"))["label_source"] == "distilled"


def test_stub_student_labels_are_refused(kcore: Config):
    _ogrenci_ciktisi(kcore, kaynak="stub")

    with pytest.raises(RuntimeError, match="stub"):
        load_distilled_labels(kcore, "pilot")


@pytest.mark.parametrize(("kapi", "rapor_kapisi"), [("FAIL", "PASS"), ("PASS", "NONE"),
                                                    ("INCOMPLETE", "INCOMPLETE")])
def test_labels_without_a_passed_fidelity_gate_are_refused(kcore: Config, kapi, rapor_kapisi):
    """Cikarim raporundaki damga TEK BASINA yetmez; damitma raporu da diskte PASS olmali."""
    _ogrenci_ciktisi(kcore, kapi=kapi, rapor_kapisi=rapor_kapisi)

    with pytest.raises(RuntimeError, match="kapisi"):
        load_distilled_labels(kcore, "pilot")


def test_proxy_labels_cannot_be_stamped_as_distilled(kcore: Config):
    with pytest.raises(ValueError, match="damgalanamaz"):
        build_atomic(kcore, "pilot", None, label_source="distilled")


# ----------------------------------------------------- 2-3. bayat dosya
def test_an_atomic_file_built_from_other_labels_is_not_silently_reused(kcore: Config):
    from gift_contamination.analysis.keyword_scan import scan_category

    scan_category(kcore, "pilot")
    build_atomic(kcore, "pilot")                       # vekil
    _ogrenci_ciktisi(kcore)

    with pytest.raises(RuntimeError, match="--force"):
        build_atomic(kcore, "pilot", load_distilled_labels(kcore, "pilot"),
                     label_source="distilled")


def test_a_condition_built_from_a_stale_atomic_file_is_not_silently_reused(kcore: Config):
    from gift_contamination.analysis.keyword_scan import scan_category

    scan_category(kcore, "pilot")
    build_atomic(kcore, "pilot")                       # vekil
    build_condition(kcore, "pilot", "C1")
    assert json.loads(condition_report_path(kcore, "pilot", "C1").read_text(encoding="utf-8"))[
        "reportable"] is False

    _ogrenci_ciktisi(kcore)
    build_atomic(kcore, "pilot", load_distilled_labels(kcore, "pilot"),
                 label_source="distilled", force=True)

    with pytest.raises(RuntimeError, match="bayat"):
        build_condition(kcore, "pilot", "C1")
    build_condition(kcore, "pilot", "C1", force=True)
    rapor = json.loads(condition_report_path(kcore, "pilot", "C1").read_text(encoding="utf-8"))
    assert rapor["label_source"] == "distilled" and rapor["reportable"] is True
