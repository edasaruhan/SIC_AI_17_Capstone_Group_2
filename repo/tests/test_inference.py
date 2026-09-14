"""Tam korpus cikarimi testleri. Model INDIRILMEZ - stub ogrenciyle kosar.

Uc ariza tipi:
  1. Eksik/fazla etiket. Deney 5-core'un HER satiri icin tam olarak bir etiket
     bekliyor; eksik satir sessizce "hediye degil" sayilirdi.
  2. Kesinti. Kaggle oturumu koparsa bitmis is yeniden hesaplanmamali.
  3. Kapi. Sadakat kapisini gecmemis bir ogrencinin etiketi deneye girmemeli.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.data.metadata import build_meta_parquet
from gift_contamination.data.preprocess import build_clean, kcore_parquet_path
from gift_contamination.detection.distill import StubStudent, model_dir
from gift_contamination.detection.distill import report_path as distill_report_path
from gift_contamination.detection.inference import (
    inference_report_path,
    inferred_path,
    input_path,
    parts_dir,
    prepare_inputs,
    run,
)


@pytest.fixture
def kcore(cfg: Config) -> Config:
    build_clean(cfg, "pilot")
    build_meta_parquet(cfg, "pilot")
    return cfg


def _stub_model(cfg: Config) -> None:
    StubStudent(cfg, "base").save(model_dir(cfg, "base"))


def test_prepare_writes_exactly_one_input_row_per_kcore_row(kcore: Config):
    prepare_inputs(kcore, "pilot")

    girdi = pl.read_parquet(input_path(kcore, "pilot"))
    k = pl.read_parquet(kcore_parquet_path(kcore, "pilot"))
    assert girdi.height == k.height
    assert set(girdi["row_id"]) == set(k["row_id"])
    # Ogretmenin girdi bicimi: kategori ve metin alanlari ogrenci metninde
    assert all("Category: Test_Cat" in t and "Review: " in t for t in girdi["text"])


def test_every_kcore_row_gets_exactly_one_label(kcore: Config):
    prepare_inputs(kcore, "pilot")
    _stub_model(kcore)

    run(kcore, "pilot", backend="stub", chunk_rows=2)

    etiket = pl.read_parquet(inferred_path(kcore, "pilot"))
    k = pl.read_parquet(kcore_parquet_path(kcore, "pilot"))
    assert etiket.height == k.height
    assert etiket["row_id"].n_unique() == k.height
    assert etiket.filter(pl.col("purchase_type").is_null()).height == 0


def test_stub_labels_are_stamped_so_the_experiment_can_refuse_them(kcore: Config):
    prepare_inputs(kcore, "pilot")
    _stub_model(kcore)

    run(kcore, "pilot", backend="stub", chunk_rows=2)

    assert pl.read_parquet(inferred_path(kcore, "pilot"))["label_source"].unique().to_list() == ["stub"]
    rapor = json.loads(inference_report_path(kcore, "pilot").read_text(encoding="utf-8"))
    assert rapor["label_source"] == "stub"
    assert set(rapor["seasonality"]) >= {"C1", "C1b"}


def test_a_restart_does_not_recompute_finished_parts(kcore: Config):
    """Kaggle oturumu koparsa bitmis parca yeniden hesaplanmamali."""
    prepare_inputs(kcore, "pilot")
    _stub_model(kcore)
    run(kcore, "pilot", backend="stub", chunk_rows=2)

    ilk = sorted(parts_dir(kcore, "pilot").glob("part_*.parquet"))
    damgalar = {p.name: p.stat().st_mtime_ns for p in ilk}
    inferred_path(kcore, "pilot").unlink()   # kosu "bitmeden" koptu

    run(kcore, "pilot", backend="stub", chunk_rows=2)

    assert {p.name: p.stat().st_mtime_ns for p in ilk} == damgalar
    assert inferred_path(kcore, "pilot").exists()


def test_a_lost_row_fails_loudly(kcore: Config):
    """Eksik etiket sessizce 'hediye degil' sayilmamali."""
    prepare_inputs(kcore, "pilot")
    _stub_model(kcore)
    run(kcore, "pilot", backend="stub", chunk_rows=2)

    parca = sorted(parts_dir(kcore, "pilot").glob("part_*.parquet"))[0]
    pl.read_parquet(parca).head(1).write_parquet(parca)
    inferred_path(kcore, "pilot").unlink()

    with pytest.raises(RuntimeError, match="TAM OLARAK"):
        run(kcore, "pilot", backend="stub", chunk_rows=2)


def test_the_real_model_refuses_a_student_that_did_not_pass(kcore: Config):
    """Kapiyi gecmemis ogrencinin etiketi deneye giremez - model yuklenmeden durur."""
    prepare_inputs(kcore, "pilot")
    p = distill_report_path(kcore, "base")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"model_key": "base"}, "verdict": "FAIL"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="FAIL"):
        run(kcore, "pilot", backend="hf")


def test_a_missing_gate_report_stops_the_real_model(kcore: Config):
    """Yedek model istenip onun raporu yoksa ana modelin PASS'i yerine gecmez."""
    prepare_inputs(kcore, "pilot")
    p = distill_report_path(kcore, "base")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"model_key": "base"}, "verdict": "PASS"}), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        run(kcore, "pilot", model_key="fallback", backend="hf")


def test_the_gate_must_belong_to_the_model_being_used(kcore: Config):
    """Elle tasinmis bir rapor baska modeli temize cikaramaz."""
    prepare_inputs(kcore, "pilot")
    p = distill_report_path(kcore, "base")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"model_key": "fallback"}, "verdict": "PASS"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="modeli icin"):
        run(kcore, "pilot", model_key="base", backend="hf")
