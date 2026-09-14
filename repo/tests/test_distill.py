"""Damitma testleri. Model INDIRILMEZ - stub ogrenciyle kosar.

Buradaki en pahali hata gorunmez olan: 500 dogrulama satiri 47.200 LLM
etiketinin ICINDE. Egitime sizarsa "ogrenci insana karsi" olcumu ogrencinin
kendi egitim verisini ezberden okumasini olcer ve sadakat kapisi gercekte
olmayan bir basariyi onaylar. `test_validation_rows_never_enter_training`
bunu kilitliyor.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.analysis.keyword_scan import scan_category
from gift_contamination.config import Config
from gift_contamination.data.labelsheet import LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path
from gift_contamination.data.sampling import build_sample, validation_path
from gift_contamination.detection.distill import (
    SPLIT_HOLDOUT,
    _training_arguments,
    bundle_path,
    bundle_report_path,
    fidelity_verdict,
    human_path,
    prepare,
    report_path,
    student_text,
    token_batches,
    train,
)
from gift_contamination.detection.llm_annotate import (
    MAX_REVIEW_CHARS,
    UNKNOWN_PRODUCT,
    annotate,
    annotation_path,
)
from gift_contamination.utils.io import read_json


def _annotated(cfg: Config, *, n_val: int = 2) -> tuple[Config, pl.DataFrame]:
    """Stub LLM etiketleri + ilk `n_val` satirdan kurulmus tek etiketleyicili dogrulama."""
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    dest = annotate(cfg, "pilot", backend_name="stub")
    labels = pl.read_parquet(dest).with_columns(pl.lit("vllm").alias("backend"))
    labels.write_parquet(dest)

    val = labels.head(n_val)
    cfg._data["validation"].update({"annotators": ["A"], "reliability": "none", "n": n_val})
    csv = labeled_path(validation_path(cfg, n_val))
    csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "val_id": list(range(1, n_val + 1)),
        "category": val["category"],
        "row_id": val["row_id"],
        "sample_frame": val["sample_frame"],
        "purchase_type": val["purchase_type"],
        "val_stratum": val["purchase_type"],
        "label_A": val["purchase_type"],
        "notes_A": [None] * n_val,
        LABEL_SOURCE_COLUMN: [LABEL_SOURCE] * n_val,
    }).write_csv(csv)
    return cfg, val


# ------------------------------------------------------------------- sizinti
def test_validation_rows_never_enter_training(cfg: Config):
    """SIZINTI KURALI - DEGISTIRILEMEZ."""
    cfg, val = _annotated(cfg, n_val=2)

    prepare(cfg, ["pilot"])

    bundle = pl.read_parquet(bundle_path(cfg))
    kesisim = bundle.join(val.select("category", "row_id"), on=["category", "row_id"], how="inner")
    assert kesisim.height == 0
    rapor = read_json(bundle_report_path(cfg))
    assert rapor["n_excluded_validation_rows"] == 2
    # Insan satirlari ayri dosyada, egitimle hic karismadan
    assert pl.read_parquet(human_path(cfg)).height == 2


def test_a_validation_row_that_cannot_be_excluded_fails_loudly(cfg: Config):
    """Dislanamayan dogrulama satiri = ogrencinin sinav sorusunu egitimde gormesi."""
    cfg, val = _annotated(cfg, n_val=2)
    csv = labeled_path(validation_path(cfg, 2))
    pl.read_csv(csv).with_columns(pl.lit(999_999).alias("row_id")).write_csv(csv)

    with pytest.raises(RuntimeError, match="dislanabildi"):
        prepare(cfg, ["pilot"])


def test_a_stub_teacher_cannot_teach(cfg: Config):
    """Kuru kosu LLM ciktisi ogretmen olamaz (gate1 ile ayni desen)."""
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    annotate(cfg, "pilot", backend_name="stub")   # backend DUZELTILMIYOR

    with pytest.raises(RuntimeError, match="[Kk]uru kosu"):
        prepare(cfg, ["pilot"])


def test_parse_failures_do_not_become_training_labels(cfg: Config):
    cfg, _ = _annotated(cfg, n_val=1)
    path = annotation_path(cfg, "pilot")
    df = pl.read_parquet(path)
    bozuk = df["row_id"][-1]
    df.with_columns(
        pl.when(pl.col("row_id") == bozuk).then(False).otherwise(pl.col("parse_ok")).alias("parse_ok")
    ).write_parquet(path)

    prepare(cfg, ["pilot"])

    assert bozuk not in pl.read_parquet(bundle_path(cfg))["row_id"].to_list()
    assert read_json(bundle_report_path(cfg))["n_excluded_parse_failures"] == 1


# --------------------------------------------------------------------- girdi
def test_student_sees_the_same_four_fields_as_the_teacher():
    """Farkli girdi, sadakat olcumunu model farkiyla girdi farkini karistiran bir sayiya cevirirdi."""
    out = student_text("Toys_and_Games", "Wooden Train", "Great", "My grandson loves it")

    assert "Toys_and_Games" in out
    assert "Wooden Train" in out
    assert "Great" in out
    assert "My grandson loves it" in out
    # Urun adi yoksa ogretmenin gordugu yer tutucu
    assert UNKNOWN_PRODUCT in student_text("X", None, "t", "x")
    # Ayni kirpma: 6.000 karakter
    uzun = student_text("X", "p", "t", "z" * (MAX_REVIEW_CHARS + 500))
    assert len(uzun.split("Review: ", 1)[1]) == MAX_REVIEW_CHARS


def test_the_split_is_deterministic(cfg: Config):
    cfg, _ = _annotated(cfg, n_val=1)

    prepare(cfg, ["pilot"])
    ilk = pl.read_parquet(bundle_path(cfg)).select("row_id", "split")
    prepare(cfg, ["pilot"], force=True)
    ikinci = pl.read_parquet(bundle_path(cfg)).select("row_id", "split")

    assert ilk.equals(ikinci)
    # Kurgu gercekten ayirma yapmali; yoksa test belirleyiciligi hic sinamaz.
    assert SPLIT_HOLDOUT in ilk["split"].to_list(), "fixture ayrilmis satir uretmedi"
    assert "train" in ilk["split"].to_list()


# ------------------------------------------------------------- GPU tarafi
def test_token_batches_cover_every_row_once_and_respect_the_budget():
    """Tahmin gruplari: her satir TAM bir kez, satir x en uzun <= butce."""
    uzunluk = [5, 300, 7, 1024, 12, 12, 900, 3, 64, 1024]

    gruplar = token_batches(uzunluk, max_tokens=1100, max_rows=3)

    hepsi = sorted(int(i) for g in gruplar for i in g)
    assert hepsi == list(range(len(uzunluk)))
    for g in gruplar:
        assert len(g) <= 3
        # Butceyi tek basina asan satir kendi grubunda gidebilir; baska turlu asilamaz
        assert len(g) == 1 or len(g) * max(uzunluk[i] for i in g) <= 1100


def test_training_arguments_match_both_transformers_generations():
    """Kaggle imajindaki surum bilinmiyor; yanlis ad model indirildikten sonra patlardi."""
    base = dict(warmup_ratio=0.1, group_by_length=True, eval_strategy="no", lr=1)

    eski = _training_arguments({"warmup_ratio", "group_by_length", "evaluation_strategy"}, **base)
    assert eski == {"lr": 1, "warmup_ratio": 0.1, "group_by_length": True,
                    "evaluation_strategy": "no"}

    yeni = _training_arguments({"warmup_steps", "train_sampling_strategy", "eval_strategy"}, **base)
    assert yeni == {"lr": 1, "warmup_steps": 0.1, "train_sampling_strategy": "group_by_length",
                    "eval_strategy": "no"}


# ---------------------------------------------------------------------- kapi
def _axes(c1: float | None, c1b: float | None) -> dict:
    return {"C1": {"f1": {"value": c1}}, "C1b": {"f1": {"value": c1b}}}


def _teacher(cfg: Config, f1: float | None) -> None:
    import json

    from gift_contamination.analysis.validation import validation_report_path

    p = validation_report_path(cfg, int(cfg.get("validation.n")))
    p.parent.mkdir(parents=True, exist_ok=True)
    if f1 is None:
        return
    p.write_text(json.dumps({"measurements": {"contamination_axes": {"sample": {
        "C1": {"f1": {"value": f1}}}}}}), encoding="utf-8")


def test_the_gate_passes_only_when_all_three_criteria_hold(cfg: Config):
    _teacher(cfg, 0.731)

    ok = fidelity_verdict(cfg, _axes(0.90, 0.88), _axes(0.70, 0.80))
    assert ok["verdict"] == "PASS"

    # Ogretmene sadik ama insana karsi ogretmenden 0,05'ten fazla kotu
    kotu_insan = fidelity_verdict(cfg, _axes(0.90, 0.88), _axes(0.67, 0.80))
    assert kotu_insan["verdict"] == "FAIL"
    assert kotu_insan["criteria"]["3_c1_axis_vs_human_not_worse_than_teacher"]["passed"] is False

    kotu_c1b = fidelity_verdict(cfg, _axes(0.90, 0.84), _axes(0.70, 0.80))
    assert kotu_c1b["verdict"] == "FAIL"


def test_the_gate_is_incomplete_without_the_teachers_measured_f1(cfg: Config):
    """Ogretmenin F1'i RAPORDAN okunur; rapor yoksa karar verilmez - PASS da yazilmaz."""
    _teacher(cfg, None)

    out = fidelity_verdict(cfg, _axes(0.95, 0.95), _axes(0.95, 0.95))

    assert out["verdict"] == "INCOMPLETE"


def test_the_fallback_model_runs_only_after_the_base_model_failed(cfg: Config):
    """Onceden kayitli kural: iki modeli egitip iyisini secmek model secimi olurdu."""
    cfg, _ = _annotated(cfg, n_val=2)
    prepare(cfg, ["pilot"])

    with pytest.raises(RuntimeError, match="yalnizca ana model FAIL"):
        train(cfg, model_key="fallback", backend="stub")   # ana model hic kosmadi

    train(cfg, model_key="base", backend="stub")           # stub -> INCOMPLETE, FAIL degil
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        train(cfg, model_key="fallback", backend="stub")

    ana = report_path(cfg, "base")
    rapor = read_json(ana)
    rapor["verdict"] = "FAIL"
    ana.write_text(json.dumps(rapor), encoding="utf-8")
    yedek = train(cfg, model_key="fallback", backend="stub")

    assert yedek["meta"]["model_key"] == "fallback"
    assert read_json(ana)["verdict"] == "FAIL", "yedegin kosusu ana modelin kanitini ezmemeli"
    assert report_path(cfg, "fallback").exists()


def test_a_stub_student_can_never_pass(cfg: Config):
    """Stub ogrenciyle kurulmus bir kapi karari gecersizdir."""
    cfg, _ = _annotated(cfg, n_val=2)
    prepare(cfg, ["pilot"])

    rapor = train(cfg, backend="stub")

    assert rapor["verdict"] == "INCOMPLETE"
    assert "stub" in rapor
    assert rapor["meta"]["model_selection"].startswith("yok")
