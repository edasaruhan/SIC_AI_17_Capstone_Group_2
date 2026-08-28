"""Kapi 1 testleri.

Bu kapinin isi bir SAYIYI degil bir KARARI uretmek. En tehlikeli ariza, bozuk
bir kosunun "gecti" damgasi almasi - o durumda 40.000 satir bozuk bir
detektorle etiketlenir ve hata Hafta 4'te gorulur.

Sinanan sey: esikler gercekten baglayici mi, atlanan olcut sessizce gecmis
sayiliyor mu, kuru kosu ciktisi kapiya girebiliyor mu.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.analysis.gate1 import (
    beyond_keyword,
    evaluate,
    gate1_path,
    load_joined,
    schema_health,
    seasonality,
    trial_agreement,
)
from gift_contamination.analysis.keyword_scan import PROXY_COL, scan_category
from gift_contamination.config import Config
from gift_contamination.data.sampling import build_sample
from gift_contamination.detection.llm_annotate import (
    annotate,
    annotation_path,
    trial_annotation_path,
)


@pytest.fixture
def annotated(cfg: Config) -> Config:
    """Fixture korpusu uzerinde stub kosusu; ciktinin backend'i sonra duzeltilir.

    Kapi gercek kosu bekliyor (ve bunu dogru sekilde reddediyor); testte o
    kontrolu asmak icin backend etiketi elle `vllm` yapiliyor. Etiketlerin
    kendisi zaten taklit - burada sinanan sey karar mantigi, model degil.
    """
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    dest = annotate(cfg, "pilot", backend_name="stub")
    pl.read_parquet(dest).with_columns(
        pl.lit("vllm").alias("backend")
    ).write_parquet(dest)
    return cfg


def _main_frame(cfg: Config, rows: list[dict]) -> pl.DataFrame:
    """Elle kurulmus `main` cercevesi: olcut mantigini izole eder."""
    return pl.DataFrame(rows, schema={
        "month": pl.Int8, "purchase_type": pl.String, PROXY_COL: pl.Boolean,
    })


# --------------------------------------------------------------- 1) mevsimsellik
def test_flat_curve_fails_seasonality(cfg: Config):
    """Duz egri KALMALI - 'sinyal yok' arizasi tam olarak bu."""
    rows = [{"month": m, "purchase_type": ("gift_given" if i % 10 == 0 else "self"),
             PROXY_COL: False}
            for m in range(1, 13) for i in range(200)]

    result = seasonality(cfg, _main_frame(cfg, rows))

    assert result["ratio"] == pytest.approx(1.0, abs=0.01)
    assert result["passed"] is False


def test_december_peak_passes_seasonality(cfg: Config):
    """Aralik-Ocak'ta 2x oran esigi gecmeli."""
    rows = []
    for m in range(1, 13):
        share = 4 if m in (12, 1) else 10   # her 4'te 1 vs her 10'da 1
        rows += [{"month": m, "purchase_type": ("gift_given" if i % share == 0 else "self"),
                  PROXY_COL: False} for i in range(400)]

    result = seasonality(cfg, _main_frame(cfg, rows))

    assert result["ratio"] > 2.0
    assert result["ci95"][0] > 1.0
    assert result["passed"] is True


def test_seasonality_needs_the_ci_not_just_the_point_estimate(cfg: Config):
    """Nokta tahmin esigi gecse bile GA 1,0'i kapsiyorsa kapi ACILMAZ.

    Az veriyle 1,3x oran kolayca sanstan cikar; esigi tek basina nokta tahmine
    baglamak Kapi 1'i anlamsizlastirir.
    """
    rows = [{"month": 12, "purchase_type": "gift_given", PROXY_COL: False}] * 2
    rows += [{"month": 12, "purchase_type": "self", PROXY_COL: False}] * 3
    rows += [{"month": 7, "purchase_type": "gift_given", PROXY_COL: False}] * 2
    rows += [{"month": 7, "purchase_type": "self", PROXY_COL: False}] * 5

    result = seasonality(cfg, _main_frame(cfg, rows))

    assert result["ratio"] >= result["threshold"]
    assert result["ci95"][0] < 1.0
    assert result["passed"] is False


# ---------------------------------------------------------- 2) vekil otesi
def test_keyword_mimicry_fails(cfg: Config):
    """LLM yalnizca vekilin isaretledigi satirlara hediye diyorsa KALMALI.

    Bu durumda bedava bir regex'in isini pahaliya tekrar ediyoruz ve butun
    damitma yigini gereksiz.
    """
    rows = [{"month": 6, "purchase_type": "gift_given", PROXY_COL: True}] * 100
    rows += [{"month": 6, "purchase_type": "self", PROXY_COL: False}] * 900

    result = beyond_keyword(cfg, _main_frame(cfg, rows))

    assert result["rate"] == 0.0
    assert result["passed"] is False


def test_finding_gifts_the_keyword_missed_passes(cfg: Config):
    rows = [{"month": 6, "purchase_type": "gift_given", PROXY_COL: True}] * 100
    rows += [{"month": 6, "purchase_type": "gift_given", PROXY_COL: False}] * 30
    rows += [{"month": 6, "purchase_type": "self", PROXY_COL: False}] * 870

    result = beyond_keyword(cfg, _main_frame(cfg, rows))

    assert result["rate"] == pytest.approx(30 / 900, abs=1e-6)
    assert result["passed"] is True


# ------------------------------------------------------------ 3) sema sagligi
def test_schema_health_reads_the_run_report(annotated: Config):
    result = schema_health(annotated, "pilot")

    assert result["passed"] is True
    assert result["parse_fail_rate"] == 0.0


def test_high_parse_failure_fails_the_gate(annotated: Config, monkeypatch):
    from gift_contamination.detection import llm_annotate

    path = llm_annotate.annotation_stats_path(annotated, "pilot")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["quality"]["parse_fail_rate"] = 0.25
    path.write_text(json.dumps(report), encoding="utf-8")

    assert schema_health(annotated, "pilot")["passed"] is False


# ------------------------------------------------------------- 4) duman testi
def test_missing_trial_run_is_skipped_not_passed(annotated: Config):
    """Atlanan olcut GECMIS sayilmamali.

    Sessizce 'gecti' demek, kapiyi hic kurmamis olmakla ayni sey.
    """
    result = trial_agreement(annotated)

    assert result["skipped"] is True
    assert result["passed"] is None


def test_skipped_criterion_blocks_the_verdict(annotated: Config):
    """Deneme kosusu yoksa karar PASS degil INCOMPLETE olmali."""
    report = evaluate(annotated, "pilot")

    assert report["verdict"] in {"FAIL", "INCOMPLETE"}
    assert "4_trial_smoke_test" in report["skipped"]


# ---------------------------------------------------------------- guvenlik
def test_stub_output_cannot_enter_the_gate(cfg: Config):
    """Taklit etiketlerle Kapi 1 kararı verilemez."""
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    annotate(cfg, "pilot", backend_name="stub")

    with pytest.raises(RuntimeError, match="Kuru kosu"):
        load_joined(cfg, "pilot")


def test_report_records_the_prompt_version(annotated: Config):
    """Hangi prompt'la verilmis bir karar oldugu rapordan okunabilmeli."""
    evaluate(annotated, "pilot")

    report = json.loads(gate1_path(annotated, "pilot").read_text(encoding="utf-8"))

    assert report["meta"]["prompt_version"] == "v3"
    assert report["meta"]["thresholds_fixed"].startswith("2026-08-28")


def test_boost_frames_are_excluded_from_the_verdict(annotated: Config):
    """Yaygınlık yalnizca `main`den okunur; boost havuzlanirsa oran 7 kat siser.

    Fixture korpusu 6 satirlik ve boost havuzlari bos kaliyor, o yuzden boost
    satiri elle ekleniyor - sinanan sey ornekleme degil, kapinin cerceveyi
    ayirip ayirmadigi.
    """
    path = annotation_path(annotated, "pilot")
    base = pl.read_parquet(path)
    n_main = base.height
    boosted = pl.concat([
        base,
        base.head(2).with_columns(
            pl.lit("boost").alias("sample_frame"),
            # row_id catismasin: `unique(row_id, category)` bunlari yutardi.
            (pl.col("row_id") + 10_000).alias("row_id"),
        ),
    ])
    boosted.write_parquet(path)

    # Ornekleme cercevesinde de karsiliklari olmali, aksi halde join bos birakir.
    sample_path = annotated.path("interim", "Test_Cat_annotation_sample.parquet")
    sample = pl.read_parquet(sample_path)
    pl.concat([
        sample,
        sample.head(2).with_columns((pl.col("row_id") + 10_000).alias("row_id")),
    ]).write_parquet(sample_path)

    report = evaluate(annotated, "pilot")

    assert report["meta"]["n_main"] == n_main
    assert n_main < boosted.height


def _write_trial_pair(
    cfg: Config, human: list[str], llm: list[str], *, backend: str = "vllm"
) -> None:
    """Deneme setinin iki yakasini kurar: insan CSV'si + LLM parquet'i."""
    ids = list(range(1, len(human) + 1))
    csv = cfg.path("human", "prompt_trial_200_labeled.csv")
    csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"trial_id": ids, "label": human}).write_csv(csv)

    dest = trial_annotation_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "trial_id": ids,
        "purchase_type": llm,
        "backend": [backend] * len(llm),
    }).write_parquet(dest)


def test_trial_agreement_is_measured_against_the_human_labels(cfg: Config):
    """Uyum, insan etiketiyle LLM etiketinin BIREBIR ortusmesi."""
    _write_trial_pair(
        cfg,
        human=["self", "gift_given", "household", "unclear"],
        llm=["self", "gift_given", "household", "self"],
    )

    result = trial_agreement(cfg)

    assert result["skipped"] is False
    assert result["n_compared"] == 4
    assert result["agreement"] == 0.75


def test_llm_only_received_label_counts_as_disagreement(cfg: Config):
    """`received` sema v3'te var, v2 rehberinde YOKTU.

    Insanin secemedigi bir etiketi uyusmazlik saymak olcutu ZORLASTIRIR;
    disarida birakmak uyumu sisirirdi. Sayisi ayrica raporlanmali ki secim
    gorunur kalsin.
    """
    _write_trial_pair(
        cfg,
        human=["gift_given", "gift_given", "self", "self"],
        llm=["received", "gift_given", "self", "self"],
    )

    result = trial_agreement(cfg)

    assert result["agreement"] == 0.75
    assert result["n_llm_received"] == 1


def test_stub_trial_run_cannot_enter_the_gate(cfg: Config):
    """Kuru kosunun etiketleri RASTGELE; uyum sayisi da rastgele cikar.

    Taklit bir kosu bu olcutu gecirebilir de bosuna dusurebilir de - iki yon de
    kabul edilemez, cunku ortaya gercek gorunen bir karar cikar.
    """
    _write_trial_pair(
        cfg, human=["self", "self"], llm=["self", "self"], backend="stub",
    )

    with pytest.raises(RuntimeError, match="Kuru kosu"):
        trial_agreement(cfg)


def test_disagreements_are_reported_by_direction(cfg: Config):
    """Tek bir uyum yuzdesi hatanin YONUNU gizler.

    `household` -> `gift_given` ile `gift_given` -> `household` ayni uyum
    sayisini verir ama bambaska iki sorundur; ilki C1'in tanimini (CLAUDE.md
    13) dogrudan ilgilendirir. Uyusanlar listeye GIRMEZ.
    """
    _write_trial_pair(
        cfg,
        human=["household", "household", "self", "self"],
        llm=["gift_given", "gift_given", "self", "unclear"],
    )

    result = trial_agreement(cfg)

    assert result["disagreements"] == {"household->gift_given": 2, "self->unclear": 1}
    assert "self->self" not in result["disagreements"]


def test_per_class_separates_over_calling_from_missing(cfg: Config):
    """Az cagirmak ile yanlis cagirmak ayni sayiya dusmemeli.

    LLM `gift_given`i FAZLA cagiriyorsa recall yuksek precision dusuk cikar;
    `household`i ATLIYORSA tam tersi. Genel uyum ikisini de gizler.
    """
    _write_trial_pair(
        cfg,
        human=["household", "household", "gift_given", "self"],
        llm=["gift_given", "household", "gift_given", "self"],
    )

    result = trial_agreement(cfg)

    assert result["per_class"]["household"] == {
        "n_human": 2, "n_llm": 1, "precision": 1.0, "recall": 0.5,
    }
    assert result["per_class"]["gift_given"] == {
        "n_human": 1, "n_llm": 2, "precision": 0.5, "recall": 1.0,
    }
