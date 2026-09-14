"""RQ1 yaygınlık tablosunun testleri.

Bu tablo projenin ANA IDDIASI. Bozulursa makaledeki ilk sayi yanlis olur ve
hicbir sey gurultu cikarmaz - o yuzden dort ayri ariza tipi burada kapatiliyor:

  1. Cerceve karismasi. `boost` havuzlanirsa oran 7 kat siser (olculdu).
  2. Guven araligi. Kucuk oranlarda Wald araligi sinirin disina tasar.
  3. Kuru kosu sizmasi. Taklit etiketten yaygınlık raporu uretilemez.
  4. Hesaplanmamis iddia. "Insan tarafindan dogrulandi" DISKTEN okunmali,
     elle yazilmamali (denetim bulgusu 7).
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.analysis.keyword_scan import scan_category
from gift_contamination.analysis.prevalence import (
    DEFINITIONS,
    category_stats,
    evaluate,
    load_main,
    prevalence_path,
    wilson,
)
from gift_contamination.analysis.validation import validation_report_path
from gift_contamination.config import Config
from gift_contamination.data.sampling import build_sample
from gift_contamination.detection.llm_annotate import annotate, annotation_path


@pytest.fixture
def annotated(cfg: Config) -> Config:
    """Fixture korpusu uzerinde stub kosusu; backend etiketi sonra duzeltilir.

    `test_gate1.annotated` ile ayni desen ve ayni gerekce: kapi gercek kosu
    bekliyor (ve dogru sekilde reddediyor), testte o kontrol elle asiliyor.
    Sinanan sey karar mantigi, model degil.
    """
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    dest = annotate(cfg, "pilot", backend_name="stub")
    pl.read_parquet(dest).with_columns(
        pl.lit("vllm").alias("backend")
    ).write_parquet(dest)
    return cfg


def _relabel(cfg: Config, labels: list[str]) -> None:
    """`main` cercevesindeki etiketleri sirayla degistirir."""
    path = annotation_path(cfg, "pilot")
    df = pl.read_parquet(path)
    main_idx = [i for i, f in enumerate(df["sample_frame"]) if f == "main"]
    yeni = list(df["purchase_type"])
    for i, lbl in zip(main_idx, labels):
        yeni[i] = lbl
    df.with_columns(
        pl.Series("purchase_type", yeni, dtype=pl.String)
    ).write_parquet(path)


# ------------------------------------------------------------- guven araligi
def test_wilson_matches_a_hand_computed_interval():
    """25/100 icin Wilson araligi ders kitabinda (17,55 - 34,30).

    Kutuphaneye degil kendi formulumuze guveniyoruz; merkez teriminin
    (p + z^2/2n) kaydirmasi kolayca unutulur ve unutuldugunda aralik
    Wald'a doner - yani tam kacinmak istedigimiz seye.
    """
    lo, hi = wilson(25, 100)

    assert lo == pytest.approx(17.55, abs=0.02)
    assert hi == pytest.approx(34.30, abs=0.02)


@pytest.mark.parametrize("k,n", [(0, 100), (100, 100), (1, 10_000), (3, 500)])
def test_wilson_never_leaves_the_unit_interval(k: int, n: int):
    """Wilson'un secilme SEBEBI bu. `received` %0,6 seviyesinde ve orada
    Wald araligi negatife dusuyor - negatif bir yaygınlık raporlanamaz.
    """
    lo, hi = wilson(k, n)

    assert 0.0 <= lo <= hi <= 100.0


def test_wilson_is_undefined_without_a_sample():
    assert wilson(0, 0) == (None, None)


# --------------------------------------------------------- cerceve ayrimi
def test_prevalence_reads_only_the_main_frame(annotated: Config):
    """`boost` havuzlanirsa kendi sectigimiz seyi olcmus oluruz.

    Fixture'in boost havuzlari bos kaldigi icin boost satiri elle ekleniyor:
    sinanan sey ornekleme degil, modulun cerceveyi ayirip ayirmadigi.
    """
    path = annotation_path(annotated, "pilot")
    base = pl.read_parquet(path)
    once = category_stats(annotated, "pilot")

    # Hepsi hediye olan bir boost bloku ekle. Cerceve ayrilmiyorsa oran zipliyor.
    pl.concat([
        base,
        base.with_columns(
            pl.lit("boost").alias("sample_frame"),
            pl.lit("gift_given").alias("purchase_type"),
            (pl.col("row_id") + 10_000).alias("row_id"),
        ),
    ]).write_parquet(path)

    sonra = category_stats(annotated, "pilot")

    assert sonra["n_main"] == once["n_main"]
    assert sonra["definitions"]["narrow"] == once["definitions"]["narrow"]


def test_main_frame_is_required(annotated: Config):
    """`main` bos gelirse sessizce 0 raporlanmaz - gurultulu hata."""
    path = annotation_path(annotated, "pilot")
    pl.read_parquet(path).with_columns(
        pl.lit("boost").alias("sample_frame")
    ).write_parquet(path)

    with pytest.raises(RuntimeError, match="main"):
        load_main(annotated, "pilot")


# ------------------------------------------------------------- iki tanim
def test_the_broad_definition_adds_exactly_household_and_received(annotated: Config):
    """genis - dar = household + received. Iki tanim ayni satiri iki kez saymamali.

    `received` bilerek erken siraya konuyor: fixture'in `main` cercevesi kucuk
    ve etiket listesinin sonu hic uygulanmayabilir - o durumda test sinadigi
    seyi sinamadan gecerdi (onceki hali tam olarak boyleydi).
    """
    _relabel(annotated, ["gift_given", "received", "household", "self", "unclear"])

    d = category_stats(annotated, "pilot")

    assert d["counts"].get("received", 0) >= 1, "test kurgusu received'i uygulayamadi"
    dar = d["definitions"]["narrow"]["k"]
    genis = d["definitions"]["broad"]["k"]
    assert genis - dar == d["counts"].get("household", 0) + d["counts"].get("received", 0)
    assert genis >= dar


def test_neither_definition_is_silently_chosen():
    """Kod tanimlardan birini secmemeli - ikisi de raporlanmali.

    Sonuca bakip buyuk olani secmek, esigi sonuc gorulduktan sonra
    dusurmekle ayni hata (CLAUDE.md 13: karar KAVRAMSAL ve ekipte).
    """
    assert set(DEFINITIONS) == {"narrow", "broad"}
    assert DEFINITIONS["narrow"] == ("gift_given",)
    # `received` 2026-09-14'te girdi: C1b "alici urunu kendisi secmedi" tanimi.
    # Yayginlik ile deneyin C1b kosulu AYNI kumeyi kullanmali.
    assert DEFINITIONS["broad"] == ("gift_given", "household", "received")


# ------------------------------------------------------------ kuru kosu
def test_a_stub_run_cannot_produce_a_prevalence_table(cfg: Config):
    """Taklit etiket yaygınlık tablosuna GIREMEZ (gate1 ile ayni desen)."""
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    annotate(cfg, "pilot", backend_name="stub")   # backend duzeltilmiyor

    with pytest.raises(RuntimeError, match="[Kk]uru kosu"):
        load_main(cfg, "pilot")


# ------------------------------------------- dogrulama iddiasi hesaplaniyor
def test_human_validation_is_read_from_disk_not_asserted(annotated: Config):
    """Insan dogrulamasinin durumu DISKTEN okunmali.

    Elle yazilmis bir alan sessizce yalan soyler; dogrulama raporu yoksa
    cevap hayir olmak zorunda (denetim bulgusu 7).
    """
    import json

    rapor = evaluate(annotated, ["pilot"])
    assert rapor["human_validation"]["report_exists"] is False
    assert any("HENUZ YOK" in c for c in rapor["caveats"])

    # Tek etiketleyicili bir rapor ortaya cikinca durum degismeli - ama
    # "dogrulandi" DEMEMELI: guvenilirlik olculmedi.
    p = validation_report_path(annotated, int(annotated.get("validation.n")))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"n_annotators": 1},
        "criteria": {"1_annotator_agreement": {"passed": None}},
        "verdict": "INCOMPLETE",
    }), encoding="utf-8")

    durum = evaluate(annotated, ["pilot"])
    assert durum["human_validation"] == {
        "report_exists": True, "n_annotators": 1,
        "reliability_measured": False, "verdict": "INCOMPLETE",
    }
    assert any("TEK etiketleyici" in c for c in durum["caveats"])


def test_the_report_ranks_categories_by_the_narrow_rate(annotated: Config):
    """RQ1 kategorileri KARSILASTIRIYOR; sira raporda hesaplanmis olmali."""
    rapor = evaluate(annotated, ["pilot"])

    oranlar = [
        rapor["categories"][s]["definitions"]["narrow"]["rate_pct"]
        for s in rapor["ordering_narrow"]
    ]
    assert oranlar == sorted(oranlar, reverse=True)
    assert prevalence_path(annotated).exists()


def test_no_pooled_rate_is_reported(annotated: Config):
    """Kategoriler ESIT tahsisle cekildi, korpus paylariyla degil.

    Havuzlanmis tek bir oran yazmak, hicbir korpusa karsilik gelmeyen bir
    sayi uretir. Rapor kategori kirilimi disinda oran tasimamali.
    """
    rapor = evaluate(annotated, ["pilot"])

    assert "pooled" not in rapor
    assert "overall" not in rapor
    assert set(rapor["categories"]) == {annotated.category_slug("pilot")}


# ------------------------------------------- insan kalibrasyonu (2026-09-14)
def test_calibrated_rate_matches_hand_arithmetic():
    """`sum_k pay(k) * PPV_k`: 0,2*0,6 + 0,8*0,1 = 0,20."""
    from gift_contamination.analysis.prevalence import calibrated_rate

    assert calibrated_rate({"gift_given": 0.2, "self": 0.8},
                           {"gift_given": 0.6, "self": 0.1}) == pytest.approx(0.20)


def test_an_unmeasured_class_with_real_mass_makes_the_rate_undefined():
    """PPV'si olmayan ama payi olan sinifi SIFIR saymak orani asagi ceker."""
    import math

    from gift_contamination.analysis.prevalence import calibrated_rate

    out = calibrated_rate({"gift_given": 0.2, "unclear": 0.1, "self": 0.7},
                          {"gift_given": 0.6, "unclear": float("nan"), "self": 0.1})
    assert math.isnan(out)
    # Payi SIFIR olan sinifin PPV'si tanimsiz olabilir - sonuc etkilenmez
    assert calibrated_rate({"gift_given": 1.0, "unclear": 0.0},
                           {"gift_given": 0.5, "unclear": float("nan")}) == 0.5


def test_calibration_end_to_end_matches_a_hand_computed_case(cfg: Config):
    """Kontrollu populasyon + kontrollu insan etiketi -> elle hesaplanan oran.

    main: 20 gift_given, 80 self  ->  paylar 0,2 / 0,8
    dogrulama (main): LLM=gift 10 satir -> insan 6 gift + 4 household
                      LLM=self 10 satir -> insan 9 self + 1 gift
    dar   = 0,2*0,6 + 0,8*0,1 = %20,0
    genis = 0,2*1,0 + 0,8*0,1 = %28,0   (household geniste)
    """
    from gift_contamination.analysis.prevalence import evaluate as prev_eval
    from gift_contamination.data.labelsheet import (
        LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path,
    )
    from gift_contamination.data.sampling import validation_path

    cfg._data["validation"].update(
        {"annotators": ["A"], "reliability": "none", "n": 20, "bootstrap_n": 300}
    )
    main = ["gift_given"] * 20 + ["self"] * 80
    dest = annotation_path(cfg, "pilot")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "sample_frame": ["main"] * 100,
        "purchase_type": main,
        "backend": ["vllm"] * 100,
        "model": ["m"] * 100,
        "prompt_version": ["v3"] * 100,
    }).write_parquet(dest)

    llm = ["gift_given"] * 10 + ["self"] * 10
    insan = ["gift_given"] * 6 + ["household"] * 4 + ["self"] * 9 + ["gift_given"]
    csv = labeled_path(validation_path(cfg, 20))
    csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "val_id": list(range(1, 21)),
        "category": [cfg.category_slug("pilot")] * 20,
        "sample_frame": ["main"] * 20,
        "purchase_type": llm,
        "val_stratum": llm,
        "label_A": insan,
        "notes_A": [None] * 20,
        LABEL_SOURCE_COLUMN: [LABEL_SOURCE] * 20,
    }).write_csv(csv)

    kal = prev_eval(cfg, ["pilot"])["calibration"]
    slug = cfg.category_slug("pilot")

    assert kal["by_category"][slug]["narrow"]["calibrated_pct"] == pytest.approx(20.0)
    assert kal["by_category"][slug]["broad"]["calibrated_pct"] == pytest.approx(28.0)
    lo, hi = kal["by_category"][slug]["narrow"]["ci95_pct"]
    assert lo <= 20.0 <= hi
    assert kal["ppv"]["narrow"]["gift_given"] == {"value": 0.6, "n": 10, "source": "main"}
    assert kal["reference"] == {"annotators": ["A"], "reliability_measured": False}


def test_calibration_is_skipped_not_invented_without_labels(annotated: Config):
    """Dogrulama etiketi yoksa kalibrasyon UYDURULMAZ; blok acikca atlanir."""
    kal = evaluate(annotated, ["pilot"])["calibration"]

    assert kal == {"skipped": True, "reason": "dogrulama etiketleri henuz yok"}


def test_the_post_hoc_sensitivity_is_separate_and_labelled(cfg: Config):
    """Sonuc goruldukten sonra eklenen analiz BIRINCIL sayinin yerine gecmemeli.

    Ayri blokta durmali, `post_hoc: true` tasimali ve birincil kalibre oran
    degismemeli.
    """
    from gift_contamination.analysis.prevalence import evaluate as prev_eval
    from gift_contamination.data.labelsheet import (
        LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path,
    )
    from gift_contamination.data.sampling import validation_path

    cfg._data["validation"].update(
        {"annotators": ["A"], "reliability": "none", "n": 20, "bootstrap_n": 50}
    )
    dest = annotation_path(cfg, "pilot")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "sample_frame": ["main"] * 100,
        "purchase_type": ["gift_given"] * 20 + ["self"] * 80,
        "backend": ["vllm"] * 100, "model": ["m"] * 100, "prompt_version": ["v3"] * 100,
    }).write_parquet(dest)
    llm = ["gift_given"] * 10 + ["self"] * 10
    csv = labeled_path(validation_path(cfg, 20))
    csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "val_id": list(range(1, 21)),
        "category": [cfg.category_slug("pilot")] * 20,
        "sample_frame": ["main"] * 20,
        "purchase_type": llm, "val_stratum": llm,
        "label_A": ["gift_given"] * 6 + ["household"] * 4 + ["self"] * 9 + ["gift_given"],
        "notes_A": [None] * 20,
        LABEL_SOURCE_COLUMN: [LABEL_SOURCE] * 20,
    }).write_csv(csv)

    kal = prev_eval(cfg, ["pilot"])["calibration"]
    slug = cfg.category_slug("pilot")

    blok = kal["post_hoc_own_category_ppv_pct"]
    assert blok["post_hoc"] is True
    assert slug in blok["by_category"]
    # Tek kategori oldugu icin kendi PPV'si = havuzlanmis PPV; birincil deger aynen
    assert blok["by_category"][slug]["narrow"] == pytest.approx(20.0)
    assert kal["by_category"][slug]["narrow"]["calibrated_pct"] == pytest.approx(20.0)


def test_a_category_interval_does_not_depend_on_processing_order(cfg: Config):
    """Bir kategorinin GA'si, yanindaki kategorilerin sirasina bagli olmamali.

    Tek bir rng butun kategorilere sirayla dagitilinca `--category all` ile tek
    tek cagri farkli GA uretiyordu (2026-09-14'te olculdu). Seed (kategori, tanim)
    basina turetiliyor.
    """
    from gift_contamination.analysis.prevalence import calibration
    from gift_contamination.data.labelsheet import (
        LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path,
    )
    from gift_contamination.data.sampling import validation_path

    cfg._data["validation"].update(
        {"annotators": ["A"], "reliability": "none", "n": 20, "bootstrap_n": 80}
    )
    llm = ["gift_given"] * 10 + ["self"] * 10
    csv = labeled_path(validation_path(cfg, 20))
    csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "val_id": list(range(1, 21)), "category": ["X"] * 10 + ["Y"] * 10,
        "sample_frame": ["main"] * 20, "purchase_type": llm, "val_stratum": llm,
        "label_A": ["gift_given"] * 6 + ["household"] * 4 + ["self"] * 9 + ["gift_given"],
        "notes_A": [None] * 20, LABEL_SOURCE_COLUMN: [LABEL_SOURCE] * 20,
    }).write_csv(csv)

    def kat(gift: int) -> dict:
        return {"n_main": 100, "counts": {"gift_given": gift, "self": 100 - gift},
                "definitions": {"narrow": {"rate_pct": gift}, "broad": {"rate_pct": gift}}}

    ab = calibration(cfg, {"X": kat(20), "Y": kat(5)})["by_category"]
    ba = calibration(cfg, {"Y": kat(5), "X": kat(20)})["by_category"]

    assert ab["X"]["narrow"]["ci95_pct"] == ba["X"]["narrow"]["ci95_pct"]
    assert ab["Y"]["broad"]["ci95_pct"] == ba["Y"]["broad"]["ci95_pct"]
