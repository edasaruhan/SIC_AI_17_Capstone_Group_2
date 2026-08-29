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
def test_the_broad_definition_adds_exactly_the_household_rows(annotated: Config):
    """genis - dar = household. Iki tanim ayni satirlari iki kez saymamali."""
    _relabel(annotated, ["gift_given", "household", "household", "self",
                         "self", "unclear", "received", "self"])

    d = category_stats(annotated, "pilot")

    dar = d["definitions"]["narrow"]["k"]
    genis = d["definitions"]["broad"]["k"]
    assert genis - dar == d["counts"].get("household", 0)
    assert genis >= dar


def test_neither_definition_is_silently_chosen():
    """Kod tanimlardan birini secmemeli - ikisi de raporlanmali.

    Sonuca bakip buyuk olani secmek, esigi sonuc gorulduktan sonra
    dusurmekle ayni hata (CLAUDE.md 13: karar KAVRAMSAL ve ekipte).
    """
    assert set(DEFINITIONS) == {"narrow", "broad"}
    assert DEFINITIONS["narrow"] == ("gift_given",)
    assert DEFINITIONS["broad"] == ("gift_given", "household")


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
    """`human_validated` DISKTEN okunmali.

    Elle `true` yazilmis bir alan sessizce yalan soyler; dogrulama raporu
    yoksa cevap hayir olmak zorunda (denetim bulgusu 7).
    """
    rapor = evaluate(annotated, ["pilot"])
    assert rapor["human_validated"] is False
    assert any("HENUZ YOK" in c for c in rapor["caveats"])

    # Dogrulama raporu ortaya cikinca iddia da degismeli.
    p = validation_report_path(annotated, int(annotated.get("validation.n")))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")

    assert evaluate(annotated, ["pilot"])["human_validated"] is True


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
