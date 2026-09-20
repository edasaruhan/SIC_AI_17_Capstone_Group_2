"""`eda` ve `deep_eda` testleri.

Bu iki modul 1.400 satir ve neredeyse test edilmiyordu (denetim 2026-09-20) -
oysa Data Research teslimindeki **her** tablo ve F1-F16'nin tamami buradan
cikiyor. Yanlis bir sayi burada uretilirse teslim belgesine girer ve kimse
fark etmez.

Sinanan sey figurlerin GORUNUSU degil, sayilarin ANLAMI: huni asagi dogru
satir kazanmiyor mu, yuzdeler dogru paydaya bolunuyor mu, olculemeyen bir
deger 0 olarak mi yaziliyor, ve teslim klasorune kopyalama gercekten
calisiyor mu.

Uc gercek ariza bu dosya yazilirken ortaya cikti ve duzeltildi:
`available_roles` config'te tanimsiz rol gorunce butun EDA'yi dusuruyordu,
`table_cross_category` tek kategorili bir kosuda sifira boluyordu,
`table_seasonality_summary` eksik bir ayda KeyError atiyordu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gift_contamination.analysis import deep_eda, eda
from gift_contamination.analysis.keyword_scan import rates_path, scan_category
from gift_contamination.config import Config
from gift_contamination.data.preprocess import funnel_path


@pytest.fixture
def scanned(cfg: Config) -> Config:
    """Fixture korpusu on islenmis ve sozcuksel vekille taranmis."""
    scan_category(cfg, "pilot")
    return cfg


def _n(text: str) -> int:
    """Tabloda sayilar `1,234` bicimli yaziliyor."""
    return int(text.replace(",", ""))


# ------------------------------------------------------------------ T1 huni
def test_the_funnel_never_gains_rows_on_the_way_down(scanned: Config):
    """On isleme yalnizca ELER. Bir asama oncekinden fazla satir gosteriyorsa
    ya filtre ya sayac bozuk demektir - ve huni teslimdeki ilk tablo."""
    rows = eda.table_funnel(scanned, ["pilot"])

    sayilar = [_n(r["rows"]) for r in rows]
    assert sayilar == sorted(sayilar, reverse=True), rows


def test_the_funnel_shows_every_stage_the_json_recorded(scanned: Config):
    """Sessizce dusen bir asama, elenen satirlarin nerede gittigini gizler."""
    ham = json.loads(funnel_path(scanned, "pilot").read_text(encoding="utf-8"))
    beklenen = [k for k in ham if k not in {"meta", "timestamp"}]

    rows = eda.table_funnel(scanned, ["pilot"])

    assert [r["stage"] for r in rows] == beklenen


def test_retained_share_is_measured_against_the_raw_stage(scanned: Config):
    """Payda bir onceki asama OLSAYDI her satir ~%100 gorunur ve hunini
    okumanin anlami kalmazdi."""
    ham = json.loads(funnel_path(scanned, "pilot").read_text(encoding="utf-8"))
    base = ham["01_raw"]["rows"]
    rows = eda.table_funnel(scanned, ["pilot"])

    son = rows[-1]
    assert son["retained"] == f"{100 * _n(son['rows']) / base:.1f}%"
    assert rows[0]["retained"] == "100.0%"


# ------------------------------------------------------------- T2 korpus
def test_the_corpus_table_reads_k_core_from_the_config(scanned: Config):
    """Ayni hata bir kez yasandi: `eda.py` `analysis.expected_peaks`i config'ten
    okumayip icinde tasiyordu; anahtar duzeltildi, tablo degismedi
    (test_config_keys.py). Burada k-core icin ayni tuzak kapatiliyor."""
    k = int(scanned.get("preprocess.k_core"))
    once = eda.table_corpus(scanned, ["pilot"])[0]
    assert f"users ≥{k}" in once

    scanned._data["preprocess"]["k_core"] = k + 3
    sonra = eda.table_corpus(scanned, ["pilot"])[0]

    assert f"users ≥{k + 3}" in sonra, "k_core degisti, tablo degismedi"
    assert f"users ≥{k}" not in sonra


# --------------------------------------------------------- bicimlendirme
def test_md_table_keeps_one_row_per_record_and_tolerates_a_missing_key():
    out = eda.md_table([{"a": 1, "b": 2}, {"a": 3}], ["a", "b"])

    satir = out.splitlines()
    assert len(satir) == 4            # baslik + ayirici + iki kayit
    assert satir[-1] == "| 3 |  |"    # eksik alan bos, satir kaymiyor


def test_an_unmeasured_value_is_marked_not_printed_as_zero():
    """`—` ile `0.00%` ayni sey degil: biri "olcemedik", digeri "olctuk, sifir".
    Ikisini karistirmak teslimde yanlis bir iddia uretir."""
    assert eda.pct(None) == "—"
    assert eda.pct(0.0) == "0.00%"
    assert eda.pct(0.1234, 1) == "12.3%"


# ------------------------------------------------------- rol secimi
def test_available_roles_refuses_to_produce_an_empty_eda(cfg: Config):
    """Tarama yoksa bos bir EDA uretmektense durmali - bos tablolar
    "hediye yok" gibi okunur."""
    with pytest.raises(FileNotFoundError, match="keyword_scan"):
        eda.available_roles(cfg)


def test_available_roles_skips_a_role_the_config_does_not_define(scanned: Config):
    """`ROLE_ORDER` ANLATI sirasi; config daha az kategori tanimlayabilir.
    Onceden `keyword_path` ConfigError atip butun EDA'yi dusuruyordu."""
    assert eda.available_roles(scanned) == ["pilot"]


# ------------------------------------------------------------ uctan uca
def test_eda_writes_every_table_and_figure_it_promises(scanned: Config):
    eda.run(scanned)

    tablolar = json.loads(
        (scanned.path("results") / "eda_tables.json").read_text(encoding="utf-8")
    )
    assert set(tablolar) == {
        "T1_preprocessing_funnel", "T2_corpus_profile", "T3_keyword_proxy_rates",
    }
    assert (scanned.path("results") / "eda_tables.md").exists()

    for ad, _ in eda.FIGURES:
        f = scanned.path("figures", f"{ad}.png")
        assert f.exists() and f.stat().st_size > 0, ad


def test_t4_is_skipped_when_the_manual_precision_file_is_missing(scanned: Config):
    """T4 elle etiketlenmis 100 satirdan geliyor ve git'e girmiyor. Yoksa
    atlanmali - uydurulmamali."""
    eda.run(scanned)
    tablolar = json.loads(
        (scanned.path("results") / "eda_tables.json").read_text(encoding="utf-8")
    )
    assert "T4_manual_precision" not in tablolar

    (scanned.path("results") / "keyword_precision.json").write_text(
        json.dumps({"proxy_precision": 0.5833}), encoding="utf-8"
    )
    eda.run(scanned)
    tablolar = json.loads(
        (scanned.path("results") / "eda_tables.json").read_text(encoding="utf-8")
    )
    assert tablolar["T4_manual_precision"] == {"proxy_precision": 0.5833}


def test_publishing_copies_the_figures_to_the_submission_folder(scanned: Config,
                                                                tmp_path: Path):
    """`data-research/figures` bu mekanizmayla guncel kaliyor; kopyalar
    kanonik `reports/figures` ile BIREBIR ayni olmali."""
    hedef = tmp_path / "teslim"
    scanned._data["eda"]["publish_figures_to"] = str(hedef)

    eda.run(scanned)

    for ad, _ in eda.FIGURES:
        kopya = hedef / f"{ad}.png"
        assert kopya.exists()
        assert kopya.read_bytes() == scanned.path("figures", f"{ad}.png").read_bytes()


def test_nothing_is_published_when_no_target_is_configured(scanned: Config):
    """Varsayilan "kopyalama": yanlis yapilandirilmis bir hedef, depodaki
    teslim klasorunu ezebilir."""
    assert "publish_figures_to" not in scanned._data["eda"]

    eda.publish(scanned, "png", eda.FIGURES)  # patlamamali, bir sey yazmamali


# ------------------------------------------------------------- deep_eda
def test_the_seasonality_summary_matches_a_hand_calculation(scanned: Config):
    """Teslimin §4.5 tablosu bu iki turetilmis sayiyi aliniyor."""
    aylar = [{"month": m, "proxy_rate": 0.02} for m in range(1, 13)]
    aylar[11]["proxy_rate"] = 0.06        # Aralik
    rates_path(scanned, "pilot").write_text(
        json.dumps({"by_month": aylar}), encoding="utf-8"
    )

    satir = deep_eda.table_seasonality_summary(scanned, ["pilot"])[0]

    assert satir["Jun–Sep trough (mean)"] == "2.00%"
    assert satir["Dec"] == "6.00%"
    assert satir["Dec ÷ summer"] == "3.00×"


def test_a_month_with_no_data_reads_as_unmeasured_not_as_zero(scanned: Config):
    """Hic review'i olmayan bir ay ile hediye orani %0 olan bir ay ayni sey
    degil. Onceden bu durum KeyError ile butun tabloyu dusuruyordu."""
    aylar = [{"month": m, "proxy_rate": 0.02} for m in (1, 2, 6, 7, 11, 12)]
    rates_path(scanned, "pilot").write_text(
        json.dumps({"by_month": aylar}), encoding="utf-8"
    )

    satir = deep_eda.table_seasonality_summary(scanned, ["pilot"])[0]

    assert satir["Jun–Sep trough (mean)"] == "2.00%"   # 6 ve 7 var, 8 ve 9 yok
    assert satir["Dec ÷ summer"] == "1.00×"


def test_composition_shares_add_up_to_the_rows_shown(scanned: Config):
    """Paylar gosterilen satirlarin toplamina gore; baska bir paydaya
    bolunurlerse tablo %100'u tutturmaz ve okuyan kisi eksik satir arar."""
    rows = deep_eda.table_recipients(scanned, ["pilot"])
    if not rows:
        pytest.skip("fixture korpusunda alici etiketi yok")

    toplam = sum(float(r["share"].rstrip("%")) for r in rows)
    assert toplam == pytest.approx(100.0, abs=0.5)


def test_cross_category_reports_unmeasurable_instead_of_dividing_by_zero(
    scanned: Config,
):
    """Tek kategorili bir kosuda "birden fazla kategoride gorulen musteri"
    kumesi bos: eskiden ZeroDivisionError, simdi "olculmedi"."""
    rows = deep_eda.table_cross_category(scanned, ["pilot"])

    assert rows[0]["value"] == "0"
    assert "—" in rows[1]["of"]


def test_deep_eda_writes_every_table_and_figure_it_promises(scanned: Config):
    deep_eda.run(scanned)

    tablolar = json.loads(
        (scanned.path("results") / "deep_eda_tables.json").read_text(encoding="utf-8")
    )
    assert len(tablolar) == 11
    assert (scanned.path("results") / "deep_eda_tables.md").exists()

    for ad, _ in deep_eda.FIGURES:
        f = scanned.path("figures", f"{ad}.png")
        assert f.exists() and f.stat().st_size > 0, ad


def test_neither_command_advertises_a_force_flag_it_ignores():
    """Ikisi de her cagrida yeniden hesapliyor; `--force` kabul edilip
    sessizce yok sayiliyordu (denetim 2026-09-20)."""
    for modul in (eda, deep_eda):
        with pytest.raises(SystemExit):
            modul.main(["--force"])
