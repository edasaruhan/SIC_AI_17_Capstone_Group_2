"""Hafta 4 dogrulama testleri.

Buradaki risk gate1'dekiyle ayni cinsten ama daha pahali: bu olcum uc kisinin
saatlerini harciyor ve projenin TEK gercek referansi. Yanlis hesaplanan bir
kappa, semanin mugalak oldugunu gizler; yanlis hesaplanan bir F1 ise detektoru
oldugundan iyi gosterir ve o rakam rapora "insan dogrulamasi" diye girer.

`test_fleiss_matches_the_hand_worked_example` kutuphaneye kor guveni kesiyor:
statsmodels'in dondurdugu sayi elle hesaplanabilir iki vakada dogrulaniyor.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from gift_contamination.analysis.validation import (
    TIE,
    compare,
    consensus,
    evaluate,
    fleiss,
    kappa_report,
    load_labels,
)
from gift_contamination.config import Config
from gift_contamination.data.labelsheet import (
    LABEL_SOURCE,
    LABEL_SOURCE_COLUMN,
    labeled_path,
)
from gift_contamination.data.sampling import validation_path


def _write_labeled(
    cfg: Config,
    labels: dict[str, list[str | None]],
    *,
    llm: list[str] | None = None,
    proxy: list[bool] | None = None,
) -> Path:
    """Doldurulmus dogrulama CSV'si kurar. `labels` = {"A": [...], "B": [...]}."""
    n = len(next(iter(labels.values())))
    llm = llm or [v if v else "unclear" for v in labels["A"]]
    proxy = proxy if proxy is not None else [v == "gift_given" for v in llm]
    data = {
        "val_id": list(range(1, n + 1)),
        "category": ["Test_Cat"] * n,
        "row_id": list(range(100, 100 + n)),
        "title": ["t"] * n,
        "text": ["review metni burada"] * n,
        "val_stratum": llm,
        "sample_frame": ["main"] * n,
        "kw_gift_proxy": proxy,
        "purchase_type": llm,
        "confidence": ["high"] * n,
        "recipient": ["child"] * n,
        "occasion": ["none"] * n,
        "evidence_span": [""] * n,
        "prompt_version": ["v3"] * n,
        "model": ["Qwen/Qwen3-4B-Instruct-2507"] * n,
        **{f"label_{k}": v for k, v in labels.items()},
        **{f"notes_{k}": [""] * n for k in labels},
        # `labelsheet --ingest` yolundan gelmis gibi damgala; damgasiz dosyanin
        # reddedildigini `test_unstamped_labels_cannot_enter_the_measurement`
        # ayrica sinaniyor.
        LABEL_SOURCE_COLUMN: [LABEL_SOURCE] * n,
    }
    dest = labeled_path(validation_path(cfg, n))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_csv(dest)
    return dest


# ---------------------------------------------------------------------- kappa
def test_fleiss_matches_the_hand_worked_example():
    """Kutuphaneye kor guven yok: iki vaka elle hesaplanabilir.

    Tam uyum -> kappa 1. Sistematik ters uyum (her satirda 2'ye 1 bolunme,
    yonu donusumlu) -> P_bar = 1/3, P_e = 1/2, kappa = -1/3.
    """
    tam = [("A", "A", "A"), ("A", "A", "A"), ("B", "B", "B"), ("B", "B", "B")]
    assert fleiss(tam, ["A", "B"]) == 1.0

    sanstan_kotu = [("A", "A", "B"), ("A", "A", "B"), ("B", "B", "A"), ("B", "B", "A")]
    assert fleiss(sanstan_kotu, ["A", "B"]) == pytest.approx(-1 / 3)


def test_single_class_yields_none_not_a_number():
    """Herkes her satira ayni etiketi verirse kappa 0/0'dir.

    statsmodels nan dondurur; nan'i sayiymis gibi rapora tasimak "kappa
    hesaplandi" izlenimi verir. None acikca "hesaplanamadi" demek.
    """
    assert fleiss([("A", "A", "A"), ("A", "A", "A")], ["A", "B"]) is None


def test_sparse_class_is_kept_out_of_the_gate_kappa():
    """`received` 500'luk sette ~3-8 satir olacak; oradaki kappa anlamsiz.

    Kural sonuc gorulmeden yazildi (DECISIONS 2026-08-29). Dislama, sinif
    sutununu silmek DEGIL o sinifin gectigi SATIRLARI dusurmek - aksi halde
    satir basina degerlendirici sayisi bozulur ve Fleiss varsayimi cokerdi.
    """
    rows = [("self", "self", "self")] * 10 + [("gift_given",) * 3] * 10
    rows += [("received", "received", "self")]  # tek seyrek satir
    df = pl.DataFrame({
        "label_A": [r[0] for r in rows],
        "label_B": [r[1] for r in rows],
        "label_C": [r[2] for r in rows],
    })

    rapor = kappa_report(df, ["label_A", "label_B", "label_C"], min_class_n=5)

    assert rapor["sparse_classes"] == ["received"]
    assert rapor["n_rows_dropped_as_sparse"] == 1
    assert rapor["n_rows_in_kappa"] == 20
    # Kisitlanmamis deger de raporlaniyor - secim seffaf kalsin
    assert rapor["kappa_unrestricted"] is not None
    assert rapor["kappa"] == 1.0  # kalan 20 satirda tam uyum


def test_incomplete_rows_never_enter_kappa():
    """Fleiss satir basina SABIT degerlendirici sayisi varsayar."""
    df = pl.DataFrame({
        "label_A": ["self", "self", "gift_given"],
        "label_B": ["self", None, "gift_given"],
        "label_C": ["self", "self", "gift_given"],
    })

    rapor = kappa_report(df, ["label_A", "label_B", "label_C"], min_class_n=1)

    assert rapor["n_rows"] == 2
    assert rapor["n_incomplete_rows"] == 1


# ------------------------------------------------------------------- uzlasi
def test_a_three_way_split_becomes_tie_not_a_class():
    """Ucu de farkli sey diyorsa dogru cevap YOKTUR.

    Sessizce `unclear`a itmek hem uzlasi oranini sisirir hem de o sinifin
    F1'ini bozar - detektor olmayan bir referansa karsi olculmus olur.
    """
    df = pl.DataFrame({
        "label_A": ["self", "gift_given"],
        "label_B": ["household", "gift_given"],
        "label_C": ["unclear", "self"],
    })

    out = consensus(df, ["label_A", "label_B", "label_C"]).to_list()

    assert out == [TIE, "gift_given"]


# ---------------------------------------------------------------- kiyaslama
def test_compare_separates_over_calling_from_missing():
    """Az cagirmak ile yanlis cagirmak ayni sayiya dusmemeli.

    `gate1.trial_agreement` ile AYNI bicim - iki rapor yan yana okunabilsin.
    """
    ref = pl.Series(["household", "household", "gift_given", "self"])
    pred = pl.Series(["gift_given", "household", "gift_given", "self"])

    out = compare(ref, pred)

    assert out["disagreements"] == {"household->gift_given": 1}
    assert out["per_class"]["household"]["precision"] == 1.0
    assert out["per_class"]["household"]["recall"] == 0.5
    assert out["per_class"]["gift_given"]["precision"] == 0.5
    assert out["per_class"]["gift_given"]["recall"] == 1.0
    assert out["accuracy"] == 0.75


# --------------------------------------------------------------------- kapi
def test_gate_passes_and_reports_both_measurements(cfg: Config):
    """Kapi kappa'ya bakar; F1 ve vekil OLCUMDUR, kapi degil."""
    n = 24
    perfect = ["gift_given", "self", "household", "unclear"] * (n // 4)
    _write_labeled(cfg, {"A": perfect, "B": perfect, "C": perfect})
    cfg._data["validation"]["n"] = n
    cfg._data["validation"]["min_class_n_for_kappa"] = 4

    rapor = evaluate(cfg, n)

    assert rapor["verdict"] == "PASS"
    assert rapor["criteria"]["1_annotator_agreement"]["passed"] is True
    # LLM olcumu var ama KAPI olcutlerinde degil
    assert "llm_vs_human" in rapor["measurements"]
    assert list(rapor["criteria"]) == ["1_annotator_agreement"]


def test_low_agreement_fails_the_gate(cfg: Config):
    """Kappa esigin altindaysa karar FAIL - sema mugalak demektir."""
    n = 24
    a = ["gift_given", "self", "household", "unclear"] * (n // 4)
    b = ["self", "household", "unclear", "gift_given"] * (n // 4)
    c = ["household", "unclear", "gift_given", "self"] * (n // 4)
    _write_labeled(cfg, {"A": a, "B": b, "C": c})
    cfg._data["validation"]["min_class_n_for_kappa"] = 4

    rapor = evaluate(cfg, n)

    assert rapor["verdict"] == "FAIL"
    assert rapor["criteria"]["1_annotator_agreement"]["kappa"] < 0.60


def test_ties_are_excluded_from_the_detector_measurement(cfg: Config):
    """Uzlasi olmayan satir referans olamaz - ama sayisi raporlanir."""
    a = ["gift_given"] * 8 + ["self"] * 8 + ["household"] * 4 + ["self"] * 4
    b = list(a)
    c = list(a)
    c[-1] = "unclear"
    c[-2] = "gift_given"
    b[-1] = "household"  # A=self, B=household, C=unclear -> tie
    _write_labeled(cfg, {"A": a, "B": b, "C": c})
    cfg._data["validation"]["min_class_n_for_kappa"] = 4

    rapor = evaluate(cfg, 24)

    assert rapor["measurements"]["n_tie"] == 1
    assert rapor["measurements"]["n_usable"] == 23
    assert rapor["measurements"]["llm_vs_human"]["n_compared"] == 23


def test_proxy_is_scored_on_the_gift_axis_only(cfg: Config):
    """Vekil IKILI: ya `gift_given` der ya demez.

    Isaretlemedigi satirlari `self` saymak onu haksiz yere cezalandirirdi -
    household/unclear ayrimini yapamiyor olmasi tasarim geregi.
    """
    n = 8
    etiket = ["gift_given", "household", "self", "unclear"] * 2
    # Vekil yalnizca ilk iki gift satirini yakaliyor, birinde de yanilıyor
    proxy = [True, True, False, False, True, False, False, False]
    _write_labeled(cfg, {"A": etiket, "B": etiket, "C": etiket},
                   llm=etiket, proxy=proxy)
    cfg._data["validation"]["min_class_n_for_kappa"] = 2

    rapor = evaluate(cfg, n)
    vekil = rapor["measurements"]["proxy_vs_human_gift_only"]["per_class"]

    # 2 gift var; vekil ikisini de yakaladi ama 1 yanlis pozitif verdi
    assert vekil["gift_given"]["n_reference"] == 2
    assert vekil["gift_given"]["recall"] == 1.0
    assert vekil["gift_given"]["precision"] == pytest.approx(2 / 3, abs=1e-3)


def test_a_missing_annotator_column_fails_loudly(cfg: Config):
    """Iki sayfayla hesaplanan kappa yanlistir; sessizce devam edilmemeli."""
    etiket = ["self"] * 8
    _write_labeled(cfg, {"A": etiket, "B": etiket})

    with pytest.raises(ValueError, match="etiketleyici kolonu"):
        load_labels(cfg, 8)


def test_unstamped_labels_cannot_enter_the_measurement(cfg: Config):
    """Elle yazilmis ya da sentetik bir dosya gercek gorunen bir PASS uretir.

    2026-08-28'de taklit LLM ciktisi gercek gorunen bir Kapi 1 karari
    uretmisti; ayni tuzak burada da var ve burada uc kisinin saatleri
    soz konusu. Damga yalnizca `labelsheet --ingest` yolundan basiliyor.
    """
    etiket = ["self"] * 8
    dest = _write_labeled(cfg, {"A": etiket, "B": etiket, "C": etiket})
    pl.read_csv(dest).drop(LABEL_SOURCE_COLUMN).write_csv(dest)

    with pytest.raises(RuntimeError, match="damgasi tasiyor"):
        load_labels(cfg, 8)


def test_the_reported_kappa_is_rounded_but_the_gate_uses_the_exact_value():
    """Rapor okunabilir olmali; kapi yuvarlamadan ETKILENMEMELI.

    Yuvarlanmis bir deger esigi gecirebilir: 0,59996 -> 0,60. Rapordaki sayi
    kisa dursun diye yapilan bir islem kapinin kararini degistiremez, o yuzden
    iki deger ayri tutuluyor.

    Ayrica commit edilen JSON'da 17 basamakli float durmasin: `gate1`in
    karisiklik siralamasiyla ayni gerekce - diff'te gercek degisiklik gorunsun.
    """
    g, s = "gift_given", "self"
    satirlar = [(g, g, s), (g, s, s), (s, s, g), (g, g, g)]

    rapor = kappa_report(
        pl.DataFrame(satirlar, schema=["label_A", "label_B", "label_C"], orient="row"),
        ["label_A", "label_B", "label_C"],
        min_class_n=0,
    )

    ham = rapor["kappa_exact"]
    assert ham is not None
    assert rapor["kappa"] == pytest.approx(round(ham, 4))
    # Rapor degeri en fazla 4 basamak; ham deger kirpilmamis olmali
    assert len(str(rapor["kappa"]).split(".")[-1]) <= 4
    # Sinif bazli degerler de yuvarlanmis
    for v in rapor["kappa_per_class"].values():
        if v is not None:
            assert len(str(v).split(".")[-1]) <= 4
