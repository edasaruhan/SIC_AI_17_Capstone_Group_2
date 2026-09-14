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

import numpy as np
import polars as pl
import pytest

from gift_contamination.analysis.validation import (
    TIE,
    annotator_mode,
    compare,
    consensus,
    evaluate,
    fleiss,
    ipw_weights,
    kappa_report,
    kendi_cocugu_table,
    load_labels,
    scored_measurement,
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


# ------------------------------------------ tek etiketleyici (2026-09-14)
def _single(cfg: Config) -> Config:
    """Production durumu: yalnizca A teslim etti, guvenilirlik olculmedi."""
    cfg._data["validation"]["annotators"] = ["A"]
    cfg._data["validation"]["reliability"] = "none"
    return cfg


def test_a_single_annotator_can_never_pass_the_gate(cfg: Config):
    """Tek etiketleyiciyle uyum OLCULEMEZ - karar INCOMPLETE, asla PASS.

    A modelle birebir ayni etiketlese bile: modelin dogrulugu yuksek diye
    etiketin guvenilir oldugu sonucu cikmaz. PASS yazmak olculmemis bir seyi
    gecmis gibi gosterirdi.
    """
    n = 24
    etiket = ["gift_given", "self", "household", "unclear"] * (n // 4)
    _write_labeled(cfg, {"A": etiket}, llm=etiket)
    _single(cfg)

    rapor = evaluate(cfg, n)

    kriter = rapor["criteria"]["1_annotator_agreement"]
    assert rapor["verdict"] == "INCOMPLETE"
    assert kriter["passed"] is None
    assert kriter["skipped"] is True
    assert "tek etiketleyici" in kriter["reason"]
    # Olcumler DOLU - kapi INCOMPLETE diye olcum atlanmiyor
    assert rapor["measurements"]["llm_vs_human"]["accuracy"] == 1.0
    assert rapor["measurements"]["llm_vs_human_sample_ci"]["accuracy"]["value"] == 1.0
    assert rapor["measurements"]["n_tie"] == 0
    assert rapor["limitations"], "sinirliliklar raporun kendisinde durmali"


def test_reliability_mode_must_agree_with_the_annotator_count(cfg: Config):
    cfg._data["validation"]["annotators"] = ["A"]
    cfg._data["validation"]["reliability"] = "fleiss"
    with pytest.raises(ValueError, match="hesaplanamaz"):
        annotator_mode(cfg)

    cfg._data["validation"]["annotators"] = ["A", "B", "C"]
    cfg._data["validation"]["reliability"] = "none"
    with pytest.raises(ValueError, match="atilmaz"):
        annotator_mode(cfg)

    cfg._data["validation"]["reliability"] = "intra"
    with pytest.raises(NotImplementedError):
        annotator_mode(cfg)


def test_an_unconfigured_sheet_cannot_sneak_into_the_measurement(cfg: Config):
    """Config yalnizca A diyorsa dosyadaki B kolonu olcume GIREMEZ."""
    etiket = ["self"] * 8
    _write_labeled(cfg, {"A": etiket, "B": etiket})
    _single(cfg)

    with pytest.raises(ValueError, match="config"):
        load_labels(cfg, 8)


def test_ipw_weights_match_a_hand_computed_example():
    """Onceden kaydedilmis kural, elle hesaplanmis ornekte.

    Hucreler (min_cell_n=5):
      (X, gift) n=5 yeterli   N=100 -> 20 + tasinan
      (Y, gift) n=2 SEYREK    N=40  -> kutlesi gift'in 7 satirina: 40/7
      (X, self) n=6 yeterli   N=300 -> 50 + tasinan
      (Y, self) n=0 SEYREK    N=60  -> kutlesi self'in 6 satirina: 60/6 = 10
      (Z, unclear) hic satir yok N=50 -> KAPSANMAYAN pay 50/550
    """
    cats = ["X"] * 5 + ["Y"] * 2 + ["X"] * 6
    llm = ["gift_given"] * 7 + ["self"] * 6
    pop = {("X", "gift_given"): 100, ("Y", "gift_given"): 40,
           ("X", "self"): 300, ("Y", "self"): 60, ("Z", "unclear"): 50}

    w, meta = ipw_weights(cats, llm, pop, min_cell_n=5)

    assert w[:5] == pytest.approx([20 + 40 / 7] * 5)
    assert w[5:7] == pytest.approx([40 / 7] * 2)
    assert w[7:] == pytest.approx([50 + 10] * 6)
    # Her sinifin toplam agirligi populasyondaki satir sayisina esit
    assert w[:7].sum() == pytest.approx(140)
    assert w[7:].sum() == pytest.approx(360)
    assert meta["uncovered_share"] == pytest.approx(50 / 550, abs=1e-6)
    assert meta["collapsed_cells"] == ["Y/gift_given (n=2)", "Y/self (n=0)"]


def test_uniform_weights_do_not_change_the_point_estimates():
    """Agirliklarin olcegi degil ORANI onemli: w=1 ile w=7 ayni sonucu vermeli."""
    ref = ["gift_given", "self", "household", "self", "gift_given", "unclear"] * 3
    pred = ["gift_given", "self", "gift_given", "self", "self", "unclear"] * 3
    keys = [("X", p) for p in pred]

    bir = scored_measurement(ref, pred, np.ones(18), keys, n_boot=50, seed=1, min_class_n=2)
    yedi = scored_measurement(ref, pred, np.full(18, 7.0), keys, n_boot=50, seed=1, min_class_n=2)

    assert bir["accuracy"]["value"] == yedi["accuracy"]["value"]
    assert bir["macro_f1"]["value"] == yedi["macro_f1"]["value"]
    assert bir["per_class"]["household"]["recall"]["value"] == 0.0


def test_the_bootstrap_interval_contains_the_point_estimate():
    ref = (["gift_given"] * 12 + ["self"] * 12 + ["household"] * 12)
    pred = (["gift_given"] * 9 + ["self"] * 3 + ["self"] * 11 + ["household"]
            + ["household"] * 8 + ["gift_given"] * 4)
    keys = [("X", p) for p in pred]

    out = scored_measurement(ref, pred, np.ones(36), keys, n_boot=400, seed=42, min_class_n=5)

    lo, hi = out["accuracy"]["ci95"]
    assert lo <= out["accuracy"]["value"] <= hi
    assert lo < hi, "yeniden ornekleme gercekten degisken olmali"


def test_a_sparse_class_is_flagged_not_dropped():
    """n < min_class_n olan sinif HESAPTAN CIKMAZ, isaretlenir (onceden kayit)."""
    ref = ["self"] * 30 + ["received"] * 3
    pred = ["self"] * 30 + ["received"] * 3
    keys = [("X", p) for p in pred]

    out = scored_measurement(ref, pred, np.ones(33), keys, n_boot=20, seed=1, min_class_n=20)

    assert out["per_class"]["received"]["sparse"] is True
    assert out["per_class"]["received"]["f1"]["value"] == 1.0
    assert out["per_class"]["self"]["sparse"] is False


def test_the_weighted_measurement_reads_the_main_frame_population(cfg: Config):
    """Populasyon sayilari `main` cercevesinden gelir; boost satirlari sayilmaz."""
    from gift_contamination.detection.llm_annotate import annotation_path

    n = 24
    etiket = ["gift_given", "self", "household", "unclear"] * (n // 4)
    _write_labeled(cfg, {"A": etiket}, llm=etiket)
    _single(cfg)

    pop = annotation_path(cfg, "pilot")
    pop.parent.mkdir(parents=True, exist_ok=True)
    main = ["gift_given"] * 10 + ["self"] * 70 + ["household"] * 15 + ["unclear"] * 5
    pl.DataFrame({
        "sample_frame": ["main"] * 100 + ["boost"] * 50,
        "purchase_type": main + ["gift_given"] * 50,
    }).write_parquet(pop)

    agirlikli = evaluate(cfg, n)["measurements"]["llm_vs_human_main_weighted"]

    assert "skipped" not in agirlikli
    assert agirlikli["weights"]["population_rows"] == 100   # boost'un 50'si YOK
    assert agirlikli["per_class"]["self"]["weighted_n_reference"] == 70
    assert agirlikli["accuracy"]["value"] == 1.0


def test_kendi_cocugu_rows_are_cross_tabulated():
    df = pl.DataFrame({
        "label_A": ["household", "household", "gift_given", "self"],
        "purchase_type": ["gift_given", "household", "gift_given", "self"],
        "notes_A": ["KENDI_COCUGU", "KENDI_COCUGU", "KENDI_COCUGU", None],
    })

    out = kendi_cocugu_table(df, "A")

    assert out["n_flagged"] == 3
    assert out["human_label"] == {"gift_given": 1, "household": 2}
    assert out["human_to_llm"]["household->gift_given"] == 1


# ------------------------------------------------- ikili eksenler (C1 / C1b)
def test_binary_axis_matches_a_hand_computed_two_by_two():
    """C1 ekseni: 10 satir, elle sayilan 2x2.

    referans pozitif (gift_given): satir 0-3        -> 4
    tahmin pozitif:                satir 0-2 + 5,6  -> 5
    TP = 3 (0,1,2)  FP = 2 (5,6)  FN = 1 (3)
    K = 3/5 = 0,6   D = 3/4 = 0,75   F1 = 2*0,6*0,75/1,35 = 0,6667
    """
    from gift_contamination.analysis.validation import binary_axis_measurement

    ref = ["gift_given"] * 4 + ["household", "self", "self", "unclear", "self", "self"]
    pred = ["gift_given"] * 3 + ["self", "self", "gift_given", "gift_given",
                                 "unclear", "self", "self"]
    keys = [("X", p) for p in pred]

    out = binary_axis_measurement(ref, pred, np.ones(10), keys, ("gift_given",),
                                  n_boot=50, seed=1)

    assert out["n_reference_positive"] == 4
    assert out["precision"]["value"] == pytest.approx(0.6)
    assert out["recall"]["value"] == pytest.approx(0.75)
    assert out["f1"]["value"] == pytest.approx(2 * 0.6 * 0.75 / 1.35, abs=1e-4)


def test_household_gift_confusion_is_not_an_error_on_the_broad_axis():
    """C1b ekseninde `household <-> gift_given` karisikligi HATA DEGIL.

    Bes sinifli makro-F1 bu satirlari yanlis sayar; deneyin C1b karari icin
    ikisi de ayni kumede. Olculdu (2026-09-14): en buyuk uyusmazlik tam burada.
    """
    from gift_contamination.analysis.validation import binary_axis_measurement
    from gift_contamination.detection.schema import CONTAMINATION

    ref = ["household"] * 5 + ["gift_given"] * 5 + ["self"] * 5
    pred = ["gift_given"] * 5 + ["household"] * 5 + ["self"] * 5
    keys = [("X", p) for p in pred]

    dar = binary_axis_measurement(ref, pred, np.ones(15), keys, CONTAMINATION["narrow"],
                                  n_boot=20, seed=1)
    genis = binary_axis_measurement(ref, pred, np.ones(15), keys, CONTAMINATION["broad"],
                                    n_boot=20, seed=1)

    assert dar["f1"]["value"] == 0.0
    assert genis["f1"]["value"] == 1.0


def test_the_report_carries_both_axes_for_the_distillation_gate(cfg: Config):
    """Damitma kapisi ogretmenin C1 F1'ini RAPORDAN okuyacak - kodla uretilmis olmali."""
    n = 24
    etiket = ["gift_given", "self", "household", "unclear"] * (n // 4)
    _write_labeled(cfg, {"A": etiket}, llm=etiket)
    _single(cfg)

    eksen = evaluate(cfg, n)["measurements"]["contamination_axes"]

    assert set(eksen["sample"]) == {"C1", "C1b"}
    assert eksen["sample"]["C1"]["f1"]["value"] == 1.0
    assert eksen["sample"]["C1b"]["positive_labels"] == ["gift_given", "household", "received"]


def test_a_completely_missed_class_counts_as_zero_in_macro_f1():
    """TP=0 olan sinif makro-F1'e 0 olarak GIRMELI, sessizce dusmemeli.

    Onceki `compare()` `if precision and recall` diyordu: precision 0.0 falsy
    oldugu icin F1 None oluyor ve sinif ortalamadan cikiyordu. Sonuc: modelin
    HIC bilemedigi bir sinif makro-F1'i YUKSELTIYORDU. (2026-09-14'te bir test
    yakaladi; gercek rapordaki sayilari etkilemedi - her sinifta en az bir TP vardi.)

    self: 2/2 dogru -> F1 1,0. gift_given: 2 referans, 2 tahmin, 0 TP -> F1 0.
    Dogru makro = (1,0 + 0) / 2 = 0,5 ; hatali eski deger 1,0 olurdu.
    """
    ref = pl.Series(["self", "self", "gift_given", "gift_given", "household", "household"])
    pred = pl.Series(["self", "self", "household", "household", "gift_given", "gift_given"])

    out = compare(ref, pred)

    assert out["per_class"]["gift_given"]["f1"] == 0.0
    assert out["per_class"]["household"]["f1"] == 0.0
    assert out["macro_f1"] == pytest.approx((1.0 + 0.0 + 0.0) / 3, abs=1e-4)

    agirlikli = scored_measurement(
        ref.to_list(), pred.to_list(), np.ones(6), [("X", p) for p in pred.to_list()],
        n_boot=10, seed=1, min_class_n=1,
    )
    assert agirlikli["per_class"]["gift_given"]["f1"]["value"] == 0.0
    assert agirlikli["macro_f1"]["value"] == pytest.approx(1 / 3, abs=1e-4)
