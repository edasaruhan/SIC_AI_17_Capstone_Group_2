"""Hafta 4 - insan dogrulamasi. Projenin TEK gercek referansi.

Girdi : data/annotations/human/validation_<n>_labeled.csv (label_A/B/C dolu)
Cikti : reports/results/validation_<n>.json + reports/figures/F18_*.png

Uc olcum var ve UCU DE AYRI SORUYA cevap veriyor:

  1. FLEISS KAPPA - "gorev tanimi acik mi?"
     Uc etiketleyici birbiriyle ne kadar uyusuyor. Dusukse sorun detektorde
     degil SEMADA: insanlar bile ayni satiri farkli etiketliyorsa modelden
     tutarlilik beklemek anlamsiz. Tek KAPI olcutu bu (esik `validation.kappa_min`).

  2. LLM vs INSAN UZLASI SI - "detektor ne kadar dogru?"
     Sinif bazli precision/recall/F1 + yon yon karisiklik. KAPI DEGIL, olcum:
     esigi hicbir belgede sabitlenmedi ve sonucu gordukten sonra bir esik
     uydurmak, kapiyi sonradan kurmak olur. Raporlanir, karara baglanmaz.

  3. VEKIL vs INSAN UZLASI SI - "LLM gercekten regex'ten iyi mi?"
     Sozcuksel vekilin ILK BAGIMSIZ precision/recall'i. Simdiye kadarki
     0,58 / 0,61 / 0,76 sayilari "yazar destekli on gecis"ti (DECISIONS
     2026-08-25, 2026-08-26); bunlar onlarin yerini alir.

KAPPA'DA SEYREK SINIF KURALI. `validation.min_class_n_for_kappa` (=20) altinda
kalan sinif genel kappa'ya GIRMEZ. Operasyonel karsiligi: o sinifi HERHANGI bir
etiketleyicinin kullandigi SATIRLAR dusurulur - kategori sutununu silmek satir
basina degerlendirici sayisini bozar ve Fleiss'in varsayimini gecersiz kilar.
Kural sonuc gorulmeden yazildi (DECISIONS 2026-08-29). Kisitlanmamis kappa da
raporlanir ki secim seffaf kalsin.

Kullanim:
    python -m gift_contamination.analysis.validation --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..data.labelsheet import LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path
from ..data.sampling import LABELS, validation_path
from ..utils.io import write_json
from ..utils.logging import get_logger
from . import viz
from .keyword_scan import PROXY_COL

log = get_logger("analysis.validation")

GIFT = "gift_given"
# Uzlasi saglanamayan satirin etiketi. Sessizce bir sinifa itmek uzlasi
# oranini SISIRIR ve tam da olcmek istedigimiz belirsizligi gizler.
TIE = "tie"


def validation_report_path(cfg: Config, n: int) -> Path:
    return cfg.path("results", f"validation_{n}.json")


def figure_path(cfg: Config, n: int) -> Path:
    ext = cfg.get("eda.figure_format", "png")
    return cfg.path("figures", f"F18_llm_vs_human_{n}.{ext}")


# ------------------------------------------------------------------- yukleme
def label_columns(df: pl.DataFrame) -> list[str]:
    """Etiketleyici kolonlari: `label_A`, `label_B`, ...

    Duz bir `startswith("label_")` YETMIYOR: `labelsheet` ciktisi ayrica
    `label_source` damgasini tasiyor ve o da onekle esleser - dordunca bir
    etiketleyici sanilir ve `load_labels` yanlis yere hata verir (olculdu).
    Etiket TAG'leri `annotator_tags` geregi TEK BUYUK HARF.
    """
    return sorted(
        c for c in df.columns
        if len(c) == len("label_") + 1
        and c.startswith("label_")
        and c[-1].isupper()
    )


def load_labels(cfg: Config, n: int | None = None) -> pl.DataFrame:
    """Doldurulmus dogrulama CSV'sini okur ve eksiksizligini dogrular."""
    n = int(n if n is not None else cfg.get("validation.n"))
    path = labeled_path(validation_path(cfg, n))
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} yok. Once sayfalari toplayin:\n"
            "  python -m gift_contamination.data.labelsheet --validation --ingest"
        )
    df = pl.read_csv(path)

    # `gate1`in backend guard'iyla AYNI gerekce. Elle yazilmis ya da sentetik
    # bir dosya gercek gorunen bir PASS uretebilir ve o rakam rapora "insan
    # dogrulamasi" diye girer. Damga yalnizca `labelsheet --ingest` yolundan
    # basiliyor.
    damga = df[LABEL_SOURCE_COLUMN][0] if LABEL_SOURCE_COLUMN in df.columns else None
    if damga != LABEL_SOURCE:
        raise RuntimeError(
            f"{path.name} '{damga}' damgasi tasiyor, '{LABEL_SOURCE}' bekleniyordu. "
            "Etiketler `labelsheet --validation --ingest` yolundan gelmeli; elle "
            "yazilmis veya sentetik bir dosya Hafta 4 olcumune giremez."
        )

    cols = label_columns(df)
    want = int(cfg.get("validation.n_annotators"))
    if len(cols) != want:
        raise ValueError(
            f"{len(cols)} etiketleyici kolonu var, {want} bekleniyordu ({cols}). "
            "Eksik bir sayfayla hesaplanan kappa yanlis olur."
        )

    eksik = df.filter(pl.any_horizontal(pl.col(c).is_null() for c in cols))
    if eksik.height:
        # Fleiss satir basina SABIT degerlendirici sayisi varsayar. Yarim
        # doldurulmus satirlari sessizce tasimak kappa'yi bozar.
        log.warning(
            "%d satirda en az bir etiketleyici bos birakmis - kappa disi kaliyor",
            eksik.height,
        )
    return df


def consensus(df: pl.DataFrame, cols: list[str]) -> pl.Series:
    """Cogunluk etiketi; cogunluk yoksa `tie`.

    Uc etiketleyicide "cogunluk yok" demek ucunun de farkli sey demesi
    demektir - o satir referans olamaz ve sayisi raporlanir.
    """
    out = []
    for row in df.select(cols).iter_rows():
        vals = [v for v in row if v is not None]
        if len(vals) < len(cols):
            out.append(None)
            continue
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        winners = [k for k, c in counts.items() if c == best]
        out.append(winners[0] if len(winners) == 1 else TIE)
    return pl.Series("consensus", out, dtype=pl.String)


# --------------------------------------------------------------------- kappa
def _rating_table(rows: list[tuple], classes: list[str]) -> np.ndarray:
    """(satir x sinif) sayim matrisi - Fleiss'in bekledigi bicim."""
    index = {c: i for i, c in enumerate(classes)}
    table = np.zeros((len(rows), len(classes)), dtype=int)
    for r, row in enumerate(rows):
        for value in row:
            table[r, index[value]] += 1
    return table


def fleiss(rows: list[tuple], classes: list[str]) -> float | None:
    """Fleiss' kappa. Satir yoksa veya tek sinif kullanildiysa None.

    Tek sinif kullanildiginda beklenen uyum 1 olur ve kappa 0/0'a duser -
    statsmodels nan dondurur. `None` dondurup raporda gormek, nan'i
    sayiymis gibi tasimaktan iyidir.
    """
    if not rows:
        return None
    from statsmodels.stats.inter_rater import fleiss_kappa  # noqa: PLC0415

    table = _rating_table(rows, classes)
    if (table.sum(axis=0) > 0).sum() < 2:
        return None
    value = float(fleiss_kappa(table, method="fleiss"))
    return None if np.isnan(value) else value


def kappa_report(
    df: pl.DataFrame, cols: list[str], *, min_class_n: int
) -> dict:
    """Genel kappa (seyrek siniflar HARIC) + sinif bazli kappa.

    Seyrek sinifi disarida birakmanin operasyonel karsiligi: o sinifi
    herhangi bir etiketleyicinin kullandigi SATIRLARI dusurmek. Kategori
    sutununu silmek satir basina degerlendirici sayisini bozardi.
    """
    tam = df.filter(~pl.any_horizontal(pl.col(c).is_null() for c in cols))
    rows = list(tam.select(cols).iter_rows())
    classes = list(LABELS)

    kullanim = {c: 0 for c in classes}
    for row in rows:
        for value in row:
            kullanim[value] = kullanim.get(value, 0) + 1

    # HIC KULLANILMAYAN sinif "seyrek" degildir - onun icin dusen satir da yok.
    # Listeye koymak raporu gurultuyle doldurur ve gercekten dislanan sinifi
    # gozden kacirtir.
    seyrek = sorted(c for c in classes if 0 < kullanim[c] < min_class_n)
    tutulan = [c for c in classes if c not in seyrek]
    kalan = [r for r in rows if not (set(r) & set(seyrek))]

    return {
        "n_rows": len(rows),
        "n_incomplete_rows": df.height - len(rows),
        "n_annotators": len(cols),
        "ratings_by_class": kullanim,
        "min_class_n": min_class_n,
        # KAPI degeri bu: seyrek siniflarin gectigi satirlar disarida.
        "kappa": fleiss(kalan, tutulan),
        "n_rows_in_kappa": len(kalan),
        "sparse_classes": seyrek,
        "n_rows_dropped_as_sparse": len(rows) - len(kalan),
        # Seffaflik icin: hicbir sey dislanmadan hesaplanan deger.
        "kappa_unrestricted": fleiss(rows, classes),
        # Bire-karsi-hepsi. Seyrek sinif da burada gorunur - dislanmasi
        # KAPI kararindan, raporlamadan degil.
        "kappa_per_class": {
            c: fleiss([tuple(x == c for x in r) for r in rows], [True, False])
            for c in classes
            if kullanim[c] > 0
        },
    }


# ---------------------------------------------------------------- kiyaslama
def compare(reference: pl.Series, prediction: pl.Series) -> dict:
    """Sinif bazli precision/recall/F1 + yon yon uyusmazlik.

    `gate1.trial_agreement` ile AYNI bicim - iki rapor yan yana okunabilsin.
    """
    ref = reference.to_list()
    pred = prediction.to_list()
    pairs = [(r, p) for r, p in zip(ref, pred) if r is not None and p is not None]
    if not pairs:
        # Bicim HER IKI dalda ayni: cagiran taraf hangi daldan geldigini
        # bilmek zorunda kalmasin.
        return {
            "n_compared": 0, "accuracy": None, "macro_f1": None,
            "per_class": {}, "disagreements": {},
        }

    classes = sorted({r for r, _ in pairs} | {p for _, p in pairs})
    per_class = {}
    for c in classes:
        tp = sum(1 for r, p in pairs if r == c and p == c)
        n_ref = sum(1 for r, _ in pairs if r == c)
        n_pred = sum(1 for _, p in pairs if p == c)
        precision = round(tp / n_pred, 3) if n_pred else None
        recall = round(tp / n_ref, 3) if n_ref else None
        f1 = (
            round(2 * precision * recall / (precision + recall), 3)
            if precision and recall
            else None
        )
        per_class[c] = {
            "n_reference": n_ref, "n_predicted": n_pred,
            "precision": precision, "recall": recall, "f1": f1,
        }

    yanlis: dict[str, int] = {}
    for r, p in pairs:
        if r != p:
            yanlis[f"{r}->{p}"] = yanlis.get(f"{r}->{p}", 0) + 1

    return {
        "n_compared": len(pairs),
        "accuracy": round(sum(1 for r, p in pairs if r == p) / len(pairs), 4),
        "macro_f1": round(
            float(np.mean([v["f1"] for v in per_class.values() if v["f1"] is not None])), 4
        ) if any(v["f1"] is not None for v in per_class.values()) else None,
        "per_class": per_class,
        "disagreements": dict(sorted(yanlis.items(), key=lambda kv: -kv[1])),
    }


def _proxy_labels(df: pl.DataFrame) -> pl.Series:
    """Sozcuksel vekili bes sinifli semaya cevirir.

    Vekil ikili: ya `gift_given` der ya demez. Demedigi satirlari `self`
    saymak onu HAKSIZ yere cezalandirirdi (household/unclear zaten ayirt
    edemez), o yuzden yalnizca `gift_given` ekseninde kiyasliyoruz:
    isaretledi -> gift_given, isaretlemedi -> not_gift.
    """
    return (
        pl.when(pl.col(PROXY_COL)).then(pl.lit(GIFT)).otherwise(pl.lit("not_gift"))
    ).alias("proxy")


def _binary_gift(series: pl.Series, name: str) -> pl.Series:
    return pl.Series(
        name, [None if v is None else (GIFT if v == GIFT else "not_gift") for v in series]
    )


# ------------------------------------------------------------------- figur
def fig_llm_vs_human(cfg: Config, per_class: dict, out: Path) -> Path:
    """F18 - sinif bazli F1. Baslik OLCUMU yazar, sonuc ilan etmez."""
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(7.6, 4.2))

    names = [c for c in LABELS if c in per_class]
    f1 = [(per_class[c]["f1"] or 0) for c in names]
    n = [per_class[c]["n_reference"] for c in names]

    bars = ax.barh(range(len(names)), f1, color=viz.SLOTS[0])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{c}\n(n={m})" for c, m in zip(names, n)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (insan uzlasisina karsi)")
    for bar, value in zip(bars, f1):
        ax.text(min(value + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", fontsize=9)

    viz.titles(
        ax,
        f"F18 · LLM'in sinif bazli F1'i — {sum(n)} satir, insan uzlasisi referans",
        "Uc bagimsiz etiketleyicinin cogunlugu referans alindi. Bu bir KAPI degil "
        "olcumdur: F1 icin hicbir esik sabitlenmedi. Dogrulama seti yaygınlık "
        "ornegi degildir - `household` ve `received` bilerek fazla temsil edildi.",
    )
    ax.grid(axis="y", visible=False)
    return viz.save(fig, out, log)


# --------------------------------------------------------------------- kosu
def evaluate(cfg: Config, n: int | None = None) -> dict:
    n = int(n if n is not None else cfg.get("validation.n"))
    df = load_labels(cfg, n)
    cols = label_columns(df)

    kappa = kappa_report(
        df, cols, min_class_n=int(cfg.get("validation.min_class_n_for_kappa"))
    )
    threshold = float(cfg.get("validation.kappa_min"))
    agreement = {
        **kappa,
        "threshold": threshold,
        "passed": None if kappa["kappa"] is None else bool(kappa["kappa"] >= threshold),
    }

    df = df.with_columns(consensus(df, cols))
    ref = df["consensus"]
    # `tie` satirlari referans olamaz: uzlasi yok demek dogru cevap yok demek.
    usable = df.filter(pl.col("consensus").is_not_null() & (pl.col("consensus") != TIE))

    vs_llm = compare(usable["consensus"], usable["purchase_type"])
    vs_proxy = compare(
        _binary_gift(usable["consensus"], "consensus"),
        usable.select(_proxy_labels(usable))["proxy"],
    )

    report = {
        "meta": {
            "n": n,
            "n_annotators": len(cols),
            "model": df["model"][0] if "model" in df.columns else None,
            "prompt_version": df["prompt_version"][0] if "prompt_version" in df.columns else None,
            "thresholds_fixed": "2026-08-29, configs/base.yaml -> validation",
        },
        # TEK kapi olcutu. Digerleri olcum.
        "criteria": {"1_annotator_agreement": agreement},
        "measurements": {
            "n_tie": int((ref == TIE).sum()),
            "n_usable": usable.height,
            "consensus_distribution": dict(
                sorted(usable.group_by("consensus").len().iter_rows())
            ),
            # DIKKAT: bunlar KAPI DEGIL. F1 icin hicbir esik sabitlenmedi ve
            # sonucu gordukten sonra esik uydurmak kapiyi sonradan kurmaktir.
            "llm_vs_human": vs_llm,
            # Vekilin ILK bagimsiz olcumu. Onceki 0,58/0,61 yazar destekliydi.
            "proxy_vs_human_gift_only": vs_proxy,
        },
        "verdict": (
            "PASS" if agreement["passed"]
            else "INCOMPLETE" if agreement["passed"] is None
            else "FAIL"
        ),
    }

    if vs_llm["per_class"]:
        fig_llm_vs_human(cfg, vs_llm["per_class"], figure_path(cfg, n))
    write_json(report, validation_report_path(cfg, n), log)

    log.info("HAFTA 4 -> %s", report["verdict"])
    log.info("  Fleiss kappa      %s (esik %.2f, %d satir)",
             kappa["kappa"], threshold, kappa["n_rows_in_kappa"])
    if kappa["sparse_classes"]:
        log.info("  kappa disi sinif  %s (n < %d) - %d satir dusuruldu",
                 ", ".join(kappa["sparse_classes"]), kappa["min_class_n"],
                 kappa["n_rows_dropped_as_sparse"])
    log.info("  LLM dogrulugu     %s (macro F1 %s) - OLCUM, kapi degil",
             vs_llm["accuracy"], vs_llm["macro_f1"])
    log.info("  vekil (gift ekseni) %s", vs_proxy["per_class"].get(GIFT))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=None, help="Dogrulama seti boyutu")
    args = parser.parse_args(argv)

    evaluate(Config.load(args.config), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
