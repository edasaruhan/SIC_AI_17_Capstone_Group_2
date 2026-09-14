"""RQ1 - Hediye yaygınlığı: dort kategoride oran tahmini (Hafta 5).

Girdi : data/annotations/<slug>_llm.parquet
Cikti : reports/results/prevalence.json
        reports/figures/F19_prevalence_by_category.png

Bu modul projenin BIRINCI arastirma sorusunu yanitliyor: hediye alimlari
korpusun ne kadarini olusturuyor ve kategoriye gore degisiyor mu?

YALNIZCA `main` CERCEVESI OKUNUR. `boost` ve `boost_received` kendi
sectigimiz seyi olcer; havuzlamak orani yedi kat sisiriyor (olculdu, Hafta 2).
`main` orantili tahsisle cekildigi icin KENDINDEN AGIRLIKLI - duz ortalama
yansiz tahmindir, ayrica agirliklandirma gerekmez.

IKI TANIM, IKISI DE RAPORLANIR (karar 2026-09-14, CLAUDE.md 13):

    dar   (C1,  birincil)   = gift_given
    genis (C1b, saglamlik)  = gift_given + household + received

Ikisi de yan yana yaziliyor ki hangisinin buyuk ciktigina bakilarak secim
yapilmasin.

UST SINIR UYARISI. Deneme kosusundaki 33 uyusmazligin 14'u tek yonde: insan
`household` derken model `gift_given` dedi; ters yon SIFIR. Yani HAM
`gift_given` orani YUKARI saplidir.

INSAN KALIBRASYONU (2026-09-14). Hafta 4 dogrulamasi LLM etiketine gore
katmanli cekildigi icin her LLM sinifi k icin "insanin gozunde gercekte ne
oldugu" tahmin edilebilir: `PPV_k = P(insan in D | LLM = k)`. Duzeltilmis
oran `sum_k pay_main,c(k) * PPV_k`. Yontem A'nin etiketleri modelle
karsilastirilmadan ONCE kaydedildi (DECISIONS 2026-09-14). Referans TEK
etiketleyici; etiket guvenilirligi olculmedi - sayi bu sinirla okunur.

GUVEN ARALIGI. Wilson skor araligi. `main` katmanli (month x rating x
uzunluk) oldugu icin gercek varyans basit rastgele ornekten KUCUK ya da ona
esittir - dolayisiyla bu aralik muhafazakardir, dar degil.

Kullanim:
    python -m gift_contamination.analysis.prevalence --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..data.sampling import LABELS
from ..detection.llm_annotate import annotation_path
from ..detection.schema import PurchaseType
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from . import viz
from .validation import (
    TIE,
    annotator_mode,
    label_columns,
    load_labels,
    reference_labels,
    validation_report_path,
)

log = get_logger("analysis.prevalence")

FRAME_MAIN = "main"
GIFT = PurchaseType.GIFT_GIVEN.value
HOUSEHOLD = PurchaseType.HOUSEHOLD.value
RECEIVED = PurchaseType.RECEIVED.value

# Dar tanim C1'e, genis tanim C1b'ye karsilik gelir (recsys/conditions.py).
# `received` genise 2026-09-14'te girdi: "alici urunu kendisi secmedi".
DEFINITIONS: dict[str, tuple[str, ...]] = {
    "narrow": (GIFT,),
    "broad": (GIFT, HOUSEHOLD, RECEIVED),
}
Z95 = 1.959963985


def prevalence_path(cfg: Config) -> Path:
    return cfg.path("results", "prevalence.json")


def figure_path(cfg: Config) -> Path:
    ext = cfg.get("eda.figure_format")
    return cfg.path("figures", f"F19_prevalence_by_category.{ext}")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson skor araligi, yuzde olarak.

    Normal yaklasim (p +- z*sqrt(p(1-p)/n)) kucuk oranlarda sinirin disina
    tasar; `received` %0,6 seviyesinde ve orada Wald araligi negatife
    dusebiliyor. Wilson bu kusuru tasimiyor.
    """
    if n <= 0:
        return (None, None)
    p = k / n
    payda = 1 + z * z / n
    merkez = (p + z * z / (2 * n)) / payda
    yari = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / payda
    return (round(100 * (merkez - yari), 2), round(100 * (merkez + yari), 2))


def load_main(cfg: Config, role: str) -> pl.DataFrame:
    """`main` cercevesi + kuru kosu korumasi (gate1 ile ayni desen)."""
    path = annotation_path(cfg, role)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} yok. Once etiketleyin:\n"
            f"  python -m gift_contamination.detection.llm_annotate --category {role}"
        )
    labels = pl.read_parquet(path)
    if "backend" in labels.columns and labels["backend"][0] != "vllm":
        raise RuntimeError(
            f"{path.name} '{labels['backend'][0]}' backend'iyle uretilmis. "
            "Kuru kosu ciktisi yaygınlık tablosuna giremez."
        )
    main = labels.filter(pl.col("sample_frame") == FRAME_MAIN)
    if not main.height:
        raise RuntimeError(f"{path.name}: `main` cercevesinde hic satir yok.")
    return main


def proxy_rate(cfg: Config, role: str) -> float | None:
    """Hafta 1'in sozcuksel vekil orani - LLM'i karsilastirmak icin.

    Vekil BEDAVA. LLM'i kosmamizin sebebi vekilin pahali olmasi degil,
    KACIRIYOR olmasi; bu oran o farki sayiya cevirir.
    """
    path = cfg.path("results", f"keyword_rates_{cfg.category_slug(role)}.json")
    if not path.exists():
        return None
    return float(read_json(path)["proxy_rate"])


def category_stats(cfg: Config, role: str) -> dict:
    """Tek kategori: sinif sayilari, iki tanim altinda oran ve guven araligi."""
    main = load_main(cfg, role)
    n = main.height
    # `.len().iter_rows()` DUZ demet veriyor: (deger, sayi). Demet anahtar degil.
    sayilar = dict(main.group_by("purchase_type").len().iter_rows())

    tanimlar = {}
    for ad, etiketler in DEFINITIONS.items():
        k = int(sum(sayilar.get(e, 0) for e in etiketler))
        lo, hi = wilson(k, n)
        tanimlar[ad] = {
            "labels": list(etiketler),
            "k": k,
            "rate_pct": round(100 * k / n, 2),
            "ci95_pct": [lo, hi],
        }

    vekil = proxy_rate(cfg, role)
    dar = tanimlar["narrow"]["rate_pct"]
    return {
        "role": role,
        "n_main": n,
        "counts": {k: int(v) for k, v in sorted(sayilar.items())},
        "share_pct": {k: round(100 * v / n, 2) for k, v in sorted(sayilar.items())},
        "definitions": tanimlar,
        "proxy_rate_pct": round(100 * vekil, 2) if vekil is not None else None,
        # Vekil kac kat kaciriyor. 1'e yakinsa LLM bedava bir regex'i taklit
        # ediyor demektir ve butun damitma yigini gereksizdir (Kapi 1, olcut 2).
        "llm_over_proxy": round(dar / (100 * vekil), 2) if vekil else None,
    }


# ------------------------------------------- insan kalibrasyonu (2026-09-14)
def _ppv_sources(
    ref: dict[str, list[str]], ref_all: dict[str, list[str]], *, min_cell_n: int
) -> dict[str, tuple[list[str], str]]:
    """Her LLM sinifi icin PPV'nin hangi satirlardan hesaplanacagi.

    Onceden kayitli kural: `main` satiri `min_cell_n`'den az olan sinif tum
    cercevelerin satirlarina duser ve bu ISARETLENIR.
    """
    out = {}
    for k in LABELS:
        rows = ref.get(k, [])
        if len(rows) >= min_cell_n:
            out[k] = (rows, "main")
        else:
            out[k] = (ref_all.get(k, []), "all_frames")
    return out


def _ppv(rows: list[str], labels: tuple[str, ...]) -> float:
    return float(np.mean([r in labels for r in rows])) if rows else float("nan")


def calibrated_rate(shares: dict[str, float], ppv: dict[str, float]) -> float:
    """`sum_k pay(k) * PPV_k`. Payi olmayan sinif katki yapmaz.

    PPV'si tanimsiz (hic satiri olmayan) bir sinifin payi SIFIR degilse sonuc
    tanimsizdir - o kutleyi sessizce sifir saymak orani asagi ceker.
    """
    toplam = 0.0
    for k, s in shares.items():
        if s == 0:
            continue
        p = ppv.get(k, float("nan"))
        if np.isnan(p):
            return float("nan")
        toplam += s * p
    return toplam


def _by_class(df: pl.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, r in zip(df["purchase_type"].to_list(), df["reference"].to_list()):
        out.setdefault(k, []).append(r)
    return out


def calibration(cfg: Config, kategoriler: dict) -> dict:
    """Insan kalibrasyonlu yaygınlık - dort kategori, iki tanim, bootstrap GA.

    Dogrulama etiketleri henuz yoksa blok `skipped` doner; yaygınlık tablosu
    kalibrasyonsuz uretilmeye devam eder.
    """
    try:
        df = load_labels(cfg)
    except FileNotFoundError:
        return {"skipped": True, "reason": "dogrulama etiketleri henuz yok"}

    tags, mode = annotator_mode(cfg)
    df = df.with_columns(reference_labels(df, label_columns(df)))
    df = df.filter(pl.col("reference").is_not_null() & (pl.col("reference") != TIE))
    min_cell_n = int(cfg.get("validation.min_cell_n"))
    n_boot = int(cfg.get("validation.bootstrap_n"))
    seed = int(cfg.get("seed"))

    main_val = df.filter(pl.col("sample_frame") == FRAME_MAIN)
    kaynak = _ppv_sources(_by_class(main_val), _by_class(df), min_cell_n=min_cell_n)

    def ppv_table(sources: dict[str, tuple[list[str], str]]) -> dict[str, dict[str, float]]:
        return {
            ad: {k: _ppv(rows, etiketler) for k, (rows, _) in sources.items()}
            for ad, etiketler in DEFINITIONS.items()
        }

    ppv = ppv_table(kaynak)
    ppv_all = ppv_table({k: (rows, "all") for k, rows in _by_class(df).items()})

    rng = np.random.default_rng(seed)
    sonuc: dict[str, dict] = {}
    duyarlilik: dict[str, dict] = {}
    for slug, d in kategoriler.items():
        n = int(d["n_main"])
        shares = {k: d["counts"].get(k, 0) / n for k in LABELS}
        sonuc[slug] = {}
        duyarlilik[slug] = {}
        for ad, etiketler in DEFINITIONS.items():
            nokta = calibrated_rate(shares, ppv[ad])
            boots = np.empty(n_boot)
            p_vec = np.array([shares[k] for k in LABELS])
            isabet = {
                k: np.array([r in etiketler for r in rows], dtype=float)
                for k, (rows, _) in kaynak.items()
            }
            for b in range(n_boot):
                # PPV: sinif icinde yeniden ornekleme; paylar: kategori icinde
                # (10.000 satir uzerinde multinomial - satir yeniden orneklemeyle esdeger)
                ppv_b = {
                    k: (float(h[rng.integers(0, len(h), len(h))].mean()) if len(h)
                        else float("nan"))
                    for k, h in isabet.items()
                }
                sayim = rng.multinomial(n, p_vec)
                boots[b] = calibrated_rate(
                    {k: sayim[i] / n for i, k in enumerate(LABELS)}, ppv_b
                )
            ok = boots[~np.isnan(boots)]
            sonuc[slug][ad] = {
                "raw_pct": d["definitions"][ad]["rate_pct"],
                "calibrated_pct": None if np.isnan(nokta) else round(100 * nokta, 2),
                "ci95_pct": (
                    [round(100 * float(np.percentile(ok, 2.5)), 2),
                     round(100 * float(np.percentile(ok, 97.5)), 2)]
                    if len(ok) else None
                ),
            }
            yedek = calibrated_rate(shares, ppv_all[ad])
            duyarlilik[slug][ad] = None if np.isnan(yedek) else round(100 * yedek, 2)

    # --- Varsayim kontrolu (secim icin KULLANILMAZ): PPV kategoriden bagimsiz mi?
    # En cok satiri olan kategori (Toys, rol agirligi 2) kendi PPV'siyle.
    kontrol = None
    if kategoriler:
        en_buyuk = max(
            kategoriler, key=lambda s: main_val.filter(pl.col("category") == s).height
        )
        kendi = main_val.filter(pl.col("category") == en_buyuk)
        if kendi.height:
            yerel = _ppv_sources(_by_class(kendi), _by_class(df), min_cell_n=min_cell_n)
            d = kategoriler[en_buyuk]
            shares = {k: d["counts"].get(k, 0) / int(d["n_main"]) for k in LABELS}
            kontrol = {
                "category": en_buyuk,
                "n_validation_main_rows": kendi.height,
                "classes_falling_back": sorted(k for k, (_, s) in yerel.items() if s != "main"),
                **{
                    ad: {
                        "pooled_ppv_pct": sonuc[en_buyuk][ad]["calibrated_pct"],
                        "own_ppv_pct": _pct(calibrated_rate(
                            shares, {k: _ppv(r, e) for k, (r, _) in yerel.items()}
                        )),
                    }
                    for ad, e in DEFINITIONS.items()
                },
            }

    return {
        "method": (
            "PPV_k = P(insan in D | LLM = k), main satirlarindan kategoriler "
            "havuzlanarak; duzeltilmis oran = sum_k pay_main(k) * PPV_k"
        ),
        "reference": {
            "annotators": tags,
            "reliability_measured": mode != "none",
        },
        "ppv": {
            ad: {
                k: {
                    "value": None if np.isnan(v) else round(v, 4),
                    "n": len(kaynak[k][0]),
                    "source": kaynak[k][1],
                }
                for k, v in tablo.items()
            }
            for ad, tablo in ppv.items()
        },
        "by_category": sonuc,
        "assumption_check": kontrol,
        "sensitivity_all_frames_pct": duyarlilik,
        "bootstrap": {"n": n_boot, "seed": seed},
    }


def _pct(x: float) -> float | None:
    return None if np.isnan(x) else round(100 * x, 2)


def human_validation_status(cfg: Config) -> dict:
    """Insan dogrulamasinin DURUMU - diskteki rapordan okunur, iddia edilmez.

    Onceki `human_validated: bool` tek etiketleyiciyle yaniltici olurdu:
    rapor var diye "dogrulandi" yazmak, olculmemis guvenilirligi olculmus
    gosterirdi (denetim bulgusu 7'nin ayni sinifi).
    """
    path = validation_report_path(cfg, int(cfg.get("validation.n")))
    if not path.exists():
        return {"report_exists": False, "n_annotators": None, "reliability_measured": None}
    rapor = read_json(path)
    kriter = rapor["criteria"]["1_annotator_agreement"]
    return {
        "report_exists": True,
        "n_annotators": rapor["meta"]["n_annotators"],
        "reliability_measured": kriter.get("passed") is not None,
        "verdict": rapor["verdict"],
    }


def fig_prevalence(cfg: Config, kategoriler: dict, out: Path, kalibrasyon: dict | None = None) -> Path:
    """F19 - kategori bazli yaygınlık, iki tanim yan yana.

    Baslik OLCUMU yazar, yorumu degil. Iki tanim ayni eksende duruyor ki
    okuyucu hangisinin secildigini degil, IKISININ DE ne oldugunu gorsun.
    Kalibrasyon varsa insan kalibrasyonlu oranlar elmas isaretle ustte.
    """
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(8.4, 4.6))

    mevcut = {v["role"] for v in kategoriler.values()}
    roller = [r for r in viz.ROLE_ORDER if r in mevcut]
    slug_of = {v["role"]: s for s, v in kategoriler.items()}
    kal = None if not kalibrasyon or kalibrasyon.get("skipped") else kalibrasyon["by_category"]

    for i, role in enumerate(roller):
        slug = slug_of[role]
        d = kategoriler[slug]
        dar, genis = d["definitions"]["narrow"], d["definitions"]["broad"]
        # Genis tanim arkada soluk, dar tanim onde dolu.
        ax.barh(i, genis["rate_pct"], height=0.62, color=viz.ROLE_COLOR[role],
                alpha=0.28, zorder=2)
        ax.barh(i, dar["rate_pct"], height=0.62, color=viz.ROLE_COLOR[role],
                zorder=3)
        lo, hi = dar["ci95_pct"]
        ax.plot([lo, hi], [i, i], color=viz.INK, lw=1.4, zorder=4,
                solid_capstyle="butt")
        uc = genis["rate_pct"]
        if kal and slug in kal:
            for ad, dolu in (("narrow", True), ("broad", False)):
                c = kal[slug][ad]
                if c["calibrated_pct"] is None:
                    continue
                y = i - 0.42
                if c["ci95_pct"]:
                    ax.plot(c["ci95_pct"], [y, y], color=viz.INK, lw=1.0, zorder=5)
                    uc = max(uc, c["ci95_pct"][1])
                ax.plot(c["calibrated_pct"], y, marker="D", markersize=5.5, zorder=6,
                        color=viz.INK, markerfacecolor=viz.INK if dolu else viz.SURFACE)
        ax.text(uc + 0.6, i,
                f"{dar['rate_pct']:.1f}%  ({genis['rate_pct']:.1f}%)",
                va="center", fontsize=9, color=viz.INK, zorder=5)

    ax.set_yticks(list(range(len(roller))))
    ax.set_yticklabels([slug_of[r].replace("_", " ") for r in roller])
    ax.invert_yaxis()
    ax.set_xlabel("share of `main` frame (%)")
    ust = max(k["definitions"]["broad"]["rate_pct"] for k in kategoriler.values())
    if kal:
        ust = max([ust] + [
            c[ad]["ci95_pct"][1] for c in kal.values() for ad in DEFINITIONS
            if c[ad]["ci95_pct"]
        ])
    ax.set_xlim(0, ust * 1.30)
    ax.grid(axis="x", color=viz.GRID, lw=0.8)
    ax.set_axisbelow(True)
    alt = (
        "Bars = LLM labels. Solid = gift_given (C1), pale = gift_given + household + "
        "received (C1b); black line = 95% Wilson CI. "
    )
    if kal:
        alt += (
            "Diamonds = human-calibrated rate with bootstrap 95% CI (filled C1, hollow "
            "C1b); reference is a SINGLE annotator, reliability not measured. "
        )
    alt += "`main` frame only, n = 10,000 per category."
    viz.titles(ax, "Gift prevalence by category, two definitions", alt)
    return viz.save(fig, out, log)


def evaluate(cfg: Config, roles: list[str]) -> dict:
    kategoriler = {}
    for role in roles:
        d = category_stats(cfg, role)
        kategoriler[cfg.category_slug(role)] = d
        lo, hi = d["definitions"]["narrow"]["ci95_pct"]
        log.info(
            "%-26s dar %5.2f%% [%5.2f, %5.2f]  genis %5.2f%%  vekil %s%% (%sx)",
            cfg.category_slug(role), d["definitions"]["narrow"]["rate_pct"],
            lo, hi, d["definitions"]["broad"]["rate_pct"],
            d["proxy_rate_pct"], d["llm_over_proxy"],
        )

    ornek = next(iter(kategoriler))
    ilk = load_main(cfg, roles[0])

    # Insan dogrulamasinin DURUMU iddia edilmiyor, diskten okunuyor.
    durum = human_validation_status(cfg)
    kalibrasyon = calibration(cfg, kategoriler)

    if not durum["report_exists"]:
        insan = "Etiketler LLM'den; insan dogrulamasi HENUZ YOK (Hafta 4)."
    elif durum["n_annotators"] == 1:
        insan = (
            "Insan referansi TEK etiketleyici; etiket guvenilirligi OLCULMEDI "
            f"(Hafta 4 karari: {durum['verdict']}). Kalibrasyonlu oranlar bu sinirla okunur."
        )
    else:
        insan = (
            f"Insan dogrulamasi: {durum['n_annotators']} etiketleyici, "
            f"Hafta 4 karari {durum['verdict']}."
        )

    report = {
        "meta": {
            "frame": FRAME_MAIN,
            "n_per_category": int(kategoriler[ornek]["n_main"]),
            "model": ilk["model"][0],
            "prompt_version": ilk["prompt_version"][0],
            "estimator": "flat mean (the main frame is proportionally "
                         "allocated, hence self-weighting)",
            "ci": "Wilson score, 95%",
            "definitions": {k: list(v) for k, v in DEFINITIONS.items()},
            # Ornekleme `clean` korpustan cekildi (sampling.py, keyword_scan.py);
            # 5-core deney korpusu DEGIL. Olculdu 2026-09-14: Toys annotation
            # satirlarinin yalnizca %17,6'si 5-core'da.
            "corpus": "clean (verified + min_words + dedup), NOT the 5-core experiment corpus",
        },
        "categories": kategoriler,
        "ordering_narrow": [
            s for s, _ in sorted(
                kategoriler.items(),
                key=lambda kv: -kv[1]["definitions"]["narrow"]["rate_pct"],
            )
        ],
        "human_validation": durum,
        "calibration": kalibrasyon,
        "caveats": [
            insan,
            "HAM dar tanim UST SINIR: deneme kosusunda household -> gift_given "
            "yonunde 14 uyusmazlik, ters yonde 0.",
            "Oranlar clean korpusuna aittir (verified + min_words + dedup); "
            "5-core deney korpusuna degil.",
            "Kategoriler arasi HAVUZLANMIS oran raporlanmaz - her kategori "
            "esit tahsisle cekildi, korpus paylariyla degil.",
        ],
    }
    fig_prevalence(cfg, kategoriler, figure_path(cfg), kalibrasyon)
    write_json(report, prevalence_path(cfg), log)
    if not durum["report_exists"]:
        log.warning(
            "insan dogrulamasi raporu yok - bu tablo Hafta 4'ten ONCE uretildi; "
            "makaleye girmeden once yeniden kosulmali."
        )
    if not kalibrasyon.get("skipped"):
        for slug, c in kalibrasyon["by_category"].items():
            log.info(
                "%-26s kalibrasyonlu dar %s%% %s  genis %s%% %s",
                slug, c["narrow"]["calibrated_pct"], c["narrow"]["ci95_pct"],
                c["broad"]["calibrated_pct"], c["broad"]["ci95_pct"],
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    evaluate(cfg, resolve_roles(cfg, args.category))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
