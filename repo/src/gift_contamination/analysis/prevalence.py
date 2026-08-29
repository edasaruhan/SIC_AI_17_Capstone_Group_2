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

IKI TANIM, IKISI DE RAPORLANIR. Hangisinin "kontaminasyon" sayilacagi
KAVRAMSAL bir ekip karari (CLAUDE.md 13) ve henuz verilmedi:

    dar   (C1)  = gift_given
    genis (C1b) = gift_given + household

Kod hicbirini secmiyor. Ikisi de yan yana yaziliyor ki karar SONUCA BAKARAK
verilmesin - hangisi daha buyuk cikarsa onu secmek, esigi sonuc gorulduktan
sonra dusurmekle ayni hata.

UST SINIR UYARISI. Deneme kosusundaki 33 uyusmazligin 14'u tek yonde: insan
`household` derken model `gift_given` dedi; ters yon SIFIR. Yani `gift_given`
sayisi asagi degil YUKARI saplidir ve dar tanim bir UST SINIR olarak
okunmalidir. Sayi Hafta 4 dogrulamasindan sonra duzeltilebilir hale gelir;
o zamana kadar rapor bunu acikca yaziyor.

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

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..detection.llm_annotate import annotation_path
from ..detection.schema import PurchaseType
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from . import viz
from .validation import validation_report_path

log = get_logger("analysis.prevalence")

FRAME_MAIN = "main"
GIFT = PurchaseType.GIFT_GIVEN.value
HOUSEHOLD = PurchaseType.HOUSEHOLD.value

# Dar tanim C1'e, genis tanim C1b'ye karsilik gelir (recsys/conditions.py).
DEFINITIONS: dict[str, tuple[str, ...]] = {
    "narrow": (GIFT,),
    "broad": (GIFT, HOUSEHOLD),
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


def fig_prevalence(cfg: Config, kategoriler: dict, out: Path) -> Path:
    """F19 - kategori bazli yaygınlık, iki tanim yan yana.

    Baslik OLCUMU yazar, yorumu degil. Iki tanim ayni eksende duruyor ki
    okuyucu hangisinin secildigini degil, IKISININ DE ne oldugunu gorsun.
    """
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(8.4, 4.4))

    mevcut = {v["role"] for v in kategoriler.values()}
    roller = [r for r in viz.ROLE_ORDER if r in mevcut]
    slug_of = {v["role"]: s for s, v in kategoriler.items()}

    for i, role in enumerate(roller):
        d = kategoriler[slug_of[role]]
        dar, genis = d["definitions"]["narrow"], d["definitions"]["broad"]
        # Genis tanim arkada soluk, dar tanim onde dolu. Aradaki fark household.
        ax.barh(i, genis["rate_pct"], height=0.62, color=viz.ROLE_COLOR[role],
                alpha=0.28, zorder=2)
        ax.barh(i, dar["rate_pct"], height=0.62, color=viz.ROLE_COLOR[role],
                zorder=3)
        lo, hi = dar["ci95_pct"]
        ax.plot([lo, hi], [i, i], color=viz.INK, lw=1.4, zorder=4,
                solid_capstyle="butt")
        ax.text(genis["rate_pct"] + 0.6, i,
                f"{dar['rate_pct']:.1f}%  ({genis['rate_pct']:.1f}%)",
                va="center", fontsize=9, color=viz.INK, zorder=5)

    ax.set_yticks(list(range(len(roller))))
    ax.set_yticklabels([slug_of[r].replace("_", " ") for r in roller])
    ax.invert_yaxis()
    ax.set_xlabel("share of `main` frame (%)")
    ax.set_xlim(0, max(
        k["definitions"]["broad"]["rate_pct"] for k in kategoriler.values()
    ) * 1.30)
    ax.grid(axis="x", color=viz.GRID, lw=0.8)
    ax.set_axisbelow(True)
    viz.titles(
        ax,
        "Gift prevalence by category, two definitions",
        "Solid = gift_given (C1). Pale = gift_given + household (C1b). "
        "Line = 95% Wilson CI on the narrow rate. `main` frame only, "
        "n = 10,000 per category.",
    )
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

    # Detektor insan tarafindan DOGRULANDI MI? Iddia EDILMIYOR, HESAPLANIYOR:
    # dogrulama raporu diskte yoksa cevap hayirdir. (Denetim bulgusu 7 -
    # hesaplanmamis iddia sessizce yalan soyler.)
    dogrulama = validation_report_path(cfg, int(cfg.get("validation.n")))
    dogrulandi = dogrulama.exists()

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
        },
        "categories": kategoriler,
        "ordering_narrow": [
            s for s, _ in sorted(
                kategoriler.items(),
                key=lambda kv: -kv[1]["definitions"]["narrow"]["rate_pct"],
            )
        ],
        "human_validated": dogrulandi,
        "caveats": [
            "Etiketler LLM'den; insan dogrulamasi "
            + ("TAMAM." if dogrulandi else "HENUZ YOK (Hafta 4)."),
            "Dar tanim UST SINIR: deneme kosusunda household -> gift_given "
            "yonunde 14 uyusmazlik, ters yonde 0.",
            "Oranlar k-core korpusuna aittir, ham korpusa degil.",
            "Kategoriler arasi HAVUZLANMIS oran raporlanmaz - her kategori "
            "esit tahsisle cekildi, korpus paylariyla degil.",
        ],
    }
    fig_prevalence(cfg, kategoriler, figure_path(cfg))
    write_json(report, prevalence_path(cfg), log)
    if not dogrulandi:
        log.warning(
            "human_validated=false - bu tablo Hafta 4 dogrulamasindan ONCE "
            "uretildi; makaleye girmeden once yeniden kosulmali."
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
