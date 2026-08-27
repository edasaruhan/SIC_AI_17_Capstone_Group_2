"""Kapi 1 - Hafta 3 sonu: detektor calisiyor mu?

Girdi : data/annotations/<slug>_llm.parquet + data/interim/<slug>_annotation_sample.parquet
Cikti : reports/results/gate1_<slug>.json + reports/figures/F17_llm_vs_proxy_by_month.png

DORT OLCUT, dordu de kosudan ONCE `configs/base.yaml` -> `gate1:` altinda
sabitlendi. Her biri farkli bir ariza tipini yakaliyor:

  1. MEVSIMSELLIK   -> "sinyal yok" arizasi.
     Hediye vermek gercek dunyada mevsimseldir ve bedava sozcuksel vekil bile
     Aralik-Ocak tepesini goruyor (T14: 1,34-1,90x). LLM'in egrisi duzse
     etiketledigi sey gurultudur. Bu olcut ETIKET GEREKTIRMEZ - dis gecerlilik
     testi (roadmap V2).

  2. VEKIL OTESI RECALL -> "anahtar kelimeyi taklit etme" arizasi.
     Vekilin isaretlemedigi satirlarda LLM hic gift_given bulmuyorsa, bedava
     bir regex'in isini pahaliya tekrar ediyoruz ve butun damitma yigini
     gereksiz demektir.

  3. SEMA SAGLIGI -> parse coksu ve uydurma kanit.

  4. DUMAN TESTI -> "asiri tetikleme" arizasi.
     200 satirlik insan denemesiyle uyum. DIKKAT: bu bir DOGRULAMA DEGIL.
     Prompt v2/v3 tam bu satirlar okunarak yazildi, dolayisiyla cikan sayi
     F1 olarak raporlanamaz (DECISIONS 2026-08-26). Yalnizca "model her seye
     hediye mi diyor" sorusunu yanitlar. Deneme kosusu yoksa olcut ATLANIR ve
     raporda oyle yazar - varsayilan olarak GECMIS sayilmaz.

Yaygınlık YALNIZCA `main` cercevesinden okunur. `boost` ve `boost_received`
kendi sectigimiz seyi olcer; havuzlamak orani 7 kat sisiriyor (olculdu).

Kullanim:
    python -m gift_contamination.analysis.gate1 --config configs/base.yaml --category high
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..data.sampling import annotation_sample_path
from ..detection.llm_annotate import annotation_path, annotation_stats_path
from ..detection.schema import PurchaseType
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from . import viz
from .keyword_scan import PROXY_COL

log = get_logger("analysis.gate1")

FRAME_MAIN = "main"
GIFT = PurchaseType.GIFT_GIVEN.value


def gate1_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"gate1_{cfg.category_slug(role)}.json")


def figure_path(cfg: Config, role: str) -> Path:
    ext = cfg.get("eda.figure_format")
    return cfg.path("figures", f"F17_llm_vs_proxy_by_month_{cfg.category_slug(role)}.{ext}")


def load_joined(cfg: Config, role: str) -> pl.DataFrame:
    """LLM etiketleri + ornekleme bayraklari, `main` cercevesi.

    `kw_gift_proxy` bilerek annotation ciktisinda tasinmiyor (cikti yalin
    kalsin); vekil otesi recall icin buradan join'leniyor.
    """
    labels = pl.read_parquet(annotation_path(cfg, role))
    if "backend" in labels.columns and labels["backend"][0] != "vllm":
        raise RuntimeError(
            f"{annotation_path(cfg, role).name} '{labels['backend'][0]}' backend'iyle "
            "uretilmis. Kuru kosu ciktisi Kapi 1'e giremez."
        )
    flags = pl.read_parquet(
        annotation_sample_path(cfg, role), columns=["row_id", "category", PROXY_COL]
    )
    return labels.join(flags, on=["row_id", "category"], how="left")


# ------------------------------------------------------------ 1) mevsimsellik
def _rate(df: pl.DataFrame, months: list[int]) -> float:
    sub = df.filter(pl.col("month").is_in(months))
    return float((sub["purchase_type"] == GIFT).mean()) if sub.height else float("nan")


def seasonality(cfg: Config, main: pl.DataFrame) -> dict:
    """Tepe/dip orani + bootstrap %95 GA.

    Oran HAVUZLANMIS hesaplaniyor (tepe aylarindaki toplam hediye / toplam
    satir), aylik oranlarin ortalamasi degil: ikincisi az satirli bir ayin
    gurultusune agirlik verir.
    """
    peak = list(cfg.get("gate1.peak_months"))
    trough = list(cfg.get("gate1.trough_months"))
    ratio = _rate(main, peak) / _rate(main, trough)

    months = main["month"].to_numpy()
    is_gift = (main["purchase_type"] == GIFT).to_numpy()
    peak_mask = np.isin(months, peak)
    trough_mask = np.isin(months, trough)

    rng = np.random.default_rng(cfg.get("seed"))
    n = len(is_gift)
    draws = np.empty(cfg.get("gate1.seasonality_bootstrap_n"))
    for i in range(draws.size):
        idx = rng.integers(0, n, n)
        g, p, t = is_gift[idx], peak_mask[idx], trough_mask[idx]
        # Bir cekilis tepe veya dip ayindan hic hediye getirmezse oran tanimsiz;
        # nan birakilir ve GA nan'lari yok sayarak hesaplanir.
        draws[i] = (g[p].mean() / g[t].mean()) if p.any() and t.any() and g[t].any() \
            else np.nan

    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    threshold = cfg.get("gate1.seasonality_ratio_min")
    return {
        "peak_months": peak,
        "trough_months": trough,
        "peak_rate": round(_rate(main, peak), 6),
        "trough_rate": round(_rate(main, trough), 6),
        "ratio": round(ratio, 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "threshold": threshold,
        # Iki kosul: nokta tahmin esigin ustunde VE GA 1,0'i disliyor.
        "passed": bool(ratio >= threshold and lo > 1.0),
    }


# ------------------------------------------------------ 2) vekil otesi recall
def beyond_keyword(cfg: Config, main: pl.DataFrame) -> dict:
    unflagged = main.filter(~pl.col(PROXY_COL).fill_null(False))
    rate = float((unflagged["purchase_type"] == GIFT).mean()) if unflagged.height else 0.0
    threshold = cfg.get("gate1.beyond_keyword_rate_min")
    flagged = main.filter(pl.col(PROXY_COL).fill_null(False))
    return {
        "n_unflagged": unflagged.height,
        "n_gift_in_unflagged": int((unflagged["purchase_type"] == GIFT).sum()),
        "rate": round(rate, 6),
        # Karsilastirma icin: vekilin isaretledigi satirlarda LLM ne diyor.
        # Bu ikisi birbirine cok yakinsa model metni degil bayragi okuyordur.
        "rate_in_flagged": round(
            float((flagged["purchase_type"] == GIFT).mean()) if flagged.height else 0.0, 6
        ),
        "threshold": threshold,
        "passed": bool(rate >= threshold),
    }


# ----------------------------------------------------------- 3) sema sagligi
def schema_health(cfg: Config, role: str) -> dict:
    q = read_json(annotation_stats_path(cfg, role))["quality"]
    parse_max = cfg.get("gate1.parse_fail_rate_max")
    span_max = cfg.get("gate1.span_downgrade_rate_max")
    return {
        "parse_fail_rate": q["parse_fail_rate"],
        "parse_fail_rate_max": parse_max,
        "span_downgrade_rate": q["span_downgrade_rate"],
        "span_downgrade_rate_max": span_max,
        "passed": bool(
            q["parse_fail_rate"] <= parse_max and q["span_downgrade_rate"] <= span_max
        ),
    }


# ------------------------------------------------------------ 4) duman testi
def trial_agreement(cfg: Config) -> dict:
    """200 satirlik insan denemesiyle uyum. Kosu yoksa ATLANIR.

    Atlanan olcut GECMIS sayilmaz: raporda `passed: null` ve `skipped: true`
    goründuğu icin genel karar da beklemeye alinir.
    """
    llm = cfg.path("annotations", "prompt_trial_200_llm.parquet")
    human = cfg.path("human", "prompt_trial_200_labeled.csv")
    if not llm.exists() or not human.exists():
        return {
            "skipped": True,
            "passed": None,
            "reason": f"deneme kosusu yok ({llm.name}) veya insan etiketi yok",
        }

    joined = (
        pl.read_csv(human).select("trial_id", pl.col("label").alias("human"))
        .join(
            pl.read_parquet(llm).select("trial_id", pl.col("purchase_type").alias("llm")),
            on="trial_id", how="inner",
        )
    )
    # v2 rehberi `received`i bilmiyordu; deneme setinde alici-tarafi satir YOK
    # (olculdu, 0/200) ama LLM yine de uretebilir. O vakayi uyusmazlik saymak
    # yaniltir - insanin secebilecegi bir etiket degildi.
    agree = float((joined["human"] == joined["llm"]).mean()) if joined.height else 0.0
    threshold = cfg.get("gate1.trial_agreement_min")
    return {
        "skipped": False,
        "n_compared": joined.height,
        "agreement": round(agree, 4),
        "threshold": threshold,
        "passed": bool(agree >= threshold),
        "note": "DOGRULAMA DEGIL: prompt bu satirlar okunarak yazildi (DECISIONS 2026-08-26)",
    }


# ------------------------------------------------------------------- figur
def fig_llm_vs_proxy(
    cfg: Config, role: str, main: pl.DataFrame, out: Path, season: dict
) -> Path:
    """F17 - LLM egrisi vekil egrisinin uzerine. Kapi 1'in gorsel karsiligi.

    Baslik OLCUMU yazar, sonucu ilan etmez. Onceki hali "the LLM reproduces the
    December-January peak" diyordu - egri duz ciksa bile. Sonucu pesinen yazan
    bir baslik, sonucu gordukten sonra esik dusurmekle ayni hata.
    """
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(8.4, 4.6))

    llm = (
        main.group_by("month")
        .agg((pl.col("purchase_type") == GIFT).mean().alias("rate"))
        .sort("month")
    )
    proxy = (
        main.group_by("month")
        .agg(pl.col(PROXY_COL).fill_null(False).mean().alias("rate"))
        .sort("month")
    )

    model_label = cfg.get("detection.primary_model").split("/")[-1]
    for d, color, name, style in (
        (llm, viz.SLOTS[0], f"LLM ({model_label})", "-"),
        (proxy, viz.MUTED, "lexical proxy", "--"),
    ):
        r = d["rate"].to_numpy() * 100
        ax.plot(d["month"], r, style, color=color, marker="o",
                markeredgecolor=viz.SURFACE, markeredgewidth=1.5, label=name)
        viz.direct_label(ax, 12, r[-1], name, color)

    for m in cfg.get("gate1.peak_months"):
        ax.axvspan(m - 0.4, m + 0.4, color=viz.GRID, alpha=0.55, zorder=0)
    ax.set_xticks(range(1, 13), viz.MONTHS)
    ax.set_xlim(0.4, 13.6)
    ax.set_xlabel("month the review was written")
    ax.set_ylabel("gift_given rate (%)")
    lo, hi = season["ci95"]
    viz.titles(
        ax,
        f"F17 · {cfg.category_slug(role).replace('_', ' ')} — Dec–Jan runs "
        f"{season['ratio']:.2f}× the summer rate (95% CI {lo:.2f}–{hi:.2f})",
        f"Gate 1 threshold: {season['threshold']}× with the CI excluding 1.0. Main "
        "frame only; boost frames over-represent gifts by design. The detector reads "
        "text only and has no access to the calendar.",
    )
    viz.legend(ax, 2, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


# --------------------------------------------------------------------- kosu
def evaluate(cfg: Config, role: str) -> dict:
    df = load_joined(cfg, role)
    main = df.filter(pl.col("sample_frame") == FRAME_MAIN)
    log.info("%s: %s satir `main` cercevesinde", role, f"{main.height:,}")

    season = seasonality(cfg, main)
    criteria = {
        "1_seasonality": season,
        "2_beyond_keyword": beyond_keyword(cfg, main),
        "3_schema_health": schema_health(cfg, role),
        "4_trial_smoke_test": trial_agreement(cfg),
    }
    decided = [c["passed"] for c in criteria.values() if c["passed"] is not None]
    skipped = [k for k, c in criteria.items() if c["passed"] is None]

    report = {
        "meta": {
            "category": cfg.category_slug(role),
            "n_main": main.height,
            "prompt_version": df["prompt_version"][0],
            "model": df["model"][0],
            # Esikler kosudan once sabitlendi; raporda da gorunsun ki
            # sonradan gevsetildigi anlasilabilsin.
            "thresholds_fixed": "2026-08-28, configs/base.yaml -> gate1",
        },
        "criteria": criteria,
        "n_passed": sum(decided),
        "n_decided": len(decided),
        "skipped": skipped,
        # Atlanan olcut varsa karar VERILMEZ; sessizce "gecti" demiyoruz.
        "verdict": (
            "PASS" if all(decided) and not skipped
            else "FAIL" if not all(decided)
            else "INCOMPLETE"
        ),
    }
    fig_llm_vs_proxy(cfg, role, main, figure_path(cfg, role), season)
    write_json(report, gate1_path(cfg, role), log)

    log.info("KAPI 1 -> %s (%s/%s olcut)", report["verdict"],
             report["n_passed"], report["n_decided"])
    for name, c in criteria.items():
        mark = {True: "GECTI", False: "KALDI", None: "ATLANDI"}[c["passed"]]
        log.info("  %-20s %s", name, mark)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        evaluate(cfg, role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
