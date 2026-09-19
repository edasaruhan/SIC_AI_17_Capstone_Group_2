"""Sonuc figurleri F20-F23 (Faz 6). Yalnizca raporlanmis JSON'lari okur; hesap YAPMAZ.

Girdi : reports/results/distill_report_<model>.json   -> F20 damitma sadakati
        reports/results/experiment_stats.json         -> F21 kosul karsitliklari (RQ2/RQ3)
        reports/results/marketing_metrics.json        -> F22 M2 yari omur, F23 M1 israf payi
Cikti : reports/figures/F20..F23_*.png

Figurler sayiyi uretmez, gosterir: her nokta ve aralik JSON'da birebir var (tablo gorunumu
JSON'un kendisi). Girdi yoksa figur atlanir ve loglanir.

Kodlama kurallari (`viz` ile ayni sistem): kategori rengi `viz.ROLE_COLOR`'dan ve ikinci
kodlama olarak marker SEKLI (Grocery'nin aqua'si acik yuzeyde 3:1'in altinda - dogrulayici
ile olculdu); tek eksen; sifir cizgisi referans. Kapi 2 PASS olmayan nokta ICI BOS cizilir
ve alt baslik bunu soyler. `reportable: false` girdiden uretilen figurun basliginda
"SMOKE TEST" yazar - sonuc raporuna giremez.

Kullanim:
    python -m gift_contamination.analysis.result_figures --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ..config import Config
from ..recsys.marketing_metrics import LAMBDA_MIN
from ..utils.io import read_json
from ..utils.logging import get_logger
from . import viz

log = get_logger("analysis.result_figures")

ROLE_MARKER = {"high": "o", "low": "s", "mid": "D", "pilot": "^"}
# Iki eksen ayni figurde: dogrulanmis sirada komsu slotlar (mavi, turuncu).
AXIS_COLOR = {"C1": viz.SLOTS[0], "C1b": viz.SLOTS[1]}
AXIS_MARKER = {"C1": "o", "C1b": "s"}
F21_ROWS = (("C1-C4", "C1 − C4 · RQ2 primary"), ("C1-C0", "C1 − C0 · RQ2"),
            ("C3-C1", "C3 − C1 · RQ3"), ("C3-C0", "C3 − C0 · RQ3"),
            ("C1b-C4b", "C1b − C4b · robustness"))
F23_ROWS = (("C0-C1", "C0 − C1 · narrow"), ("C4-C1", "C4 − C1 · narrow"),
            ("C0-C3", "C0 − C3 · narrow"), ("C0-C1b", "C0 − C1b · broad"),
            ("C4b-C1b", "C4b − C1b · broad"))
SMOKE = "SMOKE TEST — NOT REPORTABLE. "


def _short(slug: str) -> str:
    return slug.split("_")[0]


def display_model_name(name) -> str:
    """HF adi oldugu gibi; yerel klasor yolu yalnizca son parca (figur kullanici adini sizdirmasin)."""
    ad = str(name or "")
    if "\\" in ad or ":" in ad or ad.startswith("/"):
        ad = ad.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return ad


def _ext(cfg: Config) -> str:
    return cfg.get("eda.figure_format")


# ------------------------------------------------------------------ forest
def _forest(ax, rows: list[tuple[str, str]], series: list[dict], *, scale: float = 100.0,
            zero_line: bool = True) -> None:
    """Nokta + GA araligi. series: {label, color, marker, points: {satir: (fark, [lo, hi], dolu)}}."""
    offs = np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else [0.0]
    for s, off in zip(series, offs):
        etiketlendi = False
        for i, (anahtar, _) in enumerate(rows):
            p = s["points"].get(anahtar)
            if p is None:
                continue
            fark, (lo, hi), dolu = p
            y = i + off
            ax.plot([lo * scale, hi * scale], [y, y], color=s["color"], lw=2, solid_capstyle="round", zorder=3)
            ax.plot([fark * scale], [y], linestyle="none", marker=s["marker"], ms=8, color=s["color"],
                    markerfacecolor=s["color"] if dolu else viz.SURFACE, markeredgewidth=1.6, zorder=4,
                    label=None if etiketlendi else s["label"])
            if not etiketlendi:
                # Kimlik renge tek basina birakilmaz: ilk satirda dogrudan etiket.
                viz.direct_label(ax, hi * scale, y, s["label"], viz.INK_2)
                etiketlendi = True
    if zero_line:
        ax.axvline(0, color=viz.INK_2, lw=1, zorder=2)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def _header(fig, title: str, subtitle: str, *, top: float = 0.78) -> None:
    """Figur basligi + alt baslik, eksenlerin USTUNDE ayri bir blokta (cakisma olmasin)."""
    import textwrap  # noqa: PLC0415

    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=viz.INK)
    fig.text(0.01, 0.915, "\n".join(textwrap.wrap(subtitle, 150)), ha="left", va="top", fontsize=9,
             color=viz.MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, top))


def _hollow_note(any_hollow: bool) -> str:
    return " Hollow markers: Gate 2 did not pass (or was not evaluated) — not interpretable." if any_hollow else ""


# ------------------------------------------------------------------ F21
def fig_contrasts(cfg: Config, stats: dict, out: Path) -> Path | None:
    models = list(cfg.get("experiment.models"))
    roles = list(cfg.get("experiment.categories"))
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, axes = viz.plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), sharex=True, squeeze=False)
    bos_var = False
    birincil = None
    for ax, model in zip(axes[0], models):
        series = []
        for role in roles:
            r = stats["results"].get(f"{cfg.category_slug(role)}/{model}", {})
            points = {}
            for anahtar, _ in F21_ROWS:
                c = r.get("contrasts", {}).get(anahtar, {})
                if "metrics" not in c:
                    continue
                birincil = c["primary_metric"]
                m = c["metrics"][birincil]
                dolu = r.get("interpretable") is True
                bos_var |= not dolu
                points[anahtar] = (m["diff"], m["ci"], dolu)
            if points:
                series.append({"label": _short(cfg.category_slug(role)), "color": viz.ROLE_COLOR[role],
                               "marker": ROLE_MARKER[role], "points": points})
        _forest(ax, list(F21_ROWS), series)
        ax.set_title(model, fontsize=11)
        ax.set_xlabel(f"Δ {birincil or 'metric'} (percentage points)")
        viz.legend(ax, len(series), loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2)
    if birincil is None:
        viz.plt.close(fig)
        log.info("F21 atlandi: karsitlik yok")
        return None
    on = "" if stats["meta"].get("reportable") else SMOKE
    _header(fig, f"{on}F21 · Next-purchase accuracy when gifts are removed (C1) or flagged (C3)",
            "Per-user differences averaged over seeds; lines are 95% paired-bootstrap CIs. "
            "An interval covering zero on C1 − C4 reads as “not detectable at this label precision”, "
            "not “no effect”." + _hollow_note(bos_var))
    return viz.save(fig, out, log)


# ------------------------------------------------------------------ F23
def fig_m1(cfg: Config, mm: dict, out: Path) -> Path | None:
    models = list(cfg.get("experiment.models"))
    roles = list(cfg.get("experiment.categories"))
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, axes = viz.plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), sharex=True, squeeze=False)
    bos_var = False
    var = False
    for ax, model in zip(axes[0], models):
        series = []
        for role in roles:
            r = mm["results"].get(f"{cfg.category_slug(role)}/{model}", {})
            points = {}
            for anahtar, _ in F23_ROWS:
                c = r.get("M1_waste_share", {}).get(anahtar, {})
                if "diff" not in c:
                    continue
                dolu = r.get("interpretable") is True
                bos_var |= not dolu
                points[anahtar] = (c["diff"], c["ci"], dolu)
            if points:
                var = True
                series.append({"label": _short(cfg.category_slug(role)), "color": viz.ROLE_COLOR[role],
                               "marker": ROLE_MARKER[role], "points": points})
        _forest(ax, list(F23_ROWS), series)
        ax.set_title(model, fontsize=11)
        ax.set_xlabel(f"Δ share of top-{mm['meta'].get('k')} slots (percentage points)")
        viz.legend(ax, len(series), loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2)
    if not var:
        viz.plt.close(fig)
        log.info("F23 atlandi: M1 yok")
        return None
    on = "" if mm["meta"].get("reportable") else SMOKE
    _header(fig, f"{on}F23 · M1 — recommendation slots spent on sub-categories users only bought as gifts",
            "Users with at least one gift-only sub-category; positive = the left condition wastes "
            "more slots. Lines are 95% paired-bootstrap CIs." + _hollow_note(bos_var))
    return viz.save(fig, out, log)


# ------------------------------------------------------------------ F22
def m2_label(ad: str, m2: dict) -> str:
    """F22 lejanti: yari omur ve bootstrap GA'si. GA'nin ust ucu uyumun lambda sinirina
    dayaniyorsa (ln2 / LAMBDA_MIN) ust uc SINIRSIZDIR: "∞" yazilir ve haftaya cevrilmez ki
    nokta tahmini bir sure olcumu gibi okunmasin. Degerler JSON'dakiyle ayni, hesap yok."""
    hl = m2.get("fit", {}).get("half_life")
    if hl is None:
        return f"{ad} · half-life undefined"
    ci = m2.get("half_life_interactions_ci")
    if ci is not None and ci[1] >= 0.999 * math.log(2) / LAMBDA_MIN:
        return f"{ad} · half-life {hl:.1f} [{ci[0]:.1f}–∞] interactions"
    hafta = m2.get("half_life_weeks")
    ga = f" [{ci[0]:.1f}–{ci[1]:.1f}]" if ci is not None else ""
    return f"{ad} · half-life {hl:.1f}{ga} interactions" + (f" (≈{hafta:.1f} wk)" if hafta else "")


def fig_m2(cfg: Config, mm: dict, out: Path) -> Path | None:
    models = list(cfg.get("experiment.models"))
    roles = list(cfg.get("experiment.categories"))
    n_max = int(cfg.get("marketing.m2_max_self_after_gift"))
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, axes = viz.plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), sharey=True, squeeze=False)
    var = False
    for ax, model in zip(axes[0], models):
        n_seri = 0
        for role in roles:
            m2 = mm["results"].get(f"{cfg.category_slug(role)}/{model}", {}).get("M2_half_life", {})
            kovalar = m2.get("buckets")
            if not kovalar:
                continue
            var = True
            n_seri += 1
            renk, marker = viz.ROLE_COLOR[role], ROLE_MARKER[role]
            xs = [n_max if isinstance(b["n_self_after_gift"], str) else b["n_self_after_gift"] for b in kovalar]
            ys = [None if b["excess"] is None else b["excess"] * 100 for b in kovalar]
            hl = m2.get("fit", {}).get("half_life")
            etiket = m2_label(_short(cfg.category_slug(role)), m2)
            gecerli = [(x, y, b["used_in_fit"]) for x, y, b in zip(xs, ys, kovalar) if y is not None]
            if gecerli:
                # Kullanicisi olmayan kova NaN: cizgi o noktada KIRILIR, bosluk koprulenmez.
                ax.plot(xs, [np.nan if y is None else y for y in ys], color=renk, lw=2, zorder=3, label=etiket)
                for x, y, kullanildi in gecerli:
                    ax.plot([x], [y], linestyle="none", marker=marker, ms=8, color=renk, markeredgewidth=1.6,
                            markerfacecolor=renk if kullanildi else viz.SURFACE, zorder=4)
            if hl is not None:
                uyum_x = [b["n_self_after_gift"] for b in kovalar if b["used_in_fit"]]
                grid = np.linspace(min(uyum_x), max(uyum_x), 100)
                ax.plot(grid, 100 * m2["fit"]["A"] * np.exp(-m2["fit"]["lambda"] * grid), color=renk,
                        lw=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.axhline(0, color=viz.INK_2, lw=1, zorder=1)
        ax.set_xticks(range(n_max + 1), [*map(str, range(n_max)), f"≥{n_max}"])
        ax.set_xlabel("own (self) purchases since the last gift")
        ax.set_title(model, fontsize=11)
        ax.grid(axis="x", visible=False)
        if n_seri:
            # Tek seride de legend: etiket yari omru tasiyor, yalnizca kimligi degil. Eksenin
            # ALTINDA - verinin ustune binmesin.
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2))
    axes[0][0].set_ylabel("excess share of top-K in the gift's\nsub-category, C0 − C1 (pp)")
    if not var:
        viz.plt.close(fig)
        log.info("F22 atlandi: M2 yok")
        return None
    on = "" if mm["meta"].get("reportable") else SMOKE
    _header(fig, f"{on}F22 · M2 — how long a gift keeps pulling recommendations toward its sub-category",
            "Bucket means; dashed = A·exp(−λn) fit on filled buckets (hollow: too few users or the "
            "pooled ≥ bucket; gaps: no users). Weeks use the median gap between reviews, which lag "
            "purchases. BPR ignores order, so no decay is expected there.", top=0.8)
    return viz.save(fig, out, log)


# ------------------------------------------------------------------ F20
def fig_distill(cfg: Config, report: dict, out: Path) -> Path | None:
    rows = [("holdout", "student vs teacher · held-out"),
            ("kcore", "student vs teacher · held-out ∩ 5-core (not gated)"),
            ("human", "student vs human · 500 validation rows")]
    kaynak = {"holdout": report.get("student_vs_teacher_holdout", {}).get("axes", {}),
              "kcore": report.get("student_vs_teacher_holdout_in_kcore", {}).get("axes", {}),
              "human": report.get("student_vs_human", {}).get("axes", {})}
    series = []
    for eksen in ("C1", "C1b"):
        points = {}
        for satir, axes in kaynak.items():
            f1 = axes.get(eksen, {}).get("f1", {})
            if f1.get("value") is not None:
                points[satir] = (f1["value"], f1.get("ci95") or [f1["value"], f1["value"]], True)
        series.append({"label": f"{eksen} axis", "color": AXIS_COLOR[eksen], "marker": AXIS_MARKER[eksen],
                       "points": points})
    if not any(s["points"] for s in series):
        log.info("F20 atlandi: olcum yok")
        return None
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(8.4, 3.8))
    _forest(ax, rows, series, scale=1.0, zero_line=False)   # sifir F1 icin anlamsiz
    kriter = report.get("criteria", {})
    esik = kriter.get("1_c1_axis_vs_teacher", {}).get("threshold")
    if esik is not None:
        ax.plot([esik, esik], [-0.45, 0.45], color=viz.INK, lw=1.2, linestyle=(0, (3, 2)), zorder=5)
        ax.annotate(f"gate {esik:.2f}", xy=(esik, -0.45), xytext=(3, 2), textcoords="offset points",
                    fontsize=8, color=viz.INK_2, va="bottom")
    ins = kriter.get("3_c1_axis_vs_human_not_worse_than_teacher", {})
    if ins.get("teacher_f1") is not None:
        alt = ins["teacher_f1"] - ins["max_drop"]
        ax.plot([alt, alt], [1.55, 2.45], color=viz.INK, lw=1.2, linestyle=(0, (3, 2)), zorder=5)
        ax.annotate(f"teacher C1 − {ins['max_drop']:.2f}", xy=(alt, 1.55), xytext=(3, 2),
                    textcoords="offset points", fontsize=8, color=viz.INK_2, va="bottom")
    los = [p[1][0] for srs in series for p in srs["points"].values()]
    ax.set_xlim(min(0.4, min(los) - 0.05), 1.0)
    for i, (satir, _) in enumerate(rows):
        if not any(satir in srs["points"] for srs in series):
            ax.annotate("not measured", xy=(0.5, i), xycoords=("axes fraction", "data"), ha="center",
                        va="center", fontsize=9, color=viz.MUTED)
    ax.set_xlabel("F1 (95% bootstrap CI)")
    # Eksenin ALTINDA: sol alt kose dusuk F1'li noktalarin ustune biniyordu.
    viz.legend(ax, len(series), loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2)
    meta = report.get("meta", {})
    model_adi = display_model_name(meta.get("model"))
    on = SMOKE if meta.get("backend") == "stub" else ""
    viz.titles(ax, f"{on}F20 · Distilled student vs its teacher and the human reference — {report.get('verdict')}",
               f"{model_adi} · thresholds fixed before training ({meta.get('thresholds_fixed', '')}). "
               "The 5-core row measures the clean → 5-core shift and is not part of the gate.")
    return viz.save(fig, out, log)


# ------------------------------------------------------------------ kosu
def render_all(cfg: Config) -> list[Path]:
    ext = _ext(cfg)
    uretilen: list[Path] = []
    for key in ("base", "fallback"):
        yol = cfg.path("results", f"distill_report_{key}.json")
        if yol.exists():
            p = fig_distill(cfg, read_json(yol), cfg.path("figures", f"F20_distill_fidelity_{key}.{ext}"))
            uretilen += [p] if p else []
    yol = cfg.path("results", "experiment_stats.json")
    if yol.exists():
        p = fig_contrasts(cfg, read_json(yol), cfg.path("figures", f"F21_condition_contrasts.{ext}"))
        uretilen += [p] if p else []
    yol = cfg.path("results", "marketing_metrics.json")
    if yol.exists():
        mm = read_json(yol)
        for fn, ad in ((fig_m2, "F22_m2_half_life"), (fig_m1, "F23_m1_waste_share")):
            p = fn(cfg, mm, cfg.path("figures", f"{ad}.{ext}"))
            uretilen += [p] if p else []
    if not uretilen:
        log.info("uretilecek figur yok - girdi JSON'lari henuz yok")
    return uretilen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args(argv)
    render_all(Config.load(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

