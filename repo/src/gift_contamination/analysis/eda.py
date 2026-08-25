"""Data Research teslimi icin betimsel analiz: 4 tablo, 8 figur.

Cikti:
    reports/results/eda_tables.json   -- tum tablo verisi
    reports/results/eda_tables.md     -- markdown olarak render edilmis tablolar
    reports/figures/F*.png            -- figurler

Butun agregasyonlar lazy polars ile yapilir; 16M satirlik kategorilerde bile
bellege yalnizca ozetler alinir (CLAUDE.md bolum 8, kural 10).

Kullanim:
    python -m gift_contamination.analysis.eda --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, add_standard_args
from ..data.preprocess import funnel_path, raw_parquet_path
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from . import viz
from .keyword_scan import PROXY_COL, keyword_path, rates_path

log = get_logger("analysis.eda")


# ------------------------------------------------------------------ yardimcilar
def available_roles(cfg: Config) -> list[str]:
    """Artefakti hazir olan kategoriler, sabit anlati sirasinda."""
    roles = [r for r in viz.ROLE_ORDER if keyword_path(cfg, r).exists()]
    if not roles:
        raise FileNotFoundError(
            "Hicbir kategoride keyword taramasi yok. Once kosun:\n"
            "  python -m gift_contamination.analysis.keyword_scan --category all"
        )
    log.info("analize dahil kategoriler: %s", roles)
    return roles


def label(cfg: Config, role: str) -> str:
    return cfg.category_slug(role).replace("_", " ")


def gini(counts: np.ndarray) -> float:
    """Item populerligi yogunlasmasi. 0 = esit dagilim, 1 = tek item her seyi alir."""
    x = np.sort(counts.astype(float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _md_table(rows: list[dict], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def _pct(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------------- tablolar
def table_funnel(cfg: Config, roles: list[str]) -> list[dict]:
    """T1 -- on isleme hunisi. Her filtrenin kac satir/kullanici/item eledigi."""
    rows = []
    for role in roles:
        f = read_json(funnel_path(cfg, role))
        base = f["01_raw"]["rows"]
        for stage, v in f.items():
            if stage in {"meta", "timestamp"}:
                continue
            rows.append(
                {
                    "category": label(cfg, role),
                    "stage": stage,
                    "rows": f"{v['rows']:,}",
                    "users": f"{v['users']:,}",
                    "items": f"{v['items']:,}",
                    "retained": _pct(v["rows"] / base if base else None, 1),
                }
            )
    return rows


def table_corpus(cfg: Config, roles: list[str]) -> list[dict]:
    """T2 -- korpus profili. k-core'un neden sert eledigini aciklayan sayilar burada."""
    k = int(cfg.get("preprocess.k_core"))
    rows = []
    for role in roles:
        lf = pl.scan_parquet(keyword_path(cfg, role))
        n, n_u, n_i = (
            lf.select(
                pl.len(),
                pl.col("user_id").n_unique(),
                pl.col("parent_asin").n_unique(),
            )
            .collect()
            .row(0)
        )
        u = lf.group_by("user_id").len().collect()["len"].to_numpy()
        i = lf.group_by("parent_asin").len().collect()["len"].to_numpy()
        span = (
            lf.select(
                pl.col("ts").min().alias("first"), pl.col("ts").max().alias("last")
            )
            .collect()
            .row(0)
        )
        med_words = lf.select(pl.col("n_words").median()).collect().item()

        rows.append(
            {
                "category": label(cfg, role),
                "interactions": f"{n:,}",
                "users": f"{n_u:,}",
                "items": f"{n_i:,}",
                "density": f"{n / (n_u * n_i):.2e}",
                "int/user (mean)": f"{u.mean():.2f}",
                "int/user (median)": f"{int(np.median(u))}",
                f"users ≥{k}": _pct(float((u >= k).mean()), 2),
                "int/item (mean)": f"{i.mean():.2f}",
                f"items ≥{k}": _pct(float((i >= k).mean()), 1),
                "item Gini": f"{gini(i):.3f}",
                "median words": f"{int(med_words)}",
                "period": f"{span[0]:%Y-%m} → {span[1]:%Y-%m}",
            }
        )
    return rows


def table_keyword(cfg: Config, roles: list[str]) -> list[dict]:
    """T3 -- vekil hediye oranlari ve naif tek-regex'in sisirme faktoru."""
    rows = []
    for role in roles:
        d = read_json(rates_path(cfg, role))
        o = d["overall"]
        rows.append(
            {
                "category": label(cfg, role),
                "reviews": f"{d['n_reviews']:,}",
                "gift evidence": _pct(o["kw_gift_evidence"]["rate"]),
                "speculative": _pct(o["kw_gift_speculative"]["rate"]),
                "received": _pct(o["kw_gift_received"]["rate"]),
                "**proxy rate**": f"**{_pct(d['proxy_rate'])}**",
                "naive single-regex": _pct(d["naive_any_family_rate"]),
                "inflation": f"{d['inflation_factor']:.2f}×"
                if d["inflation_factor"]
                else "—",
            }
        )
    return rows


# ----------------------------------------------------------------------- figurler
RATING_SCALE = [1.0, 2.0, 3.0, 4.0, 5.0]


def rating_distribution(cfg: Config, role: str) -> tuple[np.ndarray, int]:
    """1-5 izgarasina hizalanmis rating paylari + aralik disi satir sayisi.

    Konuma degil DEGERE baglamak zorunlu: Grocery'de rating=0.0 olan tek bir bozuk
    kayit var ve `arange(len(df))` kullanan bir cizim o kategorinin tum barlarini
    sessizce bir grup kaydiriyordu. Aralik disi satirlar ayrica sayilip raporlanir.
    """
    d = (
        pl.scan_parquet(keyword_path(cfg, role))
        .group_by("rating")
        .agg(pl.len())
        .collect()
    )
    valid = d.filter(pl.col("rating").is_in(RATING_SCALE))
    out_of_range = int(d["len"].sum() - valid["len"].sum())
    counts = {r: 0 for r in RATING_SCALE}
    counts.update(dict(zip(valid["rating"].to_list(), valid["len"].to_list())))
    arr = np.array([counts[r] for r in RATING_SCALE], dtype=float)
    return arr / arr.sum(), out_of_range


def fig_rating_distribution(cfg: Config, roles: list[str], out: Path) -> Path:
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    width = 0.8 / len(roles)
    for j, role in enumerate(roles):
        share, bad = rating_distribution(cfg, role)
        if bad:
            log.warning(
                "%s: rating 1-5 disinda %d satir (grafikten cikarildi)",
                label(cfg, role),
                bad,
            )
        x = np.arange(len(RATING_SCALE)) + (j - (len(roles) - 1) / 2) * width
        ax.bar(x, share, width * 0.92, color=viz.ROLE_COLOR[role], label=label(cfg, role))

    ax.set_xticks(np.arange(len(RATING_SCALE)), ["1", "2", "3", "4", "5"])
    ax.set_xlabel("star rating")
    ax.set_ylabel("share of reviews")
    viz.titles(
        ax,
        "F1 · Rating distribution is severely positive-skewed",
        "Around 60% of reviews carry the top rating. A recommender trained on this "
        "signal sees almost no negative evidence.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


def fig_text_length(cfg: Config, roles: list[str], out: Path) -> Path:
    """Ham parquet uzerinden: min_words filtresinin kestigi kuyruk da gorunsun."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.2))
    grid = np.unique(np.logspace(0, 3, 60).astype(int))
    for role in roles:
        w = (
            pl.scan_parquet(raw_parquet_path(cfg, role))
            .select("n_words")
            .collect()["n_words"]
            .to_numpy()
        )
        ecdf = np.searchsorted(np.sort(w), grid, side="right") / w.size
        ax.plot(grid, ecdf, color=viz.ROLE_COLOR[role], label=label(cfg, role))

    ax.axvline(int(cfg.get("preprocess.min_words")), color=viz.MUTED, lw=1, ls="--")
    ax.annotate(
        f"min_words = {cfg.get('preprocess.min_words')}",
        xy=(int(cfg.get("preprocess.min_words")), 0.9),
        xytext=(8, 0),
        textcoords="offset points",
        color=viz.MUTED,
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_xlim(1, 2500)
    ax.set_xlabel("review length (words, log scale)")
    ax.set_ylabel("cumulative share of reviews")
    viz.titles(
        ax,
        "F2 · Most reviews are short, but the detector has room to work",
        "ECDF on the unfiltered corpus; the dashed line is the quality cut",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    return viz.save(fig, out, log)


def fig_user_activity(cfg: Config, roles: list[str], out: Path) -> Path:
    """F3 -- seyrek profil kaniti. M3 ve k-core tartismasinin dayanagi."""
    k = int(cfg.get("preprocess.k_core"))
    fig, ax = viz.plt.subplots(figsize=(8, 4.2))
    for role in roles:
        u = (
            pl.scan_parquet(keyword_path(cfg, role))
            .group_by("user_id")
            .len()
            .collect()["len"]
            .to_numpy()
        )
        xs = np.arange(1, 51)
        ccdf = [(u >= x).mean() for x in xs]
        ax.plot(xs, ccdf, color=viz.ROLE_COLOR[role], label=label(cfg, role))
        share_k = (u >= k).mean()
        ax.plot([k], [share_k], "o", color=viz.ROLE_COLOR[role], ms=7,
                markeredgecolor=viz.SURFACE, markeredgewidth=2, zorder=5)

    ax.axvline(k, color=viz.MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("interactions per user (log scale)")
    ax.set_ylabel("share of users with at least x (log)")
    viz.titles(
        ax,
        f"F3 · Almost no reviewer reaches the {k}-core threshold",
        f"Survival curve of reviewers by activity. Dashed line is k={k}; "
        f"the marker shows the share of reviewers that clear it.",
    )
    viz.legend(ax, len(roles))
    return viz.save(fig, out, log)


def fig_item_longtail(cfg: Config, roles: list[str], out: Path) -> Path:
    fig, ax = viz.plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1, ls="--", zorder=1)
    for role in roles:
        c = np.sort(
            pl.scan_parquet(keyword_path(cfg, role))
            .group_by("parent_asin")
            .len()
            .collect()["len"]
            .to_numpy()
        )
        cum = np.cumsum(c) / c.sum()
        frac = np.arange(1, c.size + 1) / c.size
        ax.plot(frac, cum, color=viz.ROLE_COLOR[role],
                label=f"{label(cfg, role)} (Gini {gini(c):.2f})")

    ax.set_xlabel("cumulative share of catalogue (least popular first)")
    ax.set_ylabel("cumulative share of interactions")
    viz.titles(
        ax,
        "F4 · Attention concentrates in a small slice of the catalogue",
        "Lorenz curve; the dashed diagonal is perfect equality",
    )
    viz.legend(ax, len(roles), loc="upper left")
    return viz.save(fig, out, log)


def fig_volume_over_time(cfg: Config, roles: list[str], out: Path) -> Path:
    min_year = int(cfg.get("eda.min_year"))
    fig, ax = viz.plt.subplots(figsize=(8, 4.2))
    for role in roles:
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .filter(pl.col("year") >= min_year)
            .group_by("year")
            .agg(pl.len())
            .sort("year")
            .collect()
        )
        ax.plot(d["year"], d["len"], color=viz.ROLE_COLOR[role], label=label(cfg, role))

    ax.set_yscale("log")
    ax.set_xlabel("year")
    ax.set_ylabel("reviews (log scale)")
    viz.titles(
        ax,
        "F5 · The usable corpus is concentrated in recent years",
        "Data ends September 2023, so 2023 is a partial year",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    return viz.save(fig, out, log)


def fig_gift_by_month(cfg: Config, roles: list[str], out: Path) -> Path:
    """F6 -- BASLIK FIGUR. Detektor takvimi gormuyor; tepe yine de cikiyorsa gecerlilik kaniti."""
    fig, ax = viz.plt.subplots(figsize=(8.4, 4.6))
    for role in roles:
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .group_by("month")
            .agg(pl.col(PROXY_COL).mean().alias("rate"))
            .sort("month")
            .collect()
        )
        r = d["rate"].to_numpy() * 100
        ax.plot(d["month"], r, color=viz.ROLE_COLOR[role], marker="o",
                markeredgecolor=viz.SURFACE, markeredgewidth=1.5,
                label=label(cfg, role))
        viz.direct_label(ax, 12, r[-1], label(cfg, role), viz.ROLE_COLOR[role])

    for m in (12, 1):
        ax.axvspan(m - 0.4, m + 0.4, color=viz.GRID, alpha=0.55, zorder=0)
    ax.set_xticks(range(1, 13), viz.MONTHS)
    ax.set_xlim(0.4, 13.6)
    ax.set_xlabel("month the review was written")
    ax.set_ylabel("gift proxy rate (%)")
    viz.titles(
        ax,
        "F6 · The gift signal peaks in December–January",
        "The detector reads text only and has no access to the calendar. Peaks lag "
        "purchases: reviews are written weeks after delivery.",
    )
    # Direct label'lar sagda; legend alt basligi ezmesin diye grafigin altina.
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


def fig_gift_by_category(cfg: Config, roles: list[str], out: Path) -> Path:
    """F7 -- BASLIK FIGUR. Kategori siralamasi + anahtar kelimenin iki yanlis pozitif sinifi."""
    families = ["kw_gift_evidence", "kw_gift_speculative", "kw_gift_received"]
    fig, ax = viz.plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / len(families)
    x = np.arange(len(roles))

    for j, fam in enumerate(families):
        vals = [
            read_json(rates_path(cfg, r))["overall"][fam]["rate"] * 100 for r in roles
        ]
        pos = x + (j - (len(families) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.9, color=viz.FAMILY_COLOR[fam],
                      label=viz.FAMILY_LABEL[fam])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=viz.INK_2, padding=2)

    ax.set_xticks(x, [label(cfg, r) for r in roles])
    ax.set_ylabel("share of reviews (%)")
    viz.titles(
        ax,
        "F7 · Gift language is strongly category-specific",
        "Only the first family is evidence of an actual gift. The other two are small "
        "in volume, but they are error classes a single “gift” regex would absorb "
        "silently — and T4 shows the evidence family itself is only ~58% precise.",
    )
    ax.legend()
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


def fig_rating_gift_vs_self(cfg: Config, roles: list[str], out: Path) -> Path:
    """F8 -- V4 on izlemesi: hediye alimlarinin rating dagilimi farkli mi."""
    fig, axes = viz.plt.subplots(1, len(roles), figsize=(3.2 * len(roles), 3.8),
                                 sharey=True)
    axes = np.atleast_1d(axes)
    for ax, role in zip(axes, roles):
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .filter(pl.col("rating").is_in(RATING_SCALE))
            .group_by(PROXY_COL, "rating")
            .agg(pl.len())
            .collect()
        )
        for is_gift, color, name in ((True, viz.SLOTS[0], "gift proxy"),
                                     (False, viz.SLOTS[1], "rest")):
            sub = d.filter(pl.col(PROXY_COL) == is_gift).sort("rating")
            if sub.height == 0:
                continue
            share = sub["len"].to_numpy() / sub["len"].sum()
            ax.plot(sub["rating"], share, color=color, marker="o",
                    markeredgecolor=viz.SURFACE, markeredgewidth=1.5, label=name)
        ax.set_title(label(cfg, role), fontsize=10, fontweight="normal")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("rating")

    axes[0].set_ylabel("share within group")
    axes[0].legend()
    fig.suptitle("F8 · Gift purchases are rated slightly more generously",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    return viz.save(fig, out, log)


# --------------------------------------------------------------------------- ana
FIGURES = [
    ("F1_rating_distribution", fig_rating_distribution),
    ("F2_text_length", fig_text_length),
    ("F3_user_activity", fig_user_activity),
    ("F4_item_longtail", fig_item_longtail),
    ("F5_volume_over_time", fig_volume_over_time),
    ("F6_gift_rate_by_month", fig_gift_by_month),
    ("F7_gift_rate_by_category", fig_gift_by_category),
    ("F8_rating_gift_vs_self", fig_rating_gift_vs_self),
]


def run(cfg: Config) -> None:
    viz.apply_style(int(cfg.get("eda.figure_dpi")))
    roles = available_roles(cfg)
    ext = cfg.get("eda.figure_format")

    tables = {
        "T1_preprocessing_funnel": table_funnel(cfg, roles),
        "T2_corpus_profile": table_corpus(cfg, roles),
        "T3_keyword_proxy_rates": table_keyword(cfg, roles),
    }

    precision = cfg.path("results", "keyword_precision.json")
    if precision.exists():
        tables["T4_manual_precision"] = read_json(precision)
    else:
        log.warning(
            "T4 atlandi: %s yok. Uretmek icin: "
            "python -m gift_contamination.analysis.precision_check --category all",
            precision.name,
        )

    write_json(tables, cfg.path("results", "eda_tables.json"), log)

    md = ["# EDA tables\n", "_Uretim: `python -m gift_contamination.analysis.eda`_\n"]
    for name, rows in tables.items():
        if not isinstance(rows, list) or not rows:
            continue
        md += [f"\n## {name.replace('_', ' ')}\n", _md_table(rows, list(rows[0])), ""]
    md_path = cfg.path("results", "eda_tables.md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("yazildi: %s", md_path)

    for name, fn in FIGURES:
        fn(cfg, roles, cfg.path("figures", f"{name}.{ext}"))

    _publish(cfg, ext)
    log.info("EDA tamam: %d tablo, %d figur", len(tables), len(FIGURES))


def _publish(cfg: Config, ext: str) -> None:
    """Figurleri teslim klasorune kopyalar; kanonik kaynak reports/figures kalir."""
    import shutil

    target = cfg.get("eda.publish_figures_to", default=None)
    if not target:
        return
    from ..config import REPO_ROOT

    dest = Path(target)
    if not dest.is_absolute():
        dest = (REPO_ROOT / dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name, _ in FIGURES:
        src = cfg.path("figures", f"{name}.{ext}")
        if src.exists():
            shutil.copy2(src, dest / src.name)
    log.info("figurler teslim klasorune kopyalandi: %s", dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser, category=False)
    args = parser.parse_args(argv)
    run(Config.load(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
