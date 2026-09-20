"""Tek sayfalik ozet (A4 portre PDF) ureten betik.

    python build_one_pager.py            # -> one_pager.pdf + one_pager.png

Deck'le AYNI kaynaktan beslenir: `build_deck.load()` sonuc JSON'larini okur, bu
dosya yalnizca yerlesimi cizer. Bir artefakt degisirse ikisi birden degisir.

matplotlib ile ciziliyor - proje zaten matplotlib'e bagli, yeni bir bagimlilik
eklenmiyor. PNG yalnizca gozle kontrol icin; jurinin aldigi dosya PDF.

NOT: matplotlib metni KENDILIGINDEN sarmaz. Kutuya giren her govde metni
`_w()` ile elle sariliyor; sutun genisligi degisirse oradaki karakter sayisi da
degismeli, yoksa metin sayfanin disina tasar (olculdu).
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from build_deck import (GITHUB, TEAM, ci, load, pct, pts, thousands,
                        verdict_en)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "repo" / "reports" / "results"
A4 = (8.27, 11.69)

INK = "#1C1C1C"
INK2 = "#555555"
MUTED = "#8E8E8E"
RULE = "#D9D9D9"
ACCENT = "#1F5C8B"
GREY = "#9AA7B4"
SOFT = "#E8EFF6"
SOFT2 = "#F3F6FA"
CAUTION = "#8A520A"
CAUTION_SOFT = "#FBF2E3"
FONT = "DejaVu Sans"

L, R = 0.075, 0.925
W = R - L
GAP = 0.022
BW = (W - GAP) / 2

COL_CHARS = 62        # yarim sutuna sigan karakter (6.6 pt)
FULL_CHARS = 132      # tam genislige sigan karakter (6.6 pt)


def _w(text: str, width: int = COL_CHARS) -> str:
    """Paragrafi elle sar; satir sonlarini koru."""
    return "\n".join(
        "\n".join(textwrap.wrap(p, width)) if p.strip() else ""
        for p in text.split("\n")
    )


def _box(fig, x, y, w, h, fill, edge=RULE, lw=0.8, radius=0.008):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=0))


def _t(fig, x, y, text, size=9, color=INK, weight="normal", ha="left",
       va="top", style="normal", spacing=1.25):
    return fig.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                    ha=ha, va=va, style=style, family=FONT, linespacing=spacing,
                    transform=fig.transFigure, zorder=3)


def _rule(fig, y, x0=L, x1=R, color=RULE, lw=0.9):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, linewidth=lw,
                              transform=fig.transFigure, zorder=1))


# ------------------------------------------------------------------ figur
def _plot_half_life(ax, cell_colors) -> None:
    """F22'nin tek panelli, A4'e sigan hali - AYNI JSON, ayni sayilar.

    Deck commit'li PNG'yi oldugu gibi gomuyor; burada yer kisitindan oturu
    yalnizca sirali model yeniden ciziliyor. Cizilen degerler
    `marketing_metrics.json`daki kova ortalamalarinin ta kendisi.
    """
    mk = json.loads((RESULTS / "marketing_metrics.json").read_text(encoding="utf-8"))
    for hucre, renk, isim in cell_colors:
        h = mk["results"][hucre]["M2_half_life"]
        kovalar = [b for b in h["buckets"] if b.get("used_in_fit")]
        x = [b["n_self_after_gift"] for b in kovalar]
        y = [100 * b["excess"] for b in kovalar]
        ax.plot(x, y, marker="o", markersize=3.2, linewidth=1.5, color=renk,
                label=f"{isim} · half-life {h['fit']['half_life']:.2f} self-purchases",
                zorder=3)
        fit = h["fit"]
        if fit.get("A") is not None and fit.get("lambda") is not None:
            xs = [i / 4 for i in range(0, 4 * max(x) + 1)]
            ax.plot(xs, [100 * fit["A"] * math.exp(-fit["lambda"] * v) for v in xs],
                    linestyle="--", linewidth=1.0, color=renk, alpha=0.8, zorder=2)

    ax.axhline(0, color="#BFBFBF", linewidth=0.9, zorder=1)
    ax.set_xlabel("own (self) purchases since the last gift", fontsize=6.8,
                  color=INK2, family=FONT, labelpad=2)
    ax.set_ylabel("excess top-10 share for the gift's\nsubcategory (pts), C0 − C1",
                  fontsize=6.8, color=INK2, family=FONT, labelpad=3)
    ax.tick_params(labelsize=6.3, colors=INK2, length=2, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=6.2, frameon=False, loc="upper right", handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK2)
        t.set_family(FONT)


def num6(x: float | None) -> str:
    return "—" if x is None else f"{x:+.6f}"


# ------------------------------------------------------------------ sayfa
def build(out_pdf: Path, out_png: Path) -> tuple[Path, Path]:
    N = load()
    t2 = N["m2"]["Toys_and_Games/SASRec"]
    m1b = N["m1"]["Toys_and_Games/BPR"]["c0_c1"]
    tg = N["prevalence"]["Toys_and_Games"]
    c14s = N["c1_c4"]["Toys_and_Games/SASRec"]
    c14b = N["c1_c4"]["Toys_and_Games/BPR"]
    c31 = N["c3_c1"]["Toys_and_Games/SASRec"]
    oranlar = [N["gate1"][k]["ratio"] for k in N["gate1"]]

    fig = plt.figure(figsize=A4, dpi=200)
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------ baslik
    fig.patches.append(Rectangle((0, 0.986), 1, 0.014, transform=fig.transFigure,
                                 facecolor=ACCENT, edgecolor="none", zorder=2))
    _t(fig, L, 0.968, "CAPSTONE SUMMARY · ONE PAGE", size=7.5, color=ACCENT,
       weight="bold")
    _t(fig, L, 0.958, "This Was Not For Me", size=26, weight="bold")
    _t(fig, L, 0.920,
       "Detecting gift purchases as a distinct class of noise in e-commerce\n"
       "recommender systems", size=10.5, color=INK2)
    _rule(fig, 0.880)
    _t(fig, L, 0.872, TEAM, size=7.5, weight="bold")
    _t(fig, L, 0.860, f"{GITHUB}  ·  20 September 2026  ·  code version "
                      f"{N['provenance']['experiment_stats.json']}",
       size=7, color=MUTED)

    # -------------------------------------------------- tek cumlelik bulgu
    _box(fig, L, 0.786, W, 0.060, SOFT, edge=ACCENT, lw=1.0)
    _t(fig, L + 0.016, 0.836,
       "Holding the amount of deleted data fixed, removing gift rows leaves a model that "
       "predicts\nthe user's own next purchase better than removing the same number of "
       "random rows —\nand the excess it opens in the recommendation list halves after "
       "about one self-purchase.",
       size=9, weight="bold", color=ACCENT, spacing=1.35)

    # --------------------------------------------------------- RQ kutulari
    _t(fig, L, 0.780, "THE FOUR QUESTIONS", size=7.5, color=ACCENT, weight="bold")
    bh = 0.096
    kutular = [
        ("RQ1 · how common are gifts?", f"{tg['cal']:.0f} %",
         f"of Toys and Games reviews, after human calibration. 95 % CI "
         f"[{tg['ci'][0]:.1f}, {tg['ci'][1]:.1f}]; raw LLM rate {tg['raw']:.1f} %. "
         f"Grocery, the control category, sits at "
         f"{N['prevalence']['Grocery_and_Gourmet_Food']['cal']:.1f} %. All four "
         f"categories peak in December–January ({min(oranlar):.2f}–{max(oranlar):.2f}×)."),
        ("RQ2 · does removing gifts help?",
         f"{pct(c14s['rel'], 1, sign=True)} / {pct(c14b['rel'], 1, sign=True)}",
         f"Recall@10 for C1 − C4 (against the placebo), SASRec / BPR on Toys: "
         f"{num6(c14s['diff'])} {ci(c14s['ci'])} and {num6(c14b['diff'])} "
         f"{ci(c14b['ci'])}. On the low-gift control it is "
         f"{pct(N['c1_c4']['Grocery_and_Gourmet_Food/SASRec']['rel'], 1, sign=True)} / "
         f"{verdict_en(N['c1_c4']['Grocery_and_Gourmet_Food/BPR']['verdict'])} — the "
         f"dose–response holds in all four comparisons."),
        ("RQ3 · delete them, or flag them?", "delete",
         f"The shadow-token method (C3) was worse than deleting in three of four cells "
         f"and worse than doing nothing in three of four. Toys · SASRec, C3 − C1 = "
         f"{num6(c31['diff'])} {ci(c31['ci'])} → {verdict_en(c31['verdict'])}. Other "
         f"flagging designs were not tested."),
        ("RQ4 · how long does it last?", f"{t2['half_life']:.2f}",
         f"self-purchases until the excess halves, in the sequential model on Toys. "
         f"95 % CI {ci(t2['ci'], 2)}; ≈{t2['weeks']:.1f} weeks (approximate). The gift "
         f"opens {pts(t2['start_excess'], 1)} of the top-10 list, and "
         f"{pts(m1b['diff'], 1)} of slots go to gift-only subcategories in the "
         f"classical model."),
    ]
    ust = 0.764
    for i, (bas, buyuk, govde) in enumerate(kutular):
        x = L + (i % 2) * (BW + GAP)
        y = ust - (i // 2) * (bh + 0.014)
        _box(fig, x, y - bh, BW, bh, SOFT2)
        _t(fig, x + 0.014, y - 0.008, bas, size=8, color=ACCENT, weight="bold")
        _t(fig, x + 0.014, y - 0.023, buyuk, size=16, weight="bold", color=INK)
        _t(fig, x + 0.014, y - 0.049, _w(govde), size=6.6, color=INK2, spacing=1.35)

    # --------------------------------------------------------------- figur
    y1 = ust - 2 * (bh + 0.014) - 0.010
    _t(fig, L, y1, "HOW LONG A GIFT KEEPS PULLING THE LIST · F22, SEQUENTIAL MODEL",
       size=7.5, color=ACCENT, weight="bold")
    ax = fig.add_axes([L + 0.045, y1 - 0.190, W - 0.05, 0.160])
    _plot_half_life(ax, [("Toys_and_Games/SASRec", ACCENT, "Toys and Games"),
                         ("Grocery_and_Gourmet_Food/SASRec", GREY,
                          "Grocery and Gourmet Food (control)")])

    # ------------------------------------------------------ iddia / degil
    y2 = y1 - 0.218
    ch = 0.136
    iddia = [
        "Gift purchases are measurable at scale, and common.",
        "Gift rows carry less usable preference signal than random rows, with a "
        "dose–response across two categories.",
        "The excess a gift opens decays with the user's own later purchases.",
        "The one flagging method we tested is worse than deleting.",
    ]
    degil = [
        f"That the detector is accurate: macro-F1 {N['human']['macro_f1']:.4f} "
        f"{ci(N['human']['macro_f1_ci'], 4)} against ONE annotator, and both "
        f"pre-registered gates (≥ 0.75, ≥ 0.80) were missed.",
        "That deleting gifts is a production recommendation — the net effect against "
        "doing nothing is model-dependent.",
        "That this generalises beyond two Amazon categories and two models.",
        "That wasted slots are lost revenue; conversion was never measured.",
    ]
    for j, (bas, maddeler, zemin, renk) in enumerate(
            [("WHAT WE CLAIM", iddia, SOFT, ACCENT),
             ("WHAT WE DO NOT CLAIM", degil, CAUTION_SOFT, CAUTION)]):
        x = L + j * (BW + GAP)
        _box(fig, x, y2 - ch, BW, ch, zemin)
        _t(fig, x + 0.014, y2 - 0.008, bas, size=7.5, color=renk, weight="bold")
        yy = y2 - 0.024
        for s in maddeler:
            sarili = _w(s, COL_CHARS - 3)
            _t(fig, x + 0.014, yy, "•  " + sarili.replace("\n", "\n    "),
               size=6.6, color=INK, spacing=1.35)
            yy -= 0.0105 * (1 + sarili.count("\n")) + 0.0042

    # -------------------------------------------------------------- yontem
    y3 = y2 - ch - 0.020
    _t(fig, L, y3, "HOW IT WAS MEASURED", size=7.5, color=ACCENT, weight="bold")
    yontem = (
        f"{N['teacher']['model'].split('/')[-1]} (prompt {N['teacher']['prompt']}, "
        f"{N['teacher']['backend']} on 2×T4) labelled {thousands(N['n_annotated'])} "
        f"reviews into a five-class purchase-type schema; one human annotator "
        f"independently labelled {N['human']['n_rows']} of them; a distilled "
        f"{N['distill']['model'].split('/')[-1]} student (fidelity gate "
        f"{N['distill']['verdict']}, 3/3) then labelled {thousands(N['n_labelled'])} "
        f"interactions across the two experiment categories. Six training-set "
        f"conditions — including two placebos that delete the same NUMBER of random "
        f"rows — were run through RecBole with SASRec and BPR over "
        f"{len(N['seeds'])} seeds, for {N['n_runs']} runs. Gate 1 (may the detector "
        f"be used?) passed 4/4 in all four categories; Gate 2 (may the contrasts be "
        f"interpreted?) passed in all four category × model cells. Every contrast is a "
        f"paired bootstrap over per-user Recall@10, and the interpretation rule — "
        f"lower bound / not detectable / negative — was written down before any result "
        f"was seen."
    )
    sarili = _w(yontem, FULL_CHARS)
    yh = 0.018 + 0.0105 * (1 + sarili.count("\n"))
    _box(fig, L, y3 - 0.010 - yh, W, yh, SOFT2)
    _t(fig, L + 0.014, y3 - 0.018, sarili, size=6.6, color=INK2, spacing=1.35)

    # -------------------------------------------------------------- altlik
    _rule(fig, 0.046)
    _t(fig, L, 0.038,
       "Every number above is read from a JSON artefact committed in "
       "repo/reports/results/ — reproduce them all with  python repo/scripts/demo.py",
       size=6.8, color=MUTED)
    _t(fig, L, 0.027,
       "Full report with every interval and every limitation: final-report/final-report.md"
       "   ·   Slides: presentation/SIC_AI_17_Group_2.pptx",
       size=6.8, color=MUTED)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", facecolor="white")
    fig.savefig(out_png, format="png", facecolor="white", dpi=150)
    plt.close(fig)
    return out_pdf, out_png


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "one_pager.pdf")
    args = ap.parse_args(argv)
    pdf, png = build(args.out, args.out.with_suffix(".png"))
    print(f"{pdf.name} ({pdf.stat().st_size / 1024:.0f} KB) ve {png.name} yazildi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
