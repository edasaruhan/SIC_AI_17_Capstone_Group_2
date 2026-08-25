"""Figur stili ve renk paleti.

Renkler dogrulanmis varsayilan kategorik paletten alindi (slot 1-4, light yuzey).
Slot SIRASI kozmetik degil, renk korlugu guvenligi mekanizmasi: bu sira komsu-cift
listesinde her gecidi geciyor. Hue'lari yeniden sirala ya da beslemeyi dondurerek
5. bir renk uretme.

Iki kural bu dosyanin varlik sebebi:

  1. Renk varliga baglidir, sirasina degil. Bir kategori butun figurlerde AYNI
     rengi tasir; bir figurde kategori sayisi degisince hayatta kalanlar yeniden
     boyanmaz.
  2. Light yuzeyde aqua ve yellow 3:1 kontrastin altinda kaliyor. Rolyef kurali:
     bu figurler ya dogrudan etiket ya da yanlarinda tablo ile yayimlanir.
     Data Research teslimi her figurun yaninda tablosunu tasiyor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------- chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ------------------------------------------------------- kategorik slotlar 1-4
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow

# Kategori -> slot. Hediye yogunlugu spektrumu on siralarda: onde gelen slotlarin
# kontrasti daha iyi ve ana anlati o uc kategoride.
ROLE_COLOR = {
    "high": SLOTS[0],
    "mid": SLOTS[1],
    "low": SLOTS[2],
    "pilot": SLOTS[3],
}

ROLE_ORDER = ["high", "mid", "low", "pilot"]

FAMILY_COLOR = {
    "kw_gift_evidence": SLOTS[0],
    "kw_gift_speculative": SLOTS[1],
    "kw_gift_received": SLOTS[2],
}

FAMILY_LABEL = {
    "kw_gift_evidence": "gift evidence",
    "kw_gift_speculative": "speculative (“would make a great gift”)",
    "kw_gift_received": "received as gift",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def apply_style(dpi: int = 150) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
        }
    )


def titles(ax, title: str, subtitle: str | None = None, *, wrap: int = 95) -> None:
    """Baslik + altina ikincil murekkeple aciklama.

    Ikisini tek yerde ayarlamak, baslik dolgusuyla alt basligin cakismasini onler.
    """
    if subtitle:
        import textwrap

        lines = textwrap.wrap(subtitle, wrap)
        ax.set_title(title, pad=14 + 12 * len(lines))
        ax.annotate(
            "\n".join(lines),
            xy=(0, 1.015),
            xycoords="axes fraction",
            color=MUTED,
            fontsize=9,
            va="bottom",
            linespacing=1.4,
        )
    else:
        ax.set_title(title)


def legend(ax, n_series: int, **kw) -> None:
    """Tek seride legend kutusu yok - basligin kendisi seriyi adlandirir."""
    if n_series >= 2:
        ax.legend(**kw)


def direct_label(ax, x, y, text: str, color: str, **kw) -> None:
    """Seri sonuna dogrudan etiket. Kimlik asla renge tek basina birakilmaz."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
        **kw,
    )


def save(fig, path: Path, logger=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    if logger:
        logger.info("figur: %s", path)
    return path
