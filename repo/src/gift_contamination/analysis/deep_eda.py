"""Derin, projeye ozgu betimsel analiz (T5-T10, F9-F16).

`eda.py` korpusun genel profilini cikarir. Bu modul, projenin sonraki asamalarinin
dayandigi VARSAYIMLARI tek tek olcer. Her analizin karsilik geldigi karar acikca
yazilidir; hicbiri "guzel grafik" diye burada degil.

  T5  Tespit yuzeyi         -> ModernBERT context uzunlugu gerekcesi (Teknoloji Review)
  T6  Alici / vesile        -> annotation semasi ve prompt tasarimi (CLAUDE.md bolum 4)
  T7  Sekans uygunlugu      -> "test item'i self olmali" kuralinin bedeli (bolum 5, kural 2)
  T8  clean vs kcore        -> 5-core deneyin gucunu dusuruyor mu
  T9  Karistiricilar        -> olculen kategori farki gercek mi, uzunluk artefakti mi
  T10 Kategoriler arasi     -> M1 ("yalnizca hediyeyle girilen kategori") hesaplanabilir mi
  T11 Zamansal cozunurluk   -> sekans satin alma sirasi mi, review oturumu mu
  T12 Rating imzasi         -> V4 on izlemesi
  T13 Uzunluga gore tekrar  -> min_words filtresi neyi eliyor
  T14 Mevsimsellik ozeti    -> yaz cukuru ve Aralik/yaz genligi

Kullanim:
    python -m gift_contamination.analysis.deep_eda --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, add_standard_args
from ..data.preprocess import (
    clean_parquet_path,
    kcore_parquet_path,
    raw_parquet_path,
)
from ..utils.io import write_json
from ..utils.logging import get_logger
from . import viz
from .eda import _md_table, _pct, available_roles, label
from .keyword_scan import PROXY_COL, keyword_path

log = get_logger("analysis.deep_eda")

# Ingilizce metinde kabaca 4 karakter = 1 token. Kesin degil; buyuklik mertebesi
# icin yeterli ve tek amaci "kesme kaybi olur mu" sorusunu yanitlamak.
CHARS_PER_TOKEN = 4
TRUNCATION_LIMITS = {"512 tok (BERT)": 512, "1024 tok (config)": 1024}


def _gift(role_path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(role_path).filter(pl.col(PROXY_COL))


# ------------------------------------------------------- T5: tespit yuzeyi
def table_detection_surface(cfg: Config, roles: list[str]) -> list[dict]:
    """Kanit metnin neresinde? Kesme gercekten sorun mu?"""
    rows = []
    for role in roles:
        g = _gift(keyword_path(cfg, role))
        agg = g.select(
            n=pl.len(),
            in_title=pl.col("kw_in_title").mean(),
            pos_median=pl.col("kw_pos_rel").median(),
            pos_p90=pl.col("kw_pos_rel").quantile(0.90),
            has_recipient=pl.col("kw_recipient").is_not_null().mean(),
            has_occasion=pl.col("kw_occasion").is_not_null().mean(),
            **{
                f"lost_{k}": (
                    pl.col("kw_pos_rel") * pl.col("text_len_chars")
                    > v * CHARS_PER_TOKEN
                ).mean()
                for k, v in TRUNCATION_LIMITS.items()
            },
        ).collect().row(0, named=True)

        rows.append(
            {
                "category": label(cfg, role),
                "gift-flagged reviews": f"{agg['n']:,}",
                "evidence in title": _pct(agg["in_title"], 1),
                "median position in body": f"{agg['pos_median']:.3f}",
                "90th pct position": f"{agg['pos_p90']:.3f}",
                "evidence lost @512 tok": _pct(agg["lost_512 tok (BERT)"], 3),
                "evidence lost @1024 tok": _pct(agg["lost_1024 tok (config)"], 3),
                "recipient recoverable": _pct(agg["has_recipient"], 1),
                "occasion recoverable": _pct(agg["has_occasion"], 1),
            }
        )
    return rows


# --------------------------------------------- T6: alici ve vesile bilesimi
def _composition(cfg: Config, roles: list[str], col: str, top: int) -> list[dict]:
    rows = []
    for role in roles:
        d = (
            _gift(keyword_path(cfg, role))
            .filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.len())
            .sort("len", descending=True)
            .head(top)
            .collect()
        )
        total = d["len"].sum()
        for r in d.iter_rows(named=True):
            rows.append(
                {
                    "category": label(cfg, role),
                    col.replace("kw_", ""): r[col],
                    "n": f"{r['len']:,}",
                    "share": _pct(r["len"] / total, 1),
                }
            )
    return rows


def table_recipients(cfg: Config, roles: list[str]) -> list[dict]:
    return _composition(cfg, roles, "kw_recipient", 8)


def table_occasions(cfg: Config, roles: list[str]) -> list[dict]:
    return _composition(cfg, roles, "kw_occasion", 6)


# ------------------------------------------ T7: sekans / deney uygunlugu
def table_sequence_feasibility(cfg: Config, roles: list[str]) -> list[dict]:
    """CLAUDE.md bolum 5 kural 2'nin bedeli: kac kullanici degerlendirmeden duser?

    Deney korpusu kcore uzerinde hesaplanir - degerlendirilecek evren orasi.
    """
    rows = []
    for role in roles:
        kcore = kcore_parquet_path(cfg, role)
        flags = pl.scan_parquet(keyword_path(cfg, role)).select("row_id", PROXY_COL)
        seq = (
            pl.scan_parquet(kcore)
            .select("row_id", "user_id", "timestamp")
            .join(flags, on="row_id", how="inner")
            .collect()
        )
        if seq.height == 0:
            rows.append(
                {
                    "category": label(cfg, role),
                    "users in k-core": "0",
                    "with ≥1 gift": "—",
                    "last item is a gift": "—",
                    "entire sequence is gifts": "—",
                    "**eval-eligible users**": "**n/a — no k-core**",
                    "mean contamination (gift users)": "—",
                }
            )
            continue

        per_user = (
            seq.sort(["user_id", "timestamp"])
            .group_by("user_id", maintain_order=True)
            .agg(
                n=pl.len(),
                gifts=pl.col(PROXY_COL).sum(),
                last_is_gift=pl.col(PROXY_COL).last(),
            )
            .with_columns((pl.col("gifts") / pl.col("n")).alias("frac"))
        )
        n_u = per_user.height
        any_gift = per_user.filter(pl.col("gifts") > 0)
        rows.append(
            {
                "category": label(cfg, role),
                "users in k-core": f"{n_u:,}",
                "with ≥1 gift": f"{any_gift.height:,} ({100 * any_gift.height / n_u:.1f}%)",
                "last item is a gift": _pct(per_user["last_is_gift"].mean(), 2),
                "entire sequence is gifts": _pct(
                    (per_user["frac"] == 1.0).mean(), 3
                ),
                "**eval-eligible users**": f"**{100 * (1 - per_user['last_is_gift'].mean()):.1f}%**",
                "mean contamination (gift users)": _pct(
                    any_gift["frac"].mean(), 1
                )
                if any_gift.height
                else "—",
            }
        )
    return rows


# --------------------------------------- T8: clean vs kcore kontaminasyonu
def table_corpus_comparison(cfg: Config, roles: list[str]) -> list[dict]:
    """5-core filtresi hediye oranini sistematik degistiriyor mu?

    Degistiriyorsa, deney korpusundaki kontaminasyon tam korpustakinden farkli
    demektir ve deneyin bulacagi etki buyuklugu de farkli olur. Sessiz gecilirse
    RQ2'nin cevabi yanlis kalibre edilir.
    """
    rows = []
    for role in roles:
        flags = pl.scan_parquet(keyword_path(cfg, role)).select("row_id", PROXY_COL)
        clean_rate = flags.select(pl.col(PROXY_COL).mean()).collect().item()

        kcore = kcore_parquet_path(cfg, role)
        if not kcore.exists():
            continue
        k = (
            pl.scan_parquet(kcore)
            .select("row_id")
            .join(flags, on="row_id", how="inner")
            .select(n=pl.len(), rate=pl.col(PROXY_COL).mean())
            .collect()
            .row(0, named=True)
        )
        if not k["n"]:
            rows.append(
                {
                    "category": label(cfg, role),
                    "clean corpus": _pct(clean_rate),
                    "k-core corpus": "—",
                    "shift": "k-core bos",
                }
            )
            continue
        rows.append(
            {
                "category": label(cfg, role),
                "clean corpus": _pct(clean_rate),
                "k-core corpus": _pct(k["rate"]),
                "shift": f"{(k['rate'] / clean_rate - 1) * 100:+.1f}%",
                "k-core interactions": f"{k['n']:,}",
            }
        )
    return rows


# ------------------------------------------------- T9: karistirici kontroller
def table_confounds(cfg: Config, roles: list[str]) -> list[dict]:
    """Olculen kategori farki gercek mi, yoksa uzunluk/dogrulama artefakti mi?"""
    rows = []
    for role in roles:
        lf = pl.scan_parquet(keyword_path(cfg, role))
        # Uzunluk kontrolu: en kisa ve en uzun bes dilim
        q = lf.select(
            q20=pl.col("n_words").quantile(0.2), q80=pl.col("n_words").quantile(0.8)
        ).collect().row(0, named=True)
        short = lf.filter(pl.col("n_words") <= q["q20"]).select(
            pl.col(PROXY_COL).mean()
        ).collect().item()
        long = lf.filter(pl.col("n_words") >= q["q80"]).select(
            pl.col(PROXY_COL).mean()
        ).collect().item()

        # Dogrulanmamis alimlar (ham korpusta) ayni orani mi veriyor?
        raw = pl.scan_parquet(raw_parquet_path(cfg, role))
        unverified_share = raw.select(
            (~pl.col("verified_purchase")).mean()
        ).collect().item()

        # Ingilizce disi yazim vekili + kopyala-yapistir vekili.
        # Tekrar orani HEM ham HEM temiz korpusta olculur: ham korpustaki yuksek
        # oran neredeyse tamamen "Great" / "Love it" gibi 1-2 kelimelik jenerik
        # ovgulerden geliyor ve min_words filtresi onu zaten eliyor. Tek bir sayi
        # vermek bu filtrenin ne kadar is yaptigini gizlerdi.
        quality = (
            pl.scan_parquet(raw_parquet_path(cfg, role))
            .select(
                non_ascii=pl.col("text").fill_null("").str.contains(r"[^\x00-\x7F]").mean(),
                dup=(pl.col("text").fill_null("").hash().is_duplicated()).mean(),
            )
            .collect()
            .row(0, named=True)
        )
        dup_clean = (
            pl.scan_parquet(clean_parquet_path(cfg, role))
            .select((pl.col("text").fill_null("").hash().is_duplicated()).mean())
            .collect()
            .item()
        )

        rows.append(
            {
                "category": label(cfg, role),
                "gift rate, shortest 20%": _pct(short),
                "gift rate, longest 20%": _pct(long),
                "length ratio": f"{long / short:.2f}×" if short else "—",
                "unverified share (raw)": _pct(unverified_share, 1),
                "non-ASCII text": _pct(quality["non_ascii"], 2),
                "duplicated text, raw": _pct(quality["dup"], 2),
                "duplicated text, after filters": _pct(dup_clean, 2),
            }
        )
    return rows


# ---------------------------------- T12/T13/T14: teslimi destekleyen sayilar
SUMMER = (6, 7, 8, 9)


def table_seasonality_summary(cfg: Config, roles: list[str]) -> list[dict]:
    """Mevsimsellik ozet sutunlari: yaz cukuru ve Aralik/yaz genligi.

    Teslimin §4.5 tablosu bu iki turetilmis sayiyi aliniyor; elle hesaplanmasin
    diye burada uretiliyor.
    """
    from .keyword_scan import rates_path
    from ..utils.io import read_json

    rows = []
    for role in roles:
        by_month = {
            r["month"]: r["proxy_rate"] * 100
            for r in read_json(rates_path(cfg, role))["by_month"]
        }
        trough = sum(by_month[m] for m in SUMMER) / len(SUMMER)
        rows.append(
            {
                "category": label(cfg, role),
                "Jan": f"{by_month[1]:.2f}%",
                "Feb": f"{by_month[2]:.2f}%",
                "Jun–Sep trough (mean)": f"{trough:.2f}%",
                "Nov": f"{by_month[11]:.2f}%",
                "Dec": f"{by_month[12]:.2f}%",
                "Dec ÷ summer": f"{by_month[12] / trough:.2f}×",
            }
        )
    return rows



def table_gift_rating(cfg: Config, roles: list[str]) -> list[dict]:
    """V4 on izlemesi: hediye alimlarinin rating imzasi.

    Teslimde alintilanan her sayi burada uretilir; hicbiri elle girilmez.
    """
    rows = []
    for role in roles:
        lf = pl.scan_parquet(keyword_path(cfg, role)).filter(
            pl.col("rating").is_in([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        d = (
            lf.group_by(PROXY_COL)
            .agg(
                mean=pl.col("rating").mean(),
                top=(pl.col("rating") == 5.0).mean(),
                n=pl.len(),
            )
            .collect()
        )
        g = d.filter(pl.col(PROXY_COL)).row(0, named=True)
        r = d.filter(~pl.col(PROXY_COL)).row(0, named=True)
        rows.append(
            {
                "category": label(cfg, role),
                "gift mean": f"{g['mean']:.3f}",
                "rest mean": f"{r['mean']:.3f}",
                "difference": f"{g['mean'] - r['mean']:+.3f}",
                "gift 5★": _pct(g["top"], 1),
                "rest 5★": _pct(r["top"], 1),
            }
        )
    return rows


DUP_BUCKETS = [2, 5, 10, 25, 50]
DUP_LABELS = ["1-2", "3-5", "6-10", "11-25", "26-50", "50+"]


def table_duplication_by_length(cfg: Config, roles: list[str]) -> list[dict]:
    """Ham korpustaki yuksek tekrar orani kisa jenerik ovgulerden mi geliyor?

    min_words filtresinin sinyal disinda bir is daha yaptigini gosteren tablo.
    """
    rows = []
    for role in roles:
        d = (
            pl.scan_parquet(raw_parquet_path(cfg, role))
            .select("n_words", pl.col("text").fill_null("").hash().alias("h"))
            .with_columns(pl.col("h").is_duplicated().alias("dup"))
            .group_by(
                pl.col("n_words").cut(DUP_BUCKETS, labels=DUP_LABELS).alias("bucket")
            )
            .agg(n=pl.len(), dup_rate=pl.col("dup").mean())
            .sort("bucket")
            .collect()
        )
        for r in d.iter_rows(named=True):
            rows.append(
                {
                    "category": label(cfg, role),
                    "length (words)": str(r["bucket"]),
                    "reviews": f"{r['n']:,}",
                    "duplicated text": _pct(r["dup_rate"], 1),
                }
            )
    return rows


# -------------------------------------------- T11: zamansal cozunurluk
def table_temporal_resolution(cfg: Config, roles: list[str]) -> list[dict]:
    """Sekans gercekten bir SATIN ALMA sirasi mi, yoksa review yazma oturumu mu?

    Amazon review verisinde insanlar birikmis alimlarini tek oturumda yorumluyor.
    Bu, leave-one-out degerlendirmesini dogrudan etkiliyor: test item'i bir onceki
    etkilesimle AYNI GUNDE ise, model "sonraki alimi" degil "ayni oturumdaki diger
    urunu" tahmin etmeye calisiyor. Bulgunun kendisi bir kusur degil, ama
    raporlanmazsa RQ2'nin olctugu sey yanlis anlasilir.
    """
    rows = []
    for role in roles:
        kcore = kcore_parquet_path(cfg, role)
        d = (
            pl.scan_parquet(kcore).select("user_id", "ts").collect()
            .sort(["user_id", "ts"])
        )
        if d.height == 0:
            continue
        gaps = (
            d.with_columns(
                (pl.col("ts").shift(-1).over("user_id") - pl.col("ts"))
                .dt.total_days().alias("gap")
            )["gap"].drop_nulls()
        )
        last2 = d.group_by("user_id", maintain_order=True).agg(
            last=pl.col("ts").last(), prev=pl.col("ts").gather(-2).first()
        )
        eval_gap = (last2["last"] - last2["prev"]).dt.total_days()

        rows.append(
            {
                "category": label(cfg, role),
                "consecutive pairs on the same day": _pct(float((gaps == 0).mean()), 1),
                "median gap (days)": f"{gaps.median():.0f}",
                "**held-out item same day as previous**": f"**{100 * float((eval_gap == 0).mean()):.1f}%**",
                "median gap before held-out item (days)": f"{eval_gap.median():.0f}",
            }
        )
    return rows


# ------------------------------- T10: kategoriler arasi kullanici ve M1 uygunlugu
def table_cross_category(cfg: Config, roles: list[str]) -> list[dict]:
    """M1 ("yalnizca hediye uzerinden girilen kategori") hesaplanabilir mi?

    Metrik, ayni musteriyi birden fazla kategoride gorebilmeyi gerektiriyor.
    Amazon Reviews 2023'te `user_id` kategoriler arasinda GLOBAL - yani mumkun.
    Burada olculen: kac musteri birden fazla kategoride goruluyor ve bunlarin
    kacinda bir kategoriye giris SADECE hediye alimiyla olmus.
    """
    per_cat = []
    for role in roles:
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .group_by("user_id")
            .agg(n=pl.len(), gifts=pl.col(PROXY_COL).sum())
            .with_columns(
                pl.lit(cfg.category_slug(role)).alias("category"),
                (pl.col("gifts") == pl.col("n")).alias("gift_only_entry"),
            )
            .collect()
        )
        per_cat.append(d)

    allc = pl.concat(per_cat)
    user = allc.group_by("user_id").agg(
        cats=pl.len(),
        gift_only_cats=pl.col("gift_only_entry").sum(),
    )
    multi = user.filter(pl.col("cats") >= 2)

    rows = [
        {
            "metric": "customers observed in ≥2 of the four categories",
            "value": f"{multi.height:,}",
            "of": f"{user.height:,} distinct customers ({100 * multi.height / user.height:.1f}%)",
        },
        {
            "metric": "…of those, entered ≥1 category **only** via a gift",
            "value": f"{multi.filter(pl.col('gift_only_cats') > 0).height:,}",
            "of": _pct(
                multi.filter(pl.col("gift_only_cats") > 0).height / multi.height, 1
            )
            + " of multi-category customers",
        },
        {
            "metric": "customers in all four categories",
            "value": f"{user.filter(pl.col('cats') == 4).height:,}",
            "of": "upper bound on a full cross-category profile",
        },
    ]
    return rows


# ------------------------------------------------------------------ figurler
def fig_evidence_position(cfg: Config, roles: list[str], out: Path) -> Path:
    """F9 -- kanit metnin neresinde geciyor. Context uzunlugu tartismasinin verisi."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    grid = np.linspace(0, 1, 101)
    for role in roles:
        p = (
            _gift(keyword_path(cfg, role))
            .select("kw_pos_rel")
            .drop_nulls()
            .collect()["kw_pos_rel"]
            .to_numpy()
        )
        if p.size == 0:
            continue
        ecdf = np.searchsorted(np.sort(p), grid, side="right") / p.size
        ax.plot(grid, ecdf, color=viz.ROLE_COLOR[role], label=label(cfg, role))

    ax.axhline(0.5, color=viz.MUTED, lw=1, ls="--")
    ax.annotate("half of all evidence", xy=(0.62, 0.5), xytext=(0, 6),
                textcoords="offset points", color=viz.MUTED, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("relative position of the gift evidence within the review body")
    ax.set_ylabel("cumulative share of gift-flagged reviews")
    viz.titles(
        ax,
        "F9 · Recipient disclosure comes early, not late",
        "Reviewers state who the purchase was for in the opening sentence. This is the "
        "opposite of the assumption used to justify a long-context classifier.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    return viz.save(fig, out, log)


def fig_recipient_occasion(cfg: Config, roles: list[str], out: Path) -> Path:
    """F10 -- hediye kime, hangi vesileyle. Annotation semasini dogrudan besler."""
    fig, axes = viz.plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (col, title, top) in zip(
        axes,
        [("kw_recipient", "Recipient", 8), ("kw_occasion", "Occasion", 6)],
    ):
        # Kategoriler arasi toplam siralamayi sabitle, sonra kategori bazli ciz
        pooled = (
            pl.concat(
                [
                    _gift(keyword_path(cfg, r)).filter(pl.col(col).is_not_null())
                    .select(col).collect()
                    for r in roles
                ]
            )
            .group_by(col).agg(pl.len()).sort("len", descending=True).head(top)
        )
        order = pooled[col].to_list()
        y = np.arange(len(order))[::-1]
        h = 0.8 / len(roles)
        for j, role in enumerate(roles):
            d = (
                _gift(keyword_path(cfg, role))
                .filter(pl.col(col).is_in(order))
                .group_by(col).agg(pl.len()).collect()
            )
            tot = d["len"].sum()
            m = dict(zip(d[col].to_list(), d["len"].to_list()))
            vals = [100 * m.get(o, 0) / tot if tot else 0 for o in order]
            ax.barh(y + (j - (len(roles) - 1) / 2) * h, vals, h * 0.9,
                    color=viz.ROLE_COLOR[role], label=label(cfg, role))
        ax.set_yticks(y, order)
        ax.set_xlabel("share of gift-flagged reviews (%)")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="y", visible=False)

    axes[0].legend(loc="lower right")
    fig.suptitle("F10 · Who the gifts are for, and when",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", color=viz.INK)
    fig.tight_layout()
    return viz.save(fig, out, log)


def fig_rate_by_length(cfg: Config, roles: list[str], out: Path) -> Path:
    """F11 -- karistirici kontrolu: uzun review daha cok hediye sinyali mi tasiyor."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    for role in roles:
        lf = pl.scan_parquet(keyword_path(cfg, role))
        d = (
            lf.with_columns(
                # n_words ayrik ve cok baglantili; qcut allow_duplicates ile guvenli.
                pl.col("n_words")
                .qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)
                .cast(pl.String).cast(pl.Int32).alias("decile")
            )
            .group_by("decile")
            .agg(pl.col(PROXY_COL).mean().alias("rate"),
                 pl.col("n_words").median().alias("words"))
            .sort("decile")
            .collect()
        )
        ax.plot(d["decile"] + 1, d["rate"] * 100, color=viz.ROLE_COLOR[role],
                marker="o", markeredgecolor=viz.SURFACE, markeredgewidth=1.5,
                label=label(cfg, role))

    ax.set_xticks(range(1, 11))
    ax.set_xlabel("review-length decile (1 = shortest)")
    ax.set_ylabel("gift proxy rate (%)")
    viz.titles(
        ax,
        "F11 · Longer reviews carry more gift signal — a detection artefact to control for",
        "More text means more opportunity to disclose a recipient. Category differences "
        "must survive this, and §4.6 shows they do: the length gradient is similar "
        "everywhere while the levels are not.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


def fig_user_contamination(cfg: Config, roles: list[str], out: Path) -> Path:
    """F12 -- kullanici profilinin ne kadari kirli. RQ2 ve M3'un buyuklugu."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(0, 1, 11)
    for role in roles:
        kcore = kcore_parquet_path(cfg, role)
        if not kcore.exists():
            continue
        flags = pl.scan_parquet(keyword_path(cfg, role)).select("row_id", PROXY_COL)
        per_user = (
            pl.scan_parquet(kcore).select("row_id", "user_id")
            .join(flags, on="row_id", how="inner")
            .group_by("user_id")
            .agg(frac=pl.col(PROXY_COL).mean())
            .collect()
        )
        if per_user.height == 0:
            continue
        f = per_user.filter(pl.col("frac") > 0)["frac"].to_numpy()
        hist, _ = np.histogram(f, bins=bins)
        ax.plot(bins[:-1] + 0.05, 100 * hist / hist.sum(), color=viz.ROLE_COLOR[role],
                marker="o", markeredgecolor=viz.SURFACE, markeredgewidth=1.5,
                label=label(cfg, role))

    ax.set_xlabel("share of a customer's interactions that are gifts")
    ax.set_ylabel("share of affected customers (%)")
    viz.titles(
        ax,
        "F12 · For most affected customers, gifts are a minority of the profile — "
        "but not a small one",
        "Restricted to k-core customers with at least one gift interaction.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    return viz.save(fig, out, log)


def fig_rate_by_seq_position(cfg: Config, roles: list[str], out: Path) -> Path:
    """F13 -- sekans icinde nerede. 'Test item self olmali' kuralinin dayanagi."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    width = 0.8 / len(roles)
    labels = ["first", "middle", "last"]
    for j, role in enumerate(roles):
        kcore = kcore_parquet_path(cfg, role)
        if not kcore.exists():
            continue
        flags = pl.scan_parquet(keyword_path(cfg, role)).select("row_id", PROXY_COL)
        seq = (
            pl.scan_parquet(kcore).select("row_id", "user_id", "timestamp")
            .join(flags, on="row_id", how="inner")
            .collect()
            .sort(["user_id", "timestamp"])
        )
        if seq.height == 0:
            continue
        seq = seq.with_columns(
            pl.int_range(pl.len()).over("user_id").alias("i"),
            pl.len().over("user_id").alias("n"),
        ).with_columns(
            pl.when(pl.col("i") == 0).then(pl.lit("first"))
            .when(pl.col("i") == pl.col("n") - 1).then(pl.lit("last"))
            .otherwise(pl.lit("middle")).alias("slot")
        )
        d = seq.group_by("slot").agg(pl.col(PROXY_COL).mean().alias("rate"))
        m = dict(zip(d["slot"].to_list(), d["rate"].to_list()))
        x = np.arange(3) + (j - (len(roles) - 1) / 2) * width
        bars = ax.bar(x, [100 * m.get(s, 0) for s in labels], width * 0.9,
                      color=viz.ROLE_COLOR[role], label=label(cfg, role))
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=viz.INK_2, padding=2)

    ax.set_xticks(np.arange(3), ["first interaction", "middle", "last interaction"])
    ax.set_ylabel("gift proxy rate (%)")
    viz.titles(
        ax,
        "F13 · A gift is slightly more often a customer's first recorded interaction",
        "The tilt is modest but consistent across categories: gift buying is "
        "disproportionately an entry point. The right-hand bar is the exclusion cost of "
        "the evaluation rule — customers whose held-out item is itself a gift.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)
    return viz.save(fig, out, log)


def fig_item_concentration(cfg: Config, roles: list[str], out: Path) -> Path:
    """F14 -- hediye bazi item'larda mi yogunlasiyor. M1'in temeli."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    for role in roles:
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .group_by("parent_asin")
            .agg(n=pl.len(), rate=pl.col(PROXY_COL).mean())
            .filter(pl.col("n") >= 20)  # gurultuyu kesmek icin
            .collect()
        )
        if d.height == 0:
            continue
        r = np.sort(d["rate"].to_numpy())
        ax.plot(np.linspace(0, 100, r.size), r * 100, color=viz.ROLE_COLOR[role],
                label=f"{label(cfg, role)} (n={d.height:,})")

    ax.set_xlabel("percentile of catalogue (items with ≥20 reviews)")
    ax.set_ylabel("share of that item's reviews flagged as gifts (%)")
    viz.titles(
        ax,
        "F14 · Gift buying concentrates in identifiable products",
        "Contamination is not spread evenly across the catalogue — the top of each "
        "curve is a targetable set of gift-dominated products.",
    )
    viz.legend(ax, len(roles), loc="upper left")
    return viz.save(fig, out, log)


def fig_time_to_next(cfg: Config, roles: list[str], out: Path) -> Path:
    """F15 -- hediyeden sonraki alima kadar gecen sure. M2'nin on hazirligi."""
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    for role in roles:
        kcore = kcore_parquet_path(cfg, role)
        if not kcore.exists():
            continue
        flags = pl.scan_parquet(keyword_path(cfg, role)).select("row_id", PROXY_COL)
        seq = (
            pl.scan_parquet(kcore).select("row_id", "user_id", "ts")
            .join(flags, on="row_id", how="inner")
            .collect()
            .sort(["user_id", "ts"])
        )
        if seq.height == 0:
            continue
        seq = seq.with_columns(
            (pl.col("ts").shift(-1).over("user_id") - pl.col("ts"))
            .dt.total_days().alias("gap")
        )
        for is_gift, style in ((True, "-"), (False, "--")):
            g = seq.filter(pl.col(PROXY_COL) == is_gift)["gap"].drop_nulls().to_numpy()
            if g.size == 0:
                continue
            grid = np.logspace(0, 3, 60)
            ecdf = np.searchsorted(np.sort(g), grid, side="right") / g.size
            ax.plot(grid, ecdf, style, color=viz.ROLE_COLOR[role], lw=2 if is_gift else 1.2,
                    label=f"{label(cfg, role)} — after a gift" if is_gift else None)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.annotate(
        "everything at x=1 happened within a day —\nbulk reviewing, not a purchase sequence",
        xy=(1, 0.44), xytext=(2.2, 0.16), color=viz.MUTED, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=viz.MUTED, lw=1),
    )
    ax.set_xlabel("days until the customer's next recorded interaction (log scale)")
    ax.set_ylabel("cumulative share")
    viz.titles(
        ax,
        "F15 · How long a contaminated profile stays contaminated",
        "Solid = interval after a gift purchase; dashed = after a self purchase. The gap "
        "distribution sets the timescale over which marketing metric M2 must be measured.",
    )
    viz.legend(ax, len(roles), loc="upper left")
    return viz.save(fig, out, log)


def fig_rate_over_time(cfg: Config, roles: list[str], out: Path) -> Path:
    """F16 -- oran yillar icinde kayiyor mu. Tazelik ve genellenebilirlik icin."""
    min_year = int(cfg.get("eda.min_year"))
    fig, ax = viz.plt.subplots(figsize=(8, 4.4))
    for role in roles:
        d = (
            pl.scan_parquet(keyword_path(cfg, role))
            .filter(pl.col("year") >= min_year)
            .group_by("year")
            .agg(rate=pl.col(PROXY_COL).mean(), n=pl.len())
            .filter(pl.col("n") >= 500)
            .sort("year")
            .collect()
        )
        ax.plot(d["year"], d["rate"] * 100, color=viz.ROLE_COLOR[role],
                label=label(cfg, role))

    ax.set_xlabel("year")
    ax.set_ylabel("gift proxy rate (%)")
    viz.titles(
        ax,
        "F16 · The contamination rate is not stationary",
        "Years with fewer than 500 reviews are omitted. 2023 is a partial year.",
    )
    viz.legend(ax, len(roles), loc="upper center", ncols=min(len(roles), 4),
               bbox_to_anchor=(0.5, -0.16))
    return viz.save(fig, out, log)


# --------------------------------------------------------------------------- ana
FIGURES = [
    ("F9_evidence_position", fig_evidence_position),
    ("F10_recipient_occasion", fig_recipient_occasion),
    ("F11_rate_by_length", fig_rate_by_length),
    ("F12_user_contamination", fig_user_contamination),
    ("F13_rate_by_seq_position", fig_rate_by_seq_position),
    ("F14_item_concentration", fig_item_concentration),
    ("F15_time_to_next", fig_time_to_next),
    ("F16_rate_over_time", fig_rate_over_time),
]

TABLES = [
    ("T5_detection_surface", table_detection_surface),
    ("T6_recipients", table_recipients),
    ("T6b_occasions", table_occasions),
    ("T7_sequence_feasibility", table_sequence_feasibility),
    ("T8_clean_vs_kcore", table_corpus_comparison),
    ("T9_confounds", table_confounds),
    ("T10_cross_category", table_cross_category),
    ("T11_temporal_resolution", table_temporal_resolution),
    ("T12_gift_rating_signature", table_gift_rating),
    ("T13_duplication_by_length", table_duplication_by_length),
    ("T14_seasonality_summary", table_seasonality_summary),
]


def run(cfg: Config) -> None:
    viz.apply_style(int(cfg.get("eda.figure_dpi")))
    roles = available_roles(cfg)
    ext = cfg.get("eda.figure_format")

    tables = {name: fn(cfg, roles) for name, fn in TABLES}
    write_json(tables, cfg.path("results", "deep_eda_tables.json"), log)

    md = ["# Deep EDA tables\n",
          "_Uretim: `python -m gift_contamination.analysis.deep_eda`_\n"]
    for name, rows in tables.items():
        if rows:
            md += [f"\n## {name.replace('_', ' ')}\n", _md_table(rows, list(rows[0])), ""]
    path = cfg.path("results", "deep_eda_tables.md")
    path.write_text("\n".join(md), encoding="utf-8")
    log.info("yazildi: %s", path)

    for name, fn in FIGURES:
        fn(cfg, roles, cfg.path("figures", f"{name}.{ext}"))

    from .eda import _publish

    _publish(cfg, ext, FIGURES)
    log.info("derin EDA tamam: %d tablo, %d figur", len(tables), len(FIGURES))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser, category=False)
    args = parser.parse_args(argv)
    run(Config.load(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
