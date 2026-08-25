"""Naif sozcuksel hediye vekili.

Bu bir detektor DEGIL, bir olcu cubugudur. `docs/PROJECT_SPEC.md` bolum 16 bunu
1. hafta karar araci olarak tanimliyor: anahtar kelimenin iki yonde birden
yanildigini SAYIYLA gostermek ve LLM'e gecis gerekcesini kanitlamak.

Uc desen ailesi ayri ayri sayilir:

  gift_evidence     -- gercek hediye alimina isaret eden ifadeler
  gift_speculative  -- "would make a great gift" (yanlis pozitif sinifi)
  gift_received     -- yorumcu hediyeyi almis, vermemis (yine yanlis pozitif)

Vekil oran:  kw_gift_proxy = gift_evidence AND NOT gift_received
Ayrimi yapmadan tek bir "gift" regex'i kosmak, oranı ikiye katlayan ama yanlis
olan bir sayi uretir - Data Research'te gosterilecek olan tam olarak budur.

Kullanim:
    python -m gift_contamination.analysis.keyword_scan --config configs/base.yaml --category pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

from ..config import Config, add_standard_args, resolve_roles
from ..data.preprocess import build_clean, clean_parquet_path
from ..utils.io import should_skip, write_json
from ..utils.logging import get_logger, log_output

log = get_logger("analysis.keyword_scan")

PROXY_COL = "kw_gift_proxy"


def keyword_path(cfg: Config, role: str) -> Path:
    return cfg.path("interim", f"{cfg.category_slug(role)}_keyword.parquet")


def rates_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"keyword_rates_{cfg.category_slug(role)}.json")


# ------------------------------------------------------------------- desenler
def load_patterns(cfg: Config) -> dict[str, str]:
    """Aile adi -> tek bir birlesik, buyuk/kucuk harf duyarsiz regex."""
    path = Path(cfg.get("keywords.path"))
    if not path.is_absolute():
        from ..config import REPO_ROOT

        path = REPO_ROOT / path
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    rel = "|".join(spec["relations"])
    occ = "|".join(spec["occasions"])

    compiled: dict[str, str] = {}
    for family, block in spec["families"].items():
        parts = [
            p.replace("{rel}", rel).replace("{occ}", occ) for p in block["patterns"]
        ]
        compiled[family] = "(?i)" + "|".join(f"(?:{p})" for p in parts)
    log.info("desen aileleri yuklendi: %s (%s)", list(compiled), path.name)
    return compiled


def _validate(patterns: dict[str, str]) -> None:
    """Desenler polars/Rust regex motoru tarafindan kabul ediliyor mu.

    Lookahead gibi desteklenmeyen bir sozdizimi sessizce degil, burada patlasin.
    """
    probe = pl.DataFrame({"t": ["bought this for my daughter"]})
    for family, rx in patterns.items():
        try:
            probe.select(pl.col("t").str.contains(rx))
        except Exception as exc:  # pragma: no cover - config hatasi
            raise ValueError(f"'{family}' ailesindeki regex derlenemedi: {exc}") from exc


# ---------------------------------------------------------------------- tarama
def scan_category(cfg: Config, role: str, *, force: bool = False) -> Path:
    dest = keyword_path(cfg, role)
    if should_skip(dest, force, log):
        return dest

    src = clean_parquet_path(cfg, role)
    if not src.exists():
        build_clean(cfg, role)

    patterns = load_patterns(cfg)
    _validate(patterns)

    # Basligi da tariyoruz: kisa ama sinyal yogun ("Perfect gift!").
    haystack = (
        pl.col("title").fill_null("") + pl.lit(" ") + pl.col("text").fill_null("")
    )

    flags = [
        haystack.str.contains(rx).alias(f"kw_{family}") for family, rx in patterns.items()
    ]

    df = (
        pl.scan_parquet(src)
        .select("row_id", "user_id", "parent_asin", "rating", "n_words", "ts", "year", "month", "title", "text")
        .with_columns(flags)
        .with_columns(
            # Hediye ALMAK vermek degildir; oncelik received'da.
            (pl.col("kw_gift_evidence") & ~pl.col("kw_gift_received")).alias(PROXY_COL)
        )
        .drop("title", "text")  # metin tasinmaz: gizlilik + dosya boyutu
        .collect()
    )

    df.write_parquet(dest)
    log_output(log, dest, n_rows=df.height)
    write_json(summarise(df, cfg, role), rates_path(cfg, role), log)
    return dest


# ----------------------------------------------------------------------- ozet
def summarise(df: pl.DataFrame, cfg: Config, role: str) -> dict:
    flag_cols = [c for c in df.columns if c.startswith("kw_")]
    n = df.height

    overall = {c: {"n": int(df[c].sum()), "rate": round(df[c].mean(), 6)} for c in flag_cols}

    by_month = (
        df.group_by("month")
        .agg(pl.len().alias("n"), pl.col(PROXY_COL).mean().alias("proxy_rate"))
        .sort("month")
        .to_dicts()
    )
    by_year = (
        df.group_by("year")
        .agg(pl.len().alias("n"), pl.col(PROXY_COL).mean().alias("proxy_rate"))
        .sort("year")
        .to_dicts()
    )
    by_rating = (
        df.group_by("rating")
        .agg(pl.len().alias("n"), pl.col(PROXY_COL).mean().alias("proxy_rate"))
        .sort("rating")
        .to_dicts()
    )

    # Anahtar kelimeyi tek bir "gift" regex'i gibi kosarsak ne cikardi:
    naive_any = df.select(
        (
            pl.col("kw_gift_evidence")
            | pl.col("kw_gift_speculative")
            | pl.col("kw_gift_received")
        ).mean()
    ).item()

    return {
        "category_role": role,
        "category": cfg.category_slug(role),
        "n_reviews": n,
        "overall": overall,
        "proxy_rate": round(df[PROXY_COL].mean(), 6),
        "naive_any_family_rate": round(naive_any, 6),
        "inflation_factor": round(naive_any / df[PROXY_COL].mean(), 3)
        if df[PROXY_COL].mean()
        else None,
        "by_month": by_month,
        "by_year": by_year,
        "by_rating": by_rating,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        scan_category(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
