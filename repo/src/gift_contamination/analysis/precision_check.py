"""Anahtar kelime vekilinin elle dogrulanmasi (T4).

Iki adim:

    sample -- katmanli 100 review ceker, elle etiketlenecek bir CSV yazar
    score  -- doldurulmus CSV'yi okur, precision + Wilson %95 GA hesaplar

Uc katman, uc ayri soruyu yanitliyor:

    proxy        vekilin hediye dedigi review'lar   -> PRECISION (asil sayi)
    speculative  "would make a great gift" ailesi   -> yanlis pozitif sinifi gercek mi
    unflagged    hicbir desene takilmayan review'lar -> gozden kacan hediye var mi

GIZLILIK: uretilen CSV birebir review metni tasir ve `data/annotations/human/`
altina yazilir; o dizin .gitignore'dadir. Git'e yalnizca toplulastirilmis sonuc
(reports/results/keyword_precision.json) girer.

Kullanim:
    python -m gift_contamination.analysis.precision_check sample --category all
    python -m gift_contamination.analysis.precision_check score
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from ..config import Config, DEFAULT_CONFIG, REPO_ROOT, resolve_roles
from ..data.preprocess import clean_parquet_path
from ..utils.io import write_json
from ..utils.logging import get_logger, log_output
from .keyword_scan import PROXY_COL, keyword_path

log = get_logger("analysis.precision_check")

SAMPLE_NAME = "keyword_precision_sample.csv"


def _relative_to_repo(path: Path) -> str:
    """Repo koku altindaki yolu goreli, disaridakini dosya adi olarak dondur.

    Bu deger `keyword_precision.json` icine yazilir ve o dosya commit edilir;
    mutlak yol isletim sistemi kullanici adini sizdirir.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


# Katman -> toplam ornekteki pay. proxy agirlikli, cunku asil olculen precision.
STRATA_SHARE = {"proxy": 0.6, "speculative": 0.2, "unflagged": 0.2}

# Etiket sozlugu configs/annotation_schema.json ile ayni: ileride LLM etiketleriyle
# dogrudan karsilastirilabilsin.
LABELS = ("gift_given", "self", "household", "unclear")


def sample_path(cfg: Config) -> Path:
    return cfg.path("human", SAMPLE_NAME)


def result_path(cfg: Config) -> Path:
    return cfg.path("results", "keyword_precision.json")


# ------------------------------------------------------------------ ornekleme
def _strata(lf: pl.LazyFrame) -> dict[str, pl.Expr]:
    return {
        "proxy": pl.col(PROXY_COL),
        "speculative": pl.col("kw_gift_speculative") & ~pl.col(PROXY_COL),
        "unflagged": ~pl.col("kw_gift_evidence")
        & ~pl.col("kw_gift_speculative")
        & ~pl.col("kw_gift_received"),
    }


def build_sample(cfg: Config, roles: list[str]) -> Path:
    n_total = int(cfg.get("keywords.precision_sample_n"))
    seed = int(cfg.get("keywords.precision_seed"))
    per_role = max(1, n_total // len(roles))

    frames = []
    for role in roles:
        flags = pl.scan_parquet(keyword_path(cfg, role)).select(
            "row_id", "kw_gift_evidence", "kw_gift_speculative", "kw_gift_received",
            PROXY_COL,
        )
        text = pl.scan_parquet(clean_parquet_path(cfg, role)).select(
            "row_id", "title", "text"
        )
        joined = flags.join(text, on="row_id", how="inner").collect()

        for stratum, expr in _strata(flags).items():
            pool = joined.filter(expr)
            take = max(1, round(per_role * STRATA_SHARE[stratum]))
            if pool.height == 0:
                log.warning("%s / %s katmani bos", role, stratum)
                continue
            frames.append(
                pool.sample(min(take, pool.height), seed=seed, shuffle=True)
                .with_columns(
                    pl.lit(cfg.category_slug(role)).alias("category"),
                    pl.lit(stratum).alias("stratum"),
                )
            )

    out = (
        pl.concat(frames)
        .sample(fraction=1.0, seed=seed, shuffle=True)  # katmanlari karistir: siralama ipucu vermesin
        .with_row_index("sample_id", offset=1)
        .with_columns(
            pl.lit("").alias("label"),
            pl.lit("").alias("notes"),
        )
        .select(
            "sample_id", "category", "stratum", "row_id", "title", "text",
            "kw_gift_evidence", "kw_gift_speculative", "kw_gift_received",
            "label", "notes",
        )
    )

    dest = sample_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dest)
    log_output(log, dest, n_rows=out.height)
    log.info(
        "Simdi `label` sutununu elle doldurun. Gecerli degerler: %s. "
        "Katman sutununu OKUMADAN etiketleyin - onyargi yaratir.",
        ", ".join(LABELS),
    )
    log.info("GIZLILIK: bu dosya birebir review metni tasir ve git'e GIRMEZ.")
    return dest


# ----------------------------------------------------------------- puanlama
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson %95 guven araligi. Kucuk n ve uc oranlarda normal yaklasimdan iyi."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(cfg: Config) -> dict:
    src = sample_path(cfg)
    if not src.exists():
        raise FileNotFoundError(
            f"Etiketli ornek yok: {src}\n"
            "Once uretin: python -m gift_contamination.analysis.precision_check sample --category all"
        )

    df = pl.read_csv(src, schema_overrides={"label": pl.String}).with_columns(
        pl.col("label").fill_null("").str.strip_chars().str.to_lowercase()
    )
    labelled = df.filter(pl.col("label") != "")
    bad = labelled.filter(~pl.col("label").is_in(LABELS))
    if bad.height:
        raise ValueError(
            f"{bad.height} satirda gecersiz etiket: {sorted(set(bad['label']))}. "
            f"Gecerli degerler: {LABELS}"
        )
    if labelled.height == 0:
        raise ValueError(f"{src} icinde hic etiket yok; `label` sutununu doldurun.")

    out: dict = {
        # Repo-koku goreli: mutlak yol kullanici adini iceriyor ve bu dosya
        # commit ediliyor. Projenin gizlilik taahhudu icin bkz. CLAUDE.md.
        "source": _relative_to_repo(src),
        "n_sampled": df.height,
        "n_labelled": labelled.height,
        "label_vocabulary": list(LABELS),
        "strata": {},
    }

    for stratum in STRATA_SHARE:
        sub = labelled.filter(pl.col("stratum") == stratum)
        if sub.height == 0:
            continue
        k = int((sub["label"] == "gift_given").sum())
        lo, hi = wilson(k, sub.height)
        # "not_self" = gift_given + household. Kontaminasyon acisindan asil anlamli
        # buyukluk bu: ikisinde de urun alicinin kendi tercihini yansitmiyor.
        ns = int(sub["label"].is_in(["gift_given", "household"]).sum())
        ns_lo, ns_hi = wilson(ns, sub.height)
        out["strata"][stratum] = {
            "n": sub.height,
            "gift_given": k,
            "share_gift_given": round(k / sub.height, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "not_self": ns,
            "share_not_self": round(ns / sub.height, 4),
            "ci95_not_self": [round(ns_lo, 4), round(ns_hi, 4)],
            "label_counts": {
                lab: int((sub["label"] == lab).sum()) for lab in LABELS
            },
        }

    p = out["strata"].get("proxy")
    if p:
        out["proxy_precision"] = p["share_gift_given"]
        out["proxy_precision_ci95"] = p["ci95"]
        out["proxy_precision_not_self"] = p["share_not_self"]
        out["proxy_precision_not_self_ci95"] = p["ci95_not_self"]

    write_json(out, result_path(cfg), log)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["sample", "score"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--category", default="all")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.action == "sample":
        roles = [
            r for r in resolve_roles(cfg, args.category) if keyword_path(cfg, r).exists()
        ]
        if not roles:
            raise FileNotFoundError("Once keyword_scan kosun.")
        build_sample(cfg, roles)
    else:
        res = score(cfg)
        log.info(
            "proxy precision: %s (n=%d)",
            res.get("proxy_precision"),
            res["strata"].get("proxy", {}).get("n", 0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
