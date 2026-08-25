"""Ham review dosyasini filtreleyip 5-core etkilesim tablosuna cevirir.

Iki adim, ikisi de ayri ayri idempotent:

  1. `<slug>_raw.parquet`   -- jsonl'in sadik parquet karsiligi. Filtre YOK; sadece
     ihtiyacimiz olan kolonlar + turetilmis `n_words`. JSON ayristirma pahali oldugu
     icin bir kez yapilir; parametre degisirse 2. adim ucuza yeniden kosar.
  2. `<slug>_clean.parquet` -- huni uygulanmis hali: verified filtresi, min_words,
     dedup, iteratif k-core, kronolojik sekans.

Huni sayaclari `reports/results/preprocess_funnel_<slug>.json` dosyasina yazilir;
Data Research teslimindeki veri kalitesi tablosu dogrudan bundan uretilir.

pandas kullanilmiyor: 16M+ satirlik kategoriler var (CLAUDE.md bolum 8, kural 10).

Kullanim:
    python -m gift_contamination.data.preprocess --config configs/base.yaml --category pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..utils.io import should_skip, write_json
from ..utils.logging import get_logger, log_output
from .download import raw_review_path

log = get_logger("data.preprocess")

# jsonl semasi. Sadece bu alanlar okunur; `images` gibi ic ice alanlar hic ayristirilmaz.
# Alan listesi configs/base.yaml -> dataset.required_fields ile tutarli olmalidir.
NDJSON_SCHEMA: dict[str, object] = {
    "rating": pl.Float64,
    "title": pl.String,
    "text": pl.String,
    "asin": pl.String,
    "parent_asin": pl.String,
    "user_id": pl.String,
    "timestamp": pl.Int64,
    "verified_purchase": pl.Boolean,
    "helpful_vote": pl.Int64,
}

# Veri seti Mayis 1996 - Eylul 2023 arasi (Hou et al., 2024). Disina dusenler sayilir.
VALID_FROM = "1996-01-01"
VALID_TO = "2023-12-31"


# --------------------------------------------------------------------------- yollar
def raw_parquet_path(cfg: Config, role: str) -> Path:
    return cfg.path("interim", f"{cfg.category_slug(role)}_raw.parquet")


def clean_parquet_path(cfg: Config, role: str) -> Path:
    """verified + min_words + dedup. Annotation ve EDA korpusu."""
    return cfg.path("interim", f"{cfg.category_slug(role)}_clean.parquet")


def kcore_parquet_path(cfg: Config, role: str) -> Path:
    """clean + iteratif k-core. RecBole deneyinin korpusu."""
    return cfg.path("interim", f"{cfg.category_slug(role)}_kcore.parquet")


def funnel_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"preprocess_funnel_{cfg.category_slug(role)}.json")


def _density(df: pl.DataFrame, user_col: str, item_col: str, k: int) -> dict:
    """Etkilesim yogunlugu ozeti.

    k-core'un ne kadar sert eleyecegini onceden aciklayan sayilar; Data Research'teki
    korpus profili tablosu (T2) bunlari kullanir.
    """
    if df.height == 0:
        return {}
    u = df.group_by(user_col).len()["len"]
    i = df.group_by(item_col).len()["len"]
    return {
        "interactions_per_user": {
            "mean": round(u.mean(), 3),
            "median": u.median(),
            "max": u.max(),
            f"share_ge_{k}": round((u >= k).sum() / len(u), 6),
        },
        "interactions_per_item": {
            "mean": round(i.mean(), 3),
            "median": i.median(),
            "max": i.max(),
            f"share_ge_{k}": round((i >= k).sum() / len(i), 6),
        },
    }


# ------------------------------------------------------------------ timestamp birimi
def detect_timestamp_unit(path: Path, configured: str) -> str:
    """`auto` ise saniye/milisaniye ayrimini medyandan cikarir.

    Saniye cinsinden 2023 yaklasik 1.7e9, milisaniye cinsinden 1.7e12.
    1e11 esigi ikisini guvenle ayirir.
    """
    if configured in {"s", "ms"}:
        return configured
    med = pl.scan_parquet(path).select(pl.col("timestamp").median()).collect().item()
    unit = "ms" if med and med > 1e11 else "s"
    log.info("timestamp birimi tespit edildi: %s (medyan=%.0f)", unit, med or 0)
    return unit


# ------------------------------------------------------------------------- 1. adim
def build_raw_parquet(cfg: Config, role: str, *, force: bool = False) -> Path:
    """jsonl -> parquet. Filtre uygulamaz; sadece kolon secimi ve turetme yapar."""
    src = raw_review_path(cfg, role)
    dest = raw_parquet_path(cfg, role)
    if should_skip(dest, force, log):
        return dest
    if not src.exists():
        raise FileNotFoundError(
            f"Ham dosya yok: {src}\n"
            "Once indirin: python -m gift_contamination.data.download "
            f"--category {role}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("jsonl -> parquet: %s", src)

    (
        pl.scan_ndjson(
            src,
            schema=NDJSON_SCHEMA,
            row_index_name="row_id",
            low_memory=True,
        )
        .with_columns(
            # \S+ sayimi: bosluga bolmenin aksine cift boslukta yanilmaz.
            pl.col("text").fill_null("").str.count_matches(r"\S+").alias("n_words"),
        )
        .sink_parquet(dest)
    )

    n = pl.scan_parquet(dest).select(pl.len()).collect().item()
    log_output(log, dest, n_rows=n)
    return dest


# ------------------------------------------------------------------------- k-core
def iterative_k_core(
    df: pl.DataFrame, k: int, user_col: str, item_col: str, max_iter: int = 50
) -> tuple[pl.DataFrame, int]:
    """Sabit noktaya kadar k-core.

    Tek gecis yetmez: bir kullaniciyi atmak bir item'i k'nin altina dusurebilir, ve
    tersi de gecerlidir. CLAUDE.md bolum 10'daki `test_preprocess.py` testinin
    "5-core gercekten 5-core mu" sorusu tam olarak bunu dogrular.
    """
    for i in range(1, max_iter + 1):
        before = df.height
        u_ok = df.group_by(user_col).len().filter(pl.col("len") >= k).select(user_col)
        i_ok = df.group_by(item_col).len().filter(pl.col("len") >= k).select(item_col)
        df = df.join(u_ok, on=user_col, how="semi").join(i_ok, on=item_col, how="semi")
        if df.height == before:
            return df, i
        if df.height == 0:
            log.warning("k-core %d. iterasyonda tum satirlari eledi (k=%d)", i, k)
            return df, i
    log.warning("k-core %d iterasyonda yakinsamadi; sonuc tam k-core olmayabilir", max_iter)
    return df, max_iter


def _stage(df: pl.DataFrame, user_col: str, item_col: str) -> dict:
    return {
        "rows": df.height,
        "users": df[user_col].n_unique() if df.height else 0,
        "items": df[item_col].n_unique() if df.height else 0,
    }


# ------------------------------------------------------------------------- 2. adim
def _materialise(
    src: Path, keep: pl.DataFrame, dest: Path, user_col: str, unit: str
) -> dict:
    """Hayatta kalan row_id'leri tam veriye geri baglar, sekans kurar, parquet yazar."""
    (
        pl.scan_parquet(src)
        .join(keep.select("row_id").lazy(), on="row_id", how="semi")
        .with_columns(pl.from_epoch(pl.col("timestamp"), time_unit=unit).alias("ts"))
        .with_columns(
            pl.col("ts").dt.year().alias("year"),
            pl.col("ts").dt.month().alias("month"),
        )
        .sort([user_col, "timestamp"])
        .with_columns(pl.int_range(pl.len()).over(user_col).alias("seq_pos"))
        .sink_parquet(dest)
    )
    stats = (
        pl.scan_parquet(dest)
        .select(
            n=pl.len(),
            ts_min=pl.col("ts").min(),
            ts_max=pl.col("ts").max(),
            out_of_range=(
                (pl.col("ts") < pl.lit(VALID_FROM).str.to_datetime())
                | (pl.col("ts") > pl.lit(VALID_TO).str.to_datetime())
            ).sum(),
        )
        .collect()
        .row(0, named=True)
    )
    log_output(log, dest, n_rows=stats["n"])
    return stats


def build_clean(cfg: Config, role: str, *, force: bool = False) -> tuple[Path, Path]:
    """Iki korpus uretir.

    `<slug>_clean.parquet` -- verified + min_words + dedup. LLM annotation ve
    betimsel analizin (RQ1) korpusu; k-core UYGULANMAZ, cunku hediye orani
    review'lar hakkinda bir soru, recsys alt-grafigi hakkinda degil.

    `<slug>_kcore.parquet` -- clean uzerine iteratif k-core. RecBole deneyinin
    korpusu (CLAUDE.md bolum 5, kural 3: evren burada bir kez donar).

    Ikisini ayirmak zorunlu: Amazon Reviews 2023'un review grafigi cok seyrek ve
    k-core bazi kategorilerde neredeyse her seyi eliyor. Ayrimi yapmazsak betimsel
    analiz de o elemeden zarar gorurdu.
    """
    src = build_raw_parquet(cfg, role, force=force)
    clean_dest = clean_parquet_path(cfg, role)
    kcore_dest = kcore_parquet_path(cfg, role)
    if clean_dest.exists() and kcore_dest.exists() and not force:
        log.info(
            "atlaniyor (ciktilar mevcut, --force ile ezebilirsiniz): %s, %s",
            clean_dest.name,
            kcore_dest.name,
        )
        return clean_dest, kcore_dest

    user_col = "user_id"
    item_col = cfg.get("dataset.join_key")
    k = int(cfg.get("preprocess.k_core"))
    min_words = int(cfg.get("preprocess.min_words"))
    verified_only = bool(cfg.get("preprocess.verified_only"))
    unit = detect_timestamp_unit(src, str(cfg.get("preprocess.timestamp_unit")))

    funnel: dict[str, dict] = {}

    # Huni ve k-core, metin tasimayan ince bir cerceve uzerinde yurur: 16M satirlik
    # bir kategoride `text` kolonunu bellekte tutmak gereksiz ve pahali.
    thin = (
        pl.scan_parquet(src)
        .select("row_id", user_col, item_col, "timestamp", "verified_purchase", "n_words")
        .collect()
    )
    funnel["01_raw"] = _stage(thin, user_col, item_col)

    if verified_only:
        thin = thin.filter(pl.col("verified_purchase"))
    funnel["02_verified_purchase"] = _stage(thin, user_col, item_col)

    thin = thin.filter(pl.col("n_words") >= min_words)
    funnel[f"03_min_words_{min_words}"] = _stage(thin, user_col, item_col)

    # Ayni user+item birden fazla review: en erken timestamp tutulur (CLAUDE.md bolum 9).
    thin = thin.sort("timestamp").unique(
        subset=[user_col, item_col], keep="first", maintain_order=True
    )
    funnel["04_dedup"] = _stage(thin, user_col, item_col) | _density(
        thin, user_col, item_col, k
    )

    # Annotation / betimsel analiz korpusu: k-core UYGULANMADAN yazilir.
    clean_stats = _materialise(src, thin, clean_dest, user_col, unit)

    # Recsys korpusu: uzerine iteratif k-core.
    kcore_thin, n_iter = iterative_k_core(thin, k, user_col, item_col)
    funnel[f"05_k_core_{k}"] = _stage(kcore_thin, user_col, item_col) | {
        "iterations": n_iter
    }
    if kcore_thin.height == 0:
        log.warning(
            "k-core (k=%d) bu kategoride TUM etkilesimleri eledi. Recsys deneyi bu "
            "kategoride kurulamaz; betimsel analiz clean korpusu uzerinden yurur. "
            "Yogunluk sayilari: reports/results/preprocess_funnel_%s.json",
            k,
            cfg.category_slug(role),
        )
    else:
        log.info("k-core %d iterasyonda yakinsadi", n_iter)
    _materialise(src, kcore_thin, kcore_dest, user_col, unit)

    funnel["timestamp"] = {
        "unit": unit,
        "min": str(clean_stats["ts_min"]),
        "max": str(clean_stats["ts_max"]),
        "out_of_range": int(clean_stats["out_of_range"]),
        "valid_window": [VALID_FROM, VALID_TO],
    }
    if clean_stats["out_of_range"]:
        log.warning(
            "%d satir beklenen tarih araliginin disinda", clean_stats["out_of_range"]
        )

    funnel["meta"] = {
        "category_role": role,
        "category": cfg.category_slug(role),
        "source": str(src),
        "outputs": {"clean": str(clean_dest), "kcore": str(kcore_dest)},
        "params": {
            "verified_only": verified_only,
            "min_words": min_words,
            "k_core": k,
            "dedup": True,
            "timestamp_unit": unit,
        },
    }
    write_json(funnel, funnel_path(cfg, role), log)
    return clean_dest, kcore_dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        build_clean(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
