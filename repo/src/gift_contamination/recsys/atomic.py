"""RecBole atomic file uretimi + zaman bazli leave-one-out bolme (Hafta 6).

Girdi : data/interim/<slug>_kcore.parquet  (+ etiketler, Hafta 5'ten)
Cikti : data/processed/recbole/<slug>/<slug>.inter + split.json

IKI SEY C0'DA BIR KEZ DONAR ve butun kosullarda AYNI kalir:

  1. EVREN (kullanici + urun kumesi). CLAUDE.md 5 kural 3. Kosul basina yeniden
     k-core uygulanirsa kosullar kiyaslanamaz hale gelir.
  2. BOLME (hangi etkilesim test, hangisi validasyon). Bu kural dokumanlarda
     ayrica yazmiyordu ama ilkinin dogrudan sonucu: bolmeyi her kosulda yeniden
     hesaplarsak C1'in cikardigi bir satir kullanicinin "son alimi"ni
     degistirebilir ve C0 ile C1 FARKLI test setleri uzerinde karsilastirilir.
     O noktada olculen sey mudahale degil, test setinin degismesi olur.

Bolme kurali: kullanicinin kronolojik sekansinda son etkilesim TEST, sondan
ikinci VALIDASYON, gerisi EGITIM. Rastgele bolme gelecegi egitime sizdirir.

DEGERLENDIRME KISITI: test item'i `self` OLMALI (CLAUDE.md 5 kural 2). Iddia
"hediye alimlari kisinin KENDI tercihine dair tahmini bozuyor"; test item'i da
hediyeyse olculen sey bu degildir. Uygun olmayan kullanicilar degerlendirmeden
cikarilir ama EVRENDE kalir - evren donduruldu.

Kullanim:
    python -m gift_contamination.recsys.atomic --config configs/base.yaml --category mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..data.preprocess import kcore_parquet_path
from ..utils.io import should_skip, write_json
from ..utils.logging import get_logger, log_output

log = get_logger("recsys.atomic")

SELF = "self"
# RecBole atomic file basligi: `alan:tip`. Sekans modelleri zaman damgasini
# float bekliyor.
# `split` de dosyaya YAZILIYOR. RecBole kendi bolmesini yapabilir ama kosullar
# arasi karsilastirma bolmenin C0'DA DONMUS olmasini gerektiriyor; kolon
# olmazsa `conditions` bolmeyi her kosulda yeniden hesaplamak zorunda kalirdi.
# RecBole okumadigi kolonu yok sayar (`load_col`).
INTER_HEADER = [
    "user_id:token", "item_id:token", "timestamp:float",
    "label:token", "split:token",
]

SPLIT_TRAIN = "train"
SPLIT_VALID = "valid"
SPLIT_TEST = "test"


def recbole_dir(cfg: Config, role: str) -> Path:
    return cfg.path("processed", "recbole", cfg.category_slug(role))


def inter_path(cfg: Config, role: str) -> Path:
    slug = cfg.category_slug(role)
    return recbole_dir(cfg, role) / f"{slug}.inter"


def split_path(cfg: Config, role: str) -> Path:
    return recbole_dir(cfg, role) / "split.json"


# --------------------------------------------------------------------- bolme
def assign_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Kullanici basina zaman sirasina gore train / valid / test etiketler.

    Siralama `timestamp` ve esitlik durumunda `row_id` uzerinden - iki
    etkilesim ayni milisaniyeye dustugunde sira KARARLI olsun diye. Kararsiz
    siralama, ayni veriden iki farkli test seti uretir ve kosullar arasi
    karsilastirmayi sessizce bozar.
    """
    ordered = df.sort(["user_id", "timestamp", "row_id"])
    n = pl.len().over("user_id")
    pos = pl.int_range(pl.len()).over("user_id")
    return ordered.with_columns(
        pl.when(pos == n - 1).then(pl.lit(SPLIT_TEST))
        .when(pos == n - 2).then(pl.lit(SPLIT_VALID))
        .otherwise(pl.lit(SPLIT_TRAIN))
        .alias("split")
    )


def eval_eligible(df: pl.DataFrame, label_col: str = "purchase_type") -> pl.Series:
    """Test item'i `self` olan kullanicilar.

    Butun sekansi hediye olan kullanici da buradan dusuyor (CLAUDE.md 9).
    Evrenden DEGIL yalnizca degerlendirmeden cikiyorlar.
    """
    test = df.filter(pl.col("split") == SPLIT_TEST)
    return test.filter(pl.col(label_col) == SELF)["user_id"]


# ------------------------------------------------------------------- uretim
def build_atomic(
    cfg: Config,
    role: str,
    labels: pl.DataFrame | None = None,
    *,
    label_source: str = "proxy",
    force: bool = False,
) -> Path:
    """kcore korpusundan RecBole `.inter` dosyasi + dondurulmus bolme.

    `labels` verilmezse sozcuksel vekil kullanilir ve `label_source` "proxy"
    kalir. O cikti DUMAN TESTI icindir: `conditions.freeze` vekil kaynakli
    girdiden raporlanabilir sonuc uretmeyi reddeder.
    """
    dest = inter_path(cfg, role)
    if should_skip(dest, force, log):
        return dest

    src = kcore_parquet_path(cfg, role)
    if not src.exists():
        raise FileNotFoundError(f"Once preprocess kosun: {src.name} yok")
    df = pl.read_parquet(src).select(
        "row_id", "user_id", "parent_asin", "timestamp"
    )

    if labels is None:
        from ..analysis.keyword_scan import PROXY_COL, keyword_path  # noqa: PLC0415

        kw = keyword_path(cfg, role)
        if not kw.exists():
            raise FileNotFoundError(f"Once keyword_scan kosun: {kw.name} yok")
        labels = (
            pl.read_parquet(kw)
            .select(
                "row_id",
                pl.when(pl.col(PROXY_COL))
                .then(pl.lit("gift_given"))
                .otherwise(pl.lit(SELF))
                .alias("purchase_type"),
            )
        )
        log.warning(
            "etiket verilmedi - SOZCUKSEL VEKIL kullaniliyor. Bu cikti duman "
            "testi icindir, sonuc tablosuna giremez."
        )

    n_before = df.height
    df = df.join(labels.select("row_id", "purchase_type"), on="row_id", how="inner")
    if df.height != n_before:
        raise RuntimeError(
            f"{role}: etkilesim ile etiket satirlari eslesmiyor "
            f"({n_before} -> {df.height}). row_id kaymis olabilir."
        )

    df = assign_splits(df)
    uygun = set(eval_eligible(df).to_list())

    dest.parent.mkdir(parents=True, exist_ok=True)
    (
        df.select(
            pl.col("user_id").alias(INTER_HEADER[0]),
            pl.col("parent_asin").alias(INTER_HEADER[1]),
            pl.col("timestamp").cast(pl.Float64).alias(INTER_HEADER[2]),
            pl.col("purchase_type").alias(INTER_HEADER[3]),
            pl.col("split").alias(INTER_HEADER[4]),
        ).write_csv(dest, separator="\t")
    )
    log_output(log, dest, n_rows=df.height)

    users = df["user_id"].unique()
    items = df["parent_asin"].unique()
    write_json(
        {
            "category": cfg.category_slug(role),
            "label_source": label_source,
            "n_interactions": df.height,
            # C0'DA DONAN EVREN. Sonraki kosullar bunu okur, yeniden hesaplamaz.
            "n_users": users.len(),
            "n_items": items.len(),
            "split_counts": dict(sorted(df.group_by("split").len().iter_rows())),
            "n_eval_eligible_users": len(uygun),
            "eval_eligible_share": round(len(uygun) / users.len(), 4),
            "eval_rule": "test_item_must_be=self (CLAUDE.md 5, kural 2)",
            "order": "timestamp, row_id (kararli)",
            "note": (
                "Bolme ve evren C0'DA DONDU. Kosul basina yeniden hesaplanirsa "
                "C0 ile C1 farkli test setleri uzerinde karsilastirilir."
            ),
        },
        split_path(cfg, role),
        log,
    )
    log.info(
        "%s: %s etkilesim | %s kullanici | %s urun | degerlendirilebilir %.1f%%",
        role, f"{df.height:,}", f"{users.len():,}", f"{items.len():,}",
        100 * len(uygun) / users.len(),
    )
    return dest


def load_atomic(cfg: Config, role: str) -> pl.DataFrame:
    """Uretilmis `.inter` dosyasini geri okur (kolon adlari sadelestirilmis)."""
    path = inter_path(cfg, role)
    if not path.exists():
        raise FileNotFoundError(f"Once atomic dosyayi uretin: {path.name} yok")
    df = pl.read_csv(path, separator="\t")
    return df.rename({old: old.split(":")[0] for old in df.columns})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        build_atomic(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
