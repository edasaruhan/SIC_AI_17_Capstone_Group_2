"""C0-C4 kosullari (Hafta 6-7). Deneyin gecerliligi bu modulde duruyor.

Kosullar YALNIZCA EGITIM satirlarina dokunur. Test ve validasyon satirlari
C0'da donduruldu ve hicbir kosul onlara dokunmaz - dokunursa C0 ile C1 farkli
test setleri uzerinde karsilastirilir ve olculen sey mudahale olmaktan cikar.

| kod  | ne yapar                                                    |
|------|-------------------------------------------------------------|
| C0   | baseline - butun etkilesimler                                |
| C1   | `gift_given` egitimden cikarilir                             |
| C1b  | `gift_given` + `household` cikarilir  (SAGLAMLIK KONTROLU)   |
| C2   | hediye etkilesimleri loss'ta w ile agirliklandirilir         |
| C3   | hediye bayragi FEATURE olarak eklenir, satir silinmez        |
| C4   | PLASEBO - C1 kadar RASTGELE satir cikarilir                  |

C4 OPSIYONEL DEGIL. C1 kazaniyorsa C4'ten de kazanmak zorunda; yoksa gordugumuz
sey hediye etkisi degil "veri azaldi" etkisidir. Juri'deki ilk akilli kisi bunu
soracak.

C1b NEDEN VAR: `household` korpusun %20,2'si ve projenin kendi kontaminasyon
tanimina ("alan kisi urunu kendisi icin secmedi") gore o da kontaminasyon.
Karar KAVRAMSAL ve ekipte (CLAUDE.md 13). Kod ikisini de kosabilir olsun diye
buradalar; sonuc HER IKI tanim altinda raporlanacak - hangisi iyi cikarsa o
secilmeyecek.

Kullanim:
    python -m gift_contamination.recsys.conditions --config configs/base.yaml \\
           --category mid --condition C1
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger, log_output
from .atomic import (
    SPLIT_TRAIN,
    inter_path,
    load_atomic,
    recbole_dir,
    split_path,
)

log = get_logger("recsys.conditions")

GIFT = "gift_given"
HOUSEHOLD = "household"

# Her kosulun EGITIMDEN cikardigi etiketler. C0/C2/C3 satir silmez.
REMOVED_LABELS: dict[str, tuple[str, ...]] = {
    "C0": (),
    "C1": (GIFT,),
    "C1b": (GIFT, HOUSEHOLD),
    "C2": (),
    "C3": (),
    "C4": (),  # rastgele - etikete BAKMAZ, sayisi C1'den gelir
}
CONDITIONS = tuple(REMOVED_LABELS)

# Vekil etiketten uretilmis cikti raporlanabilir DEGILDIR. `gate1`in backend
# guard'iyla ayni desen: duman testi sonucu sonuc tablosuna giremez.
REPORTABLE_LABEL_SOURCE = "llm"


def condition_path(cfg: Config, role: str, code: str) -> Path:
    return recbole_dir(cfg, role) / f"{code}.inter"


def condition_report_path(cfg: Config, role: str, code: str) -> Path:
    return cfg.path("results", f"condition_{cfg.category_slug(role)}_{code}.json")


def _seed_for(base: int, key: str) -> int:
    """`sampling._stratum_seed` ile ayni desen: crc32, surecler arasi KARARLI.

    `hash()` PYTHONHASHSEED ile degisir ve `seed: 42` ile yeniden
    uretilebilirlik iddiasini yalanlar.
    """
    return (base + zlib.crc32(str(key).encode("utf-8"))) % (2**31 - 1)


def gift_mask(code: str) -> pl.Expr:
    """Kosulun EGITIMDEN cikardigi satirlar. C4 icin anlamsiz (rastgele secer)."""
    labels = REMOVED_LABELS[code]
    if not labels:
        return pl.lit(False)
    return pl.col("label").is_in(list(labels))


def apply_condition(
    df: pl.DataFrame,
    code: str,
    *,
    seed: int,
    weights: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Kosulu uygular; (etkilesimler, rapor) doner.

    `df` C0'in TAM cikitisi olmali - bolme kolonu dahil. Cikarma yalnizca
    `split == train` satirlarina uygulanir.
    """
    if code not in CONDITIONS:
        raise ValueError(f"bilinmeyen kosul: {code}. Gecerli: {', '.join(CONDITIONS)}")

    train = pl.col("split") == SPLIT_TRAIN
    n_train = int(df.filter(train).height)

    if code == "C4":
        # PLASEBO: C1'in cikardigi KADAR satiri rastgele cikar. Sayiyi C1'den
        # turetiyoruz ki "ayni kadar" iddiasi hesaplanmis olsun, elle girilmis
        # degil (denetim bulgusu 7: hesaplanmamis iddia sessizce yalan soyler).
        n_remove = int(df.filter(train & gift_mask("C1")).height)
        indexed = df.with_row_index("_row")
        aday = indexed.filter(train)["_row"]
        drop = (
            aday.sample(n_remove, seed=_seed_for(seed, "C4"), shuffle=True).to_list()
            if n_remove
            else []
        )
        out = indexed.filter(~pl.col("_row").is_in(drop)).drop("_row")
    else:
        out = df.filter(~(train & gift_mask(code)))

    n_removed = df.height - out.height
    report = {
        "condition": code,
        "removed_labels": list(REMOVED_LABELS[code]),
        "n_before": df.height,
        "n_after": out.height,
        "n_removed": n_removed,
        "n_train_before": n_train,
        "removal_share_of_train": round(n_removed / n_train, 4) if n_train else 0.0,
    }

    if code == "C2":
        # Satir silinmiyor; loss agirligi kolonu ekleniyor.
        w = (weights or {}).get("gift", 0.5)
        out = out.with_columns(
            pl.when(pl.col("label").is_in([GIFT, HOUSEHOLD]) & train)
            .then(pl.lit(float(w)))
            .otherwise(pl.lit(1.0))
            .alias("weight")
        )
        report["gift_weight"] = float(w)
    if code == "C3":
        # Bayrak FEATURE olarak giriyor - Wang et al. yaklasiminin testi.
        out = out.with_columns(
            (pl.col("label") == GIFT).cast(pl.Int8).alias("is_gift")
        )

    # --- CLAUDE.md 9: sessizce gecilecek bir detay DEGIL.
    # C0'da var olup bu kosulun EGITIMINDE hic kalmayan urunler eval'de cold
    # item olur ve C1 ile C0 farkinin bir kismini aciklayabilir.
    c0_items = set(df.filter(train)["item_id"].unique().to_list())
    kalan = set(out.filter(train)["item_id"].unique().to_list())
    kaybolan = sorted(c0_items - kalan)
    report["n_items_lost_from_training"] = len(kaybolan)
    report["share_items_lost"] = (
        round(len(kaybolan) / len(c0_items), 4) if c0_items else 0.0
    )
    report["items_lost_sample"] = kaybolan[:20]
    if kaybolan:
        log.warning(
            "%s: %d urun egitimden tamamen kayboldu (%.2f%%) - eval'de cold item",
            code, len(kaybolan), 100 * len(kaybolan) / len(c0_items),
        )
    return out, report


def universe(df: pl.DataFrame) -> tuple[set, set]:
    """Kullanici ve urun evreni. C0'da donar, her kosulda AYNI kalmali."""
    return (
        set(df["user_id"].unique().to_list()),
        set(df["item_id"].unique().to_list()),
    )


def build_condition(
    cfg: Config, role: str, code: str, *, force: bool = False
) -> Path:
    """Kosulun `.inter` dosyasini uretir ve raporunu yazar."""
    dest = condition_path(cfg, role, code)
    if should_skip_condition(dest, force):
        return dest

    meta = read_json(split_path(cfg, role))
    df = load_atomic(cfg, role)
    # `label` kolonu atomic dosyada `label:token` olarak yaziliyor ve
    # `load_atomic` sadelestiriyor.
    out, report = apply_condition(
        df,
        code,
        seed=int(cfg.get("seed")),
        weights={"gift": (cfg.get("experiment.soft_weights") or [0.5])[0]},
    )

    # Evren C0'da DONDU ve RecBole'a oradan verilecek. Burada kilitlenen sey
    # tek yonlu: hicbir kosul C0'da olmayan bir kullanici/urun EKLEYEMEZ.
    # "Evren hic degismedi" demek YANLIS olurdu - C1 bir urunun butun egitim
    # satirlarini silebilir ve o urun ciktida gecmez; bu bir hata degil,
    # yukarida ayrica sayilan cold item durumudur.
    u0, i0 = universe(df)
    u1, i1 = universe(out)
    report["universe_is_subset_of_c0"] = bool(u1 <= u0 and i1 <= i0)
    report["n_users_without_rows"] = len(u0 - u1)
    report["n_items_without_rows"] = len(i0 - i1)
    if u0 - u1:
        # Test satirlarina dokunulmadigi icin bunun ASLA olmamasi gerekiyor.
        raise RuntimeError(
            f"{code}: {len(u0 - u1)} kullanici hic satir birakmadan dustu. "
            "Test/valid satirlari korunuyorsa bu imkansiz - bolme bozulmus olabilir."
        )
    report["label_source"] = meta.get("label_source")
    report["reportable"] = meta.get("label_source") == REPORTABLE_LABEL_SOURCE
    if not report["reportable"]:
        log.warning(
            "label_source=%s - bu cikti DUMAN TESTIDIR ve sonuc tablosuna giremez",
            meta.get("label_source"),
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.rename({c: f"{c}:token" if c in ("user_id", "item_id", "label") else c
                for c in out.columns}).write_csv(dest, separator="\t")
    log_output(log, dest, n_rows=out.height)
    write_json(report, condition_report_path(cfg, role, code), log)
    log.info(
        "%s: %s satir kaldi (%s cikarildi, egitimin %.2f%%'i)",
        code, f"{out.height:,}", f"{report['n_removed']:,}",
        100 * report["removal_share_of_train"],
    )
    return dest


def should_skip_condition(dest: Path, force: bool) -> bool:
    from ..utils.io import should_skip  # noqa: PLC0415

    return should_skip(dest, force, log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument(
        "--condition",
        default="all",
        help=f"Kosul kodu veya 'all'. Gecerli: {', '.join(CONDITIONS)}",
    )
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    codes = (
        list(cfg.get("experiment.conditions"))
        if args.condition == "all"
        else [args.condition]
    )
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        for code in codes:
            build_condition(cfg, role, code, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
