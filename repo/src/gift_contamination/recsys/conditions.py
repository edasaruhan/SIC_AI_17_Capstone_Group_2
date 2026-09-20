"""Deney kosullari (Hafta 6-7). Deneyin gecerliligi bu modulde duruyor.

Kosullar YALNIZCA EGITIM satirlarina dokunur. Test ve validasyon satirlari
C0'da donduruldu ve hicbir kosul onlara dokunmaz - dokunursa C0 ile C1 farkli
test setleri uzerinde karsilastirilir ve olculen sey mudahale olmaktan cikar.

| kod  | ne yapar                                                          | rol       |
|------|-------------------------------------------------------------------|-----------|
| C0   | baseline - butun etkilesimler                                     | taban     |
| C1   | `gift_given` egitimden cikarilir                                  | birincil  |
| C4   | PLASEBO - C1 kadar RASTGELE egitim satiri cikarilir               | C1'e      |
| C1b  | `gift_given` + `household` + `received` cikarilir                 | saglamlik |
| C4b  | PLASEBO - C1b kadar RASTGELE egitim satiri cikarilir              | C1b'ye    |
| C3   | C1'in satirlari silinmez, GOLGE TOKEN olur (`<urun>::gift`)       | RQ3       |
| C2   | UYGULANMADI - cagrilirsa NotImplementedError                      | -         |

Etiket kumeleri TEK YERDEN: `detection.contamination.CONTAMINATION` (yayginlik
ve dogrulama eksenleri de ayni kumeyi okuyor). DECISIONS 2026-09-14.

C4 OPSIYONEL DEGIL. C1 kazaniyorsa C4'ten de kazanmak zorunda; yoksa gordugumuz
sey hediye etkisi degil "veri azaldi" etkisidir. C1b, C1'in ~iki kati satir
cikardigi icin kendi plasebosu C4b ile karsilastirilir.

C3 NEDEN GOLGE TOKEN. Bayragi feature olarak eklemek hediye urununu egitimde
TAHMIN HEDEFI birakirdi; kirlilik cikis katmanindan oneri listesine geri sizardi.
Golge token hediye olayini sekansta tutar ama gercek urunun ne gommesini ne
hedefini kirletir. Degerlendirmede golge urunlerin skoru -inf yapilir
(`run_experiment`). Bedeli: golge ile gercek urun gommesi bilgi paylasmaz.

C2 NEDEN KILITLI. Onceki kod bir `weight` kolonu ekliyordu ama hicbir sey onu
okumuyordu - kosulsa C0'in aynisi egitilir ve "C2" diye raporlanirdi.

Kullanim:
    python -m gift_contamination.recsys.conditions --config configs/base.yaml \\
           --category mid --condition C1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..config import Config, add_standard_args, resolve_roles
# `detection.schema` DEGIL: bu modul `.venv-recbole` altinda da import ediliyor
# (run_experiment) ve orada pydantic yok.
from ..detection.contamination import CONTAMINATION
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger, log_output
from ..utils.seeding import seed_for
from .atomic import (
    SPLIT_TRAIN,
    load_atomic,
    recbole_dir,
    split_path,
)

log = get_logger("recsys.conditions")

# Her kosulun EGITIMDEN cikardigi etiketler. C0/C3 satir silmez; C4/C4b
# etikete BAKMAZ, sayilarini `PLACEBO_OF`daki kosuldan alir.
REMOVED_LABELS: dict[str, tuple[str, ...]] = {
    "C0": (),
    "C1": CONTAMINATION["narrow"],
    "C4": (),
    "C1b": CONTAMINATION["broad"],
    "C4b": (),
    "C3": (),
}
PLACEBO_OF = {"C4": "C1", "C4b": "C1b"}
# C3: bu etiketlerin EGITIM satirlari golge token olur. C1 ile AYNI kume -
# RQ3 "silmek mi, soylemek mi" karsilastirmasi ayni satirlar uzerinde.
SHADOW_LABELS = {"C3": CONTAMINATION["narrow"]}
SHADOW_SUFFIX = "::gift"
CONDITIONS = tuple(REMOVED_LABELS)
NOT_IMPLEMENTED = {
    "C2": (
        "C2 (loss agirliklandirma) UYGULANMADI. Onceki kod bir `weight` kolonu "
        "ekliyordu ama RecBole onu hic okumuyordu - kosulsa C0'in aynisi 'C2' diye "
        "raporlanirdi. RQ3 C3 ile cevaplaniyor (DECISIONS 2026-09-14)."
    ),
}

# Vekil ya da stub etiketten uretilmis cikti raporlanabilir DEGILDIR. `gate1`in
# backend guard'iyla ayni desen: duman testi sonucu sonuc tablosuna giremez.
REPORTABLE_LABEL_SOURCES = frozenset({"llm", "distilled"})


def condition_path(cfg: Config, role: str, code: str) -> Path:
    return recbole_dir(cfg, role) / f"{code}.inter"


def condition_report_path(cfg: Config, role: str, code: str) -> Path:
    return cfg.path("results", f"condition_{cfg.category_slug(role)}_{code}.json")


def gift_mask(code: str) -> pl.Expr:
    """Kosulun EGITIMDEN cikardigi etiketler. C4/C4b icin anlamsiz (rastgele secer)."""
    labels = REMOVED_LABELS[code]
    if not labels:
        return pl.lit(False)
    return pl.col("label").is_in(list(labels))


def _check_code(code: str) -> None:
    if code in NOT_IMPLEMENTED:
        raise NotImplementedError(NOT_IMPLEMENTED[code])
    if code not in CONDITIONS:
        raise ValueError(f"bilinmeyen kosul: {code}. Gecerli: {', '.join(CONDITIONS)}")


def real_item(expr: pl.Expr) -> pl.Expr:
    """Golge soneki soyulmus urun kimligi - evren karsilastirmasi gercek urun uzerinden."""
    return expr.cast(pl.String).str.strip_suffix(SHADOW_SUFFIX)


def apply_condition(df: pl.DataFrame, code: str, *, seed: int) -> tuple[pl.DataFrame, dict]:
    """Kosulu uygular; (etkilesimler, rapor) doner.

    `df` C0'in TAM cikitisi olmali - bolme kolonu dahil. Degisiklik yalnizca
    `split == train` satirlarina uygulanir.
    """
    _check_code(code)
    train = pl.col("split") == SPLIT_TRAIN
    n_train = int(df.filter(train).height)
    report: dict = {"condition": code, "removed_labels": list(REMOVED_LABELS[code])}

    if code in PLACEBO_OF:
        # PLASEBO: kaynagin cikardigi KADAR egitim satirini rastgele cikar. Sayi
        # kaynaktan TURETILIYOR ki "ayni kadar" iddiasi hesaplanmis olsun, elle
        # girilmis degil (denetim bulgusu 7: hesaplanmamis iddia sessizce yalan soyler).
        kaynak = PLACEBO_OF[code]
        n_remove = int(df.filter(train & gift_mask(kaynak)).height)
        indexed = df.with_row_index("_row")
        aday = indexed.filter(train)["_row"]
        drop = (
            aday.sample(n_remove, seed=seed_for(seed, code), shuffle=True)
            if n_remove
            else pl.Series("_row", [], dtype=aday.dtype)
        )
        out = indexed.join(drop.to_frame(), on="_row", how="anti").sort("_row").drop("_row")
        report["placebo_of"] = kaynak
    elif code in SHADOW_LABELS:
        golge = train & pl.col("label").is_in(list(SHADOW_LABELS[code]))
        out = df.with_columns(
            pl.when(golge)
            .then(pl.col("item_id").cast(pl.String) + pl.lit(SHADOW_SUFFIX))
            .otherwise(pl.col("item_id").cast(pl.String))
            .alias("item_id")
        )
        report["shadow_labels"] = list(SHADOW_LABELS[code])
        report["shadow_suffix"] = SHADOW_SUFFIX
        report["n_shadow_rows"] = int(df.filter(golge).height)
        report["n_shadow_items"] = int(
            out.filter(pl.col("item_id").str.ends_with(SHADOW_SUFFIX))["item_id"].n_unique()
        )
    else:
        out = df.filter(~(train & gift_mask(code)))

    n_removed = df.height - out.height
    report.update({
        "n_before": df.height,
        "n_after": out.height,
        "n_removed": n_removed,
        "n_train_before": n_train,
        "removal_share_of_train": round(n_removed / n_train, 4) if n_train else 0.0,
    })

    # --- CLAUDE.md 9: sessizce gecilecek bir detay DEGIL.
    # C0'da var olup bu kosulun EGITIMINDE GERCEK kimligiyle hic kalmayan urunler
    # eval'de cold item olur ve kosul ile C0 farkinin bir kismini aciklayabilir.
    # C3'te yalnizca golge kimligiyle kalan urun de buraya sayilir: gercek
    # gommesi egitilmiyor.
    c0_items = set(df.filter(train)["item_id"].cast(pl.String).unique().to_list())
    kalan = set(
        out.filter(train & ~pl.col("item_id").cast(pl.String).str.ends_with(SHADOW_SUFFIX))
        ["item_id"].cast(pl.String).unique().to_list()
    )
    kaybolan = sorted(c0_items - kalan)
    report["n_items_lost_from_training"] = len(kaybolan)
    report["share_items_lost"] = (
        round(len(kaybolan) / len(c0_items), 4) if c0_items else 0.0
    )
    report["items_lost_sample"] = kaybolan[:20]
    if kaybolan:
        log.warning(
            "%s: %d urun egitimden gercek kimligiyle kayboldu (%.2f%%) - eval'de cold item",
            code, len(kaybolan), 100 * len(kaybolan) / len(c0_items),
        )
    return out, report


def universe(df: pl.DataFrame) -> tuple[set, set]:
    """Kullanici ve GERCEK urun evreni (golge soneki soyulmus). C0'da donar."""
    return (
        set(df["user_id"].unique().to_list()),
        set(df.select(real_item(pl.col("item_id")))["item_id"].unique().to_list()),
    )


def build_condition(
    cfg: Config, role: str, code: str, *, force: bool = False
) -> Path:
    """Kosulun `.inter` dosyasini uretir ve raporunu yazar."""
    _check_code(code)
    dest = condition_path(cfg, role, code)
    meta = read_json(split_path(cfg, role))
    if dest.exists() and not force:
        # BAYAT KOSUL KORUMASI. Atomic dosya baska bir etiket kaynagiyla yeniden
        # uretildiyse (vekil -> damitilmis) eski kosul dosyasini "zaten var" diye
        # atlamak, deneyi sessizce ESKI etiketlerle kosturur.
        rapor_yolu = condition_report_path(cfg, role, code)
        eski = read_json(rapor_yolu) if rapor_yolu.exists() else {}
        if (eski.get("label_source"), eski.get("n_before")) != (
            meta.get("label_source"), meta.get("n_interactions")
        ):
            raise RuntimeError(
                f"{dest.name} bayat: etiket kaynagi {eski.get('label_source')} / "
                f"{eski.get('n_before')} satir, atomic dosya {meta.get('label_source')} / "
                f"{meta.get('n_interactions')} satir. `--force` ile yeniden uretin."
            )
    if should_skip_condition(dest, force):
        return dest

    df = load_atomic(cfg, role)
    # `label` kolonu atomic dosyada `label:token` olarak yaziliyor ve
    # `load_atomic` sadelestiriyor.
    out, report = apply_condition(df, code, seed=int(cfg.get("seed")))

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
    report["reportable"] = meta.get("label_source") in REPORTABLE_LABEL_SOURCES
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
