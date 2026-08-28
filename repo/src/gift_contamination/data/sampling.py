"""Katmanli annotation ornekleme (Hafta 2).

LLM'in etiketleyecegi ornegi ceker. UC AYRI CERCEVE uretir ve bu ayrim projenin
istatistiksel olarak en kolay bozulan ozelligi:

    main            -- month x rating x text_length_bucket hucrelerine ORANTILI
                       tahsisle cekilmis katmanli rastgele ornek. Orantili oldugu
                       icin KENDINDEN AGIRLIKLI: yaygınlık duz ortalamayla hesaplanir.
    boost           -- yalnizca `kw_gift_proxy` havuzundan cekilmis EK ornek.
                       Distillation egitim setine daha cok pozitif koymak icin.
    boost_received  -- `kw_gift_received` havuzundan ek ornek. Vekilin tanim geregi
                       disladigi (PROXY = evidence AND NOT received) ama LLM'in
                       ayirmasi gereken en zor negatif sinif; korpusta %0.08-0.28.

Yaygınlık tahmini YALNIZCA `main` uzerinden hesaplanir. Takviye cercevelerini
`main`'e katip oran hesaplamak, kendi sectiginiz seyi olcmek demektir. `configs/base.yaml` icindeki "ayri tutulur, ana orana KATILMAZ"
notu tam olarak bunu soyluyor; `sample_frame` kolonu o notu makine tarafindan
uygulanabilir hale getiriyor. Iki cerceve AYRIKTIR - bir satir ikisinde birden
gecmez.

Kategori basina esit tahsis: RQ1 kategorileri karsilastirdigi icin her kategori
esit kesinlikte tahmin almali. Havuzlanmis bir oran raporlanmiyor, dolayisiyla
havuzlama agirligi da gerekmiyor.

Kullanim:
    python -m gift_contamination.data.sampling --config configs/base.yaml --category all
    python -m gift_contamination.data.sampling --trial 200 --category all
    python -m gift_contamination.data.sampling --trial-source 200
"""

from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Hashable

import polars as pl

from ..analysis.keyword_scan import PROXY_COL, keyword_path
from ..config import Config, add_standard_args, resolve_roles
from ..data.download import raw_meta_path
from ..data.metadata import build_meta_parquet, load_join_frame, meta_parquet_path
from ..data.preprocess import build_clean, clean_parquet_path
from ..utils.io import should_skip, write_json
from ..utils.logging import get_logger, log_output

log = get_logger("data.sampling")

FRAME_MAIN = "main"
FRAME_BOOST = "boost"
FRAME_RECEIVED = "boost_received"

# Yalnizca FRAME_MAIN yaygınlık hesabina girer. Digerleri kasitli olarak
# carpitilmis havuzlardan cekilir.
SUPPLEMENTARY_FRAMES = (FRAME_BOOST, FRAME_RECEIVED)

# `_stratum_id()` bu uc boyutu SABIT kuruyor. Config'teki `sampling.strata`
# listesi ile ayrisirsa `_check_strata` gurultulu hata verir - aksi halde
# config'te bir parametre gorunur ama degistirmek hicbir sey yapmaz.
SUPPORTED_STRATA = ["month", "rating", "text_length_bucket"]

TRIAL_NAME = "prompt_trial_{n}.csv"
TRIAL_IDS_NAME = "prompt_trial_ids.json"

# Prompt gelistirme orneginin katman paylari. precision_check ile ayni mantik:
# insanin HER IKI hata yonunu de gormesi gerekiyor, yoksa prompt tek yone ayarlanir.
#
# `received` 2026-08-27 denetiminde eklendi. Eskiden uc katman vardi ve
# `kw_gift_received` satirlari HICBIRINE dusmuyordu: 200 satirlik deneme
# gecisinde "hediye ALMIS" vakasindan sifir ornek vardi. Oysa "receiving a gift
# is not giving one" prompt'un uc kritik ayrimindan biri - yani hic sinanmadan
# dogrulanmis sayilacakti.
TRIAL_STRATA = {
    "proxy": 0.35,        # vekilin yanlis pozitifleri
    "speculative": 0.25,  # "hediye olur" tuzagi
    "received": 0.15,     # hediye ALAN, veren degil
    "unflagged": 0.25,    # vekilin kacirdiklari
}

# Elle doldurulacak etiket sozlugu; configs/annotation_schema.json ile ayni.
# v3 (2026-08-27): `received` eklendi - hediye ALAN, veren degil.
LABELS = ("gift_given", "self", "household", "received", "unclear")

# Ornekten tasinacak kolonlar. Metin dahil - LLM'e gidecek.
SAMPLE_COLUMNS = [
    "row_id", "category", "sample_frame", "stratum_id",
    "month", "rating", "n_words",
    # Urun kimligi ve adi: LLM "she loved it" cumlesindeki "it"i bilmeden
    # etiketliyordu. `parent_asin` izlenebilirlik icin tasiniyor.
    "parent_asin", "product_title", "product_category",
    "title", "text",
    "kw_gift_evidence", "kw_gift_speculative", "kw_gift_received", PROXY_COL,
]


def _stratum_seed(base: int, key: str) -> int:
    """Katman basina turetilmis, kosular ve platformlar arasi KARARLI seed.

    Tum katmanlarda ayni seed'i kullanmak, ayni buyuklukteki havuzlarda BIREBIR
    ayni konumlarin secilmesine yol aciyordu (2026-08-27 denetimi: iki farkli
    katman da [275, 607, 687, 702, 851] konumlarini sectі). Nokta tahmini yansiz
    kaliyordu ama katmanlar arasi bagimsizlik yoktu ve bootstrap guven araliklari
    bagimsiz cekim varsayiyor.

    `hash()` kullanilamaz: PYTHONHASHSEED ile string hash'i her sureçte degisir
    ve ornek yeniden uretilemez hale gelir. crc32 deterministiktir.
    """
    return (base + zlib.crc32(str(key).encode("utf-8"))) % (2**31 - 1)


def _check_strata(cfg: Config) -> None:
    """Config'teki katman listesi kodun uyguladigiyla ayni mi."""
    declared = list(cfg.get("sampling.strata"))
    if declared != SUPPORTED_STRATA:
        raise ValueError(
            f"sampling.strata = {declared}, kod ise {SUPPORTED_STRATA} uyguluyor. "
            "_stratum_id() bu boyutlari sabit kuruyor; config'i tek basina "
            "degistirmek ciktiyi degistirmez. Ikisini birlikte guncelleyin."
        )


def annotation_sample_path(cfg: Config, role: str) -> Path:
    return cfg.path("interim", f"{cfg.category_slug(role)}_annotation_sample.parquet")


def allocation_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"sampling_allocation_{cfg.category_slug(role)}.json")


def trial_path(cfg: Config, n: int) -> Path:
    return cfg.path("human", TRIAL_NAME.format(n=n))


def trial_ids_path(cfg: Config) -> Path:
    return cfg.path("human", TRIAL_IDS_NAME)


# --------------------------------------------------------------------- tahsis
def proportional_allocation(
    counts: dict[Hashable, int], n: int
) -> dict[Hashable, int]:
    """Orantili tahsis, en buyuk kalan yontemiyle.

    Toplami TAM OLARAK min(n, sum(counts)) eder ve hicbir hucreye o hucrede
    var olandan fazlasini vermez. Yuvarlama kalanini en buyuk kesirli paya sahip
    hucreye vermek, kucuk hucrelerin sistematik olarak sifirlanmasini engeller.
    """
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in counts}

    target = min(n, total)
    exact = {k: v * target / total for k, v in counts.items()}
    alloc = {k: min(int(math.floor(x)), counts[k]) for k, x in exact.items()}

    # Kalanlari kesirli payi buyukten kucuge dagit. Anahtar sirasi determinist
    # olsun diye ikinci anahtar olarak repr kullaniliyor (seed'den bagimsiz).
    order = sorted(counts, key=lambda k: (-(exact[k] - math.floor(exact[k])), repr(k)))
    remainder = target - sum(alloc.values())
    while remainder > 0:
        progressed = False
        for k in order:
            if remainder == 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                remainder -= 1
                progressed = True
        if not progressed:  # pragma: no cover - matematiksel olarak ulasilamaz
            break
    return alloc


def allocate_by_share(
    shares: dict[Hashable, float], n: int, caps: dict[Hashable, int] | None = None
) -> dict[Hashable, int]:
    """Sabit paylari tam sayilara boler; toplam TAM OLARAK n eder (kapasite izin verdikce).

    `round()` ile pay pay yuvarlamak toplami kaydirir - Python bankaci yuvarlamasi
    kullandigi icin round(12.5) == 12 ve 200 istenen bir sette 196 satir cikar.
    Burada da en buyuk kalan yontemi kullaniliyor.
    """
    caps = caps or {}
    exact = {k: n * w for k, w in shares.items()}
    alloc = {
        k: min(int(math.floor(x)), caps.get(k, n)) for k, x in exact.items()
    }
    order = sorted(shares, key=lambda k: (-(exact[k] - math.floor(exact[k])), repr(k)))
    remainder = min(n, sum(caps.get(k, n) for k in shares)) - sum(alloc.values())
    while remainder > 0:
        progressed = False
        for k in order:
            if remainder == 0:
                break
            if alloc[k] < caps.get(k, n):
                alloc[k] += 1
                remainder -= 1
                progressed = True
        if not progressed:  # pragma: no cover - kapasite doldu
            break
    return alloc


def _length_bucket(cuts: list[int]) -> pl.Expr:
    """`n_words` -> kova etiketi. Kesme noktalari config'ten gelir."""
    expr = pl.when(pl.col("n_words") < cuts[0]).then(pl.lit(f"<{cuts[0]}"))
    for lo, hi in zip(cuts, cuts[1:]):
        expr = expr.when(pl.col("n_words") < hi).then(pl.lit(f"{lo}-{hi}"))
    return expr.otherwise(pl.lit(f">={cuts[-1]}")).alias("len_bucket")


def _stratum_id() -> pl.Expr:
    """month | rating | uzunluk kovasi -> tek bir katman anahtari."""
    return pl.concat_str(
        [
            pl.col("month").cast(pl.String),
            pl.col("rating").cast(pl.Int64).cast(pl.String),
            pl.col("len_bucket"),
        ],
        separator="|",
    ).alias("stratum_id")


# -------------------------------------------------------------------- ornekleme
def _thin_frame(cfg: Config, role: str, cuts: list[int]) -> pl.DataFrame:
    """Metinsiz ince cerceve: 12.4M satirlik kategoride metni bellege almiyoruz.

    `build_clean` ile ayni strateji (bkz. data/preprocess.py). Metin ancak
    secilen ~15K satir icin, semi-join ile geri getiriliyor.
    """
    flags = pl.scan_parquet(keyword_path(cfg, role)).select(
        "row_id", "kw_gift_evidence", "kw_gift_speculative", "kw_gift_received",
        PROXY_COL,
    )
    thin = pl.scan_parquet(clean_parquet_path(cfg, role)).select(
        "row_id", "rating", "month", "n_words"
    )
    return (
        thin.join(flags, on="row_id", how="inner")
        .with_columns(_length_bucket(cuts))
        .with_columns(_stratum_id())
        .drop("len_bucket")
        .collect()
    )


def _draw_main(thin: pl.DataFrame, n: int, seed: int) -> tuple[pl.DataFrame, dict]:
    """Orantili katmanli cekim. Ikinci deger tahsis raporu."""
    tally = thin.group_by("stratum_id").len().sort("stratum_id")
    counts = dict(zip(tally["stratum_id"].to_list(), tally["len"].to_list()))
    alloc = proportional_allocation(counts, n)

    picks = []
    for stratum, take in alloc.items():
        if take <= 0:
            continue
        pool = thin.filter(pl.col("stratum_id") == stratum)
        # Katman basina TURETILMIS seed - ortak seed ayni boyutlu havuzlarda
        # ayni konumlari sectiriyordu. Bkz. _stratum_seed.
        picks.append(pool.sample(take, seed=_stratum_seed(seed, stratum), shuffle=True))

    drawn = pl.concat(picks) if picks else thin.head(0)
    report = {
        "n_strata": len(counts),
        "n_strata_allocated": sum(1 for v in alloc.values() if v > 0),
        "n_requested": n,
        "n_drawn": drawn.height,
        "population": sum(counts.values()),
        "cells": [
            {"stratum_id": k, "population": counts[k], "allocated": alloc[k]}
            for k in sorted(counts, key=lambda s: -counts[s])
        ],
    }
    return drawn, report


def _draw_boost(
    thin: pl.DataFrame,
    exclude: pl.DataFrame,
    n: int,
    seed: int,
    *,
    mask: pl.Expr | None = None,
    tag: str = FRAME_BOOST,
) -> pl.DataFrame:
    """Carpitilmis bir havuzdan ek ornek; daha once cekilenlerle AYRIK.

    `mask` verilmezse `kw_gift_proxy` havuzu kullanilir (pozitif takviyesi).
    `received` takviyesi icin `kw_gift_received` maskesi geciliyor.
    """
    mask = pl.col(PROXY_COL) if mask is None else mask
    pool = thin.filter(mask).join(exclude.select("row_id"), on="row_id", how="anti")
    if pool.height == 0:
        log.warning("'%s' havuzu bos - kategoride uygun satir yok", tag)
        return thin.head(0)
    if pool.height < n:
        log.warning("'%s' havuzunda %d satir var, %d istendi", tag, pool.height, n)
    return pool.sample(min(n, pool.height), seed=_stratum_seed(seed, tag), shuffle=True)


def build_sample(cfg: Config, role: str, *, force: bool = False) -> Path:
    dest = annotation_sample_path(cfg, role)
    if should_skip(dest, force, log):
        return dest

    if not clean_parquet_path(cfg, role).exists():
        build_clean(cfg, role)
    if not keyword_path(cfg, role).exists():
        raise FileNotFoundError(
            f"Once keyword_scan kosun: {keyword_path(cfg, role)} yok"
        )
    # `build_clean` ile ayni davranis: turetilebilir olan turetilir, indirilmesi
    # gereken icin gurultulu hata verilir. Metadata indirmesi aga bagli, o yuzden
    # sessizce halledilemez.
    if not meta_parquet_path(cfg, role).exists():
        if not raw_meta_path(cfg, role).exists():
            raise FileNotFoundError(
                f"Urun metadata'si yok: {raw_meta_path(cfg, role)}\n"
                "Once indirin:\n"
                f"  python -m gift_contamination.data.download --category {role} --meta"
            )
        build_meta_parquet(cfg, role)

    _check_strata(cfg)
    seed = int(cfg.get("seed"))
    cuts = list(cfg.get("sampling.text_length_buckets"))
    n_roles = len(cfg.category_roles())
    n_main = int(cfg.get("sampling.n_annotate")) // n_roles
    n_boost = round(n_main * float(cfg.get("sampling.keyword_boost_fraction")))
    n_received = round(n_main * float(cfg.get("sampling.received_boost_fraction")))

    thin = _thin_frame(cfg, role, cuts)
    log.info("%s: %s satirlik havuz, %s katman", role, f"{thin.height:,}", thin["stratum_id"].n_unique())

    main, report = _draw_main(thin, n_main, seed)
    boost = _draw_boost(thin, main, n_boost, seed)
    # "Hediye ALDIM" takviyesi: vekilin tanim geregi disladigi (PROXY = evidence
    # AND NOT received) ama LLM'in ayirmasi gereken en zor negatif sinif. Korpusta
    # %0.08-0.28 oraninda; takviye edilmezse ne egitim setinde ne de dogrulama
    # setinde yeterli ornek olur.
    received = _draw_boost(
        thin,
        pl.concat([main.select("row_id"), boost.select("row_id")]),
        n_received,
        seed,
        mask=pl.col("kw_gift_received"),
        tag=FRAME_RECEIVED,
    )

    drawn = pl.concat(
        [
            main.with_columns(pl.lit(FRAME_MAIN).alias("sample_frame")),
            boost.with_columns(pl.lit(FRAME_BOOST).alias("sample_frame")),
            received.with_columns(pl.lit(FRAME_RECEIVED).alias("sample_frame")),
        ]
    )

    # Metni ve urun kimligini ancak simdi, secilen satirlar icin getiriyoruz.
    text = (
        pl.scan_parquet(clean_parquet_path(cfg, role))
        .select("row_id", "title", "text", "parent_asin")
        .join(drawn.lazy().select("row_id"), on="row_id", how="semi")
        .collect()
    )
    # Urun adi LEFT join: metadata'da olmayan parent_asin bos string kalir ve
    # kacirma orani raporlanir. "kapsam tamdir" varsayilmaz, olculur.
    meta = load_join_frame(cfg, role)
    joined = text.join(meta, on="parent_asin", how="left").with_columns(
        pl.col("product_title").fill_null(""),
        pl.col("product_category").fill_null(""),
    )
    # Join kacirmasi VE metadata'da bos gelen basliklar birlikte sayiliyor:
    # LLM icin ikisi de ayni sey - urun adi gormeyecek.
    n_missing = int((joined["product_title"].str.strip_chars() == "").sum())
    if n_missing:
        log.warning(
            "%s: %s/%s satirda urun adi bos (eslesmeyen parent_asin veya metadata'da bos baslik)",
            role, f"{n_missing:,}", f"{joined.height:,}",
        )

    out = (
        drawn.join(joined, on="row_id", how="inner")
        .with_columns(pl.lit(cfg.category_slug(role)).alias("category"))
        .select(SAMPLE_COLUMNS)
        .sort("row_id")
    )

    out.write_parquet(dest)
    log_output(
        log, dest, n_rows=out.height,
        note=f"main={main.height:,} boost={boost.height:,} received={received.height:,}",
    )

    report.update(
        {
            "category": cfg.category_slug(role),
            "seed": seed,
            "text_length_buckets": cuts,
            "strata": SUPPORTED_STRATA,
            "n_boost_requested": n_boost,
            "n_boost_drawn": boost.height,
            "n_received_requested": n_received,
            "n_received_drawn": received.height,
            "n_missing_product_title": n_missing,  # bos VEYA eslesmeyen
            # HESAPLANIR, iddia edilmez: rapora dogrulanmamis bir sabit yazmak,
            # kod degistiginde sessizce yalan soyleyen bir alan birakir.
            "frames_disjoint": bool(out["row_id"].n_unique() == out.height),
            "frames": dict(
                out.group_by("sample_frame").len().sort("sample_frame").iter_rows()
            ),
            "note": (
                "Yaygınlık tahmini YALNIZCA sample_frame == 'main' uzerinden "
                "hesaplanir; 'boost' vekil-isaretli havuzdan cekildigi icin "
                "orani yukari cekerdi."
            ),
        }
    )
    write_json(report, allocation_path(cfg, role), log)
    return dest


# ------------------------------------------------------------- prompt denemesi
def _trial_strata(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Dort AYRIK ve TUKETICI katman: her satir tam olarak birine duser.

    Eski uc katmanli hali bayrak uzayini kapsamiyordu: `kw_gift_received`
    satirlari hicbirine dusmuyordu ve deneme setine giremiyordu.
    `test_trial_strata_partition_the_frame` bunu kilitliyor.
    """
    received = pl.col("kw_gift_received")
    return {
        "proxy": df.filter(pl.col(PROXY_COL)),  # evidence AND NOT received
        "received": df.filter(received),
        "speculative": df.filter(
            pl.col("kw_gift_speculative") & ~pl.col(PROXY_COL) & ~received
        ),
        "unflagged": df.filter(
            ~pl.col("kw_gift_evidence") & ~pl.col("kw_gift_speculative") & ~received
        ),
    }


def build_trial(cfg: Config, roles: list[str], n: int, *, force: bool = False) -> Path:
    """Prompt gelistirme icin elle etiketlenecek kucuk CSV.

    Bu bir TAHMIN ornegi DEGIL: zor vakalari kasitli olarak fazla temsil ediyor,
    cunku prompt'u kiran seyler onlar. Buradan yaygınlık okunmaz.

    Cekilen row_id'ler ayrica JSON'a yazilir. Prompt bu satirlar uzerinde
    ayarlanacagi icin, Hafta 4'un dogrulama seti onlari DISLAMALIDIR - aksi
    halde prompt kendi test setine fit edilmis olur.
    """
    dest = trial_path(cfg, n)
    if should_skip(dest, force, log):
        return dest

    seed = int(cfg.get("seed"))
    # Once n'i kategorilere, sonra her kategorinin payini katmanlara boluyoruz;
    # iki adim da en buyuk kalan yontemiyle, boylece toplam tam olarak n kalir.
    per_role = allocate_by_share({r: 1 / len(roles) for r in roles}, n)
    frames = []

    for role in roles:
        src = annotation_sample_path(cfg, role)
        if not src.exists():
            raise FileNotFoundError(f"Once ornegi uretin: {src} yok")
        df = pl.read_parquet(src)
        pools = _trial_strata(df)
        takes = allocate_by_share(
            TRIAL_STRATA,
            per_role[role],
            caps={s: p.height for s, p in pools.items()},
        )
        for stratum, pool in pools.items():
            take = takes[stratum]
            if pool.height == 0 or take == 0:
                log.warning("%s / %s katmani bos veya pay almadi", role, stratum)
                continue
            frames.append(
                pool.sample(take, seed=_stratum_seed(seed, f"{role}|{stratum}"), shuffle=True)
                .with_columns(pl.lit(stratum).alias("trial_stratum"))
            )

    out = (
        pl.concat(frames)
        # Katmanlari karistir: siralamanin kendisi etiketleyiciye ipucu vermesin.
        .sample(fraction=1.0, seed=seed, shuffle=True)
        .with_row_index("trial_id", offset=1)
        .with_columns(pl.lit("").alias("label"), pl.lit("").alias("notes"))
        .select(
            "trial_id", "category", "row_id", "title", "text", "label", "notes",
            "trial_stratum", "sample_frame", PROXY_COL,
        )
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(dest)
    log_output(log, dest, n_rows=out.height)

    write_json(
        {
            "n": out.height,
            "seed": seed,
            # KATEGORI BAZLI. `row_id` her kategoride 0'dan basliyor, yani
            # kategoriler arasinda CAKISIYOR (olculdu: dort kategori arasinda
            # 78 ortak deger). Duz bir row_id listesiyle dislama yapmak baska
            # kategorilerde masum satirlari da atar.
            "excluded": {
                cat[0]: sorted(g["row_id"].to_list())
                for cat, g in sorted(out.group_by("category"), key=lambda kv: kv[0])
            },
            "purpose": "prompt gelistirme (v1 -> v2)",
            "exclude_from_validation": True,
            "note": (
                "Prompt bu satirlar uzerinde ayarlandi. Hafta 4'un 500'luk "
                "dogrulama seti bu satirlari DISLAMALI; aksi halde prompt kendi "
                "test setine fit edilmis olur. Dislama anahtari (category, row_id) "
                "CIFTIDIR - tek basina row_id benzersiz DEGIL."
            ),
        },
        trial_ids_path(cfg),
        log,
    )
    log.info(
        "Simdi `label` sutununu elle doldurun. Gecerli degerler: %s. "
        "`trial_stratum` ve kw_* kolonlarini OKUMADAN etiketleyin - onyargi yaratir.",
        ", ".join(LABELS),
    )
    log.info("GIZLILIK: bu dosya birebir review metni tasir ve git'e GIRMEZ.")
    return dest


# ------------------------------------------- deneme setinin LLM'e verilecek hali
TRIAL_SOURCE_NAME = "prompt_trial_{n}_source.parquet"

# CSV'de olmayan `stratum_id` disinda SAMPLE_COLUMNS ile ayni; ustune `trial_id`.
# Ayni sekle sadik kalmak, `llm_annotate`in ayni kolon secicilerini kullanmasini
# ve deneme kosusunun gercek kosudan farkli bir yol izlememesini sagliyor.
TRIAL_SOURCE_COLUMNS = [
    "trial_id", "row_id", "category", "sample_frame",
    "month", "rating", "n_words",
    "parent_asin", "product_title", "product_category",
    "title", "text",
]


def trial_source_path(cfg: Config, n: int) -> Path:
    return cfg.path("interim", TRIAL_SOURCE_NAME.format(n=n))


def build_trial_source(cfg: Config, n: int, *, force: bool = False) -> Path:
    """Elle etiketlenen deneme CSV'sini LLM'e verilebilir parquet'e cevirir.

    NEDEN AYRI BIR ADIM: deneme satirlari annotation orneginin ICINDE DEGIL.
    `build_trial` onlari 2026-08-26 tarihli ornekten cekmisti; ornek ertesi gun
    `boost_received` cercevesi eklenince yeniden cekildi ve 200 satirin hicbiri
    yeni cekiliste kalmadi (olculdu: 50 Toys satirinin 0'i). Bu bir hata degil -
    16M satirlik korpustan 11.800 cekiliste 50 satirin beklenen kesisimi 0,04.
    Ama Kapi 1'in dorduncu olcutu insan etiketiyle LLM etiketini karsilastirdigi
    icin bu 200 satirin AYRICA etiketlenmesi gerekiyor.

    CSV `product_title` TASIMIYOR - etiketleyiciye gosterilmemisti. LLM uretimde
    urun adini goruyor, o yuzden korpustan geri getiriliyor: aksi halde uyum
    sayisi prompt'un gercekte kostugu girdiyi olcmezdi.
    """
    dest = trial_source_path(cfg, n)
    if should_skip(dest, force, log):
        return dest

    src = trial_path(cfg, n)
    if not src.exists():
        raise FileNotFoundError(f"Once deneme CSV'sini uretin: {src} yok")

    # ETIKETSIZ dosya okunuyor. Etiketli olan ayni satirlari tasiyor ama insan
    # etiketini bu boru hattina hic sokmamak "LLM etiketi gordu mu" sorusunu
    # bastan ortadan kaldiriyor.
    trial = pl.read_csv(src)
    slug_to_role = {cfg.category_slug(r): r for r in cfg.get("dataset.categories")}

    frames = []
    for key, group in sorted(trial.group_by("category"), key=lambda kv: kv[0]):
        slug = key[0]
        role = slug_to_role.get(slug)
        if role is None:
            raise ValueError(
                f"Deneme CSV'sindeki '{slug}' kategorisi config'te yok. "
                f"Tanimli olanlar: {sorted(slug_to_role)}"
            )
        wanted = group.select("trial_id", "row_id", "sample_frame", "text")
        corpus = (
            pl.scan_parquet(clean_parquet_path(cfg, role))
            .select("row_id", "title", "text", "parent_asin",
                    "rating", "month", "n_words")
            .join(wanted.lazy().select("row_id"), on="row_id", how="semi")
            .collect()
        )
        if corpus.height != group.height:
            raise RuntimeError(
                f"{slug}: deneme CSV'sinde {group.height} satir var, korpusta "
                f"{corpus.height} eslesti. `row_id` kaymis olabilir - deneme "
                "CSV'si ile clean parquet ayni surumden gelmiyor."
            )

        joined = wanted.rename({"text": "csv_text"}).join(
            corpus, on="row_id", how="inner"
        )
        # BUTUNLUK KONTROLU: `row_id` korpus capinda kararli olmali. Metin birebir
        # tutmuyorsa join dogru satiri getirmemis demektir ve insan etiketi baska
        # bir review'a bagli kalir - uyum sayisi anlamsizlasir.
        drift = int(
            (joined["csv_text"].fill_null("") != joined["text"].fill_null("")).sum()
        )
        if drift:
            raise RuntimeError(
                f"{slug}: {drift}/{joined.height} satirda korpus metni deneme "
                "CSV'siyle tutmuyor. `row_id` eslesmesi guvenilir degil."
            )

        meta = load_join_frame(cfg, role)
        frames.append(
            joined.join(meta, on="parent_asin", how="left")
            .with_columns(
                pl.col("product_title").fill_null(""),
                pl.col("product_category").fill_null(""),
                pl.lit(slug).alias("category"),
            )
            .select(TRIAL_SOURCE_COLUMNS)
        )

    out = pl.concat(frames).sort("trial_id")
    n_missing = int(out["product_title"].str.strip_chars().eq("").sum())
    if n_missing:
        log.warning("%s/%s satirda urun adi bos", n_missing, out.height)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    log_output(log, dest, n_rows=out.height,
               note=f"{out['category'].n_unique()} kategori")
    log.info("GIZLILIK: bu dosya birebir review metni tasir ve git'e GIRMEZ.")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument(
        "--trial",
        type=int,
        metavar="N",
        help="Ornek yerine N satirlik prompt gelistirme CSV'si uret",
    )
    parser.add_argument(
        "--trial-source",
        type=int,
        metavar="N",
        help="Elle etiketlenmis deneme CSV'sini LLM'e verilebilir parquet'e cevir "
             "(Kapi 1'in dorduncu olcutu icin)",
    )
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    roles = resolve_roles(cfg, args.category)

    if args.trial:
        build_trial(cfg, roles, args.trial, force=args.force)
        return 0

    if args.trial_source:
        build_trial_source(cfg, args.trial_source, force=args.force)
        return 0

    for role in roles:
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        build_sample(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
