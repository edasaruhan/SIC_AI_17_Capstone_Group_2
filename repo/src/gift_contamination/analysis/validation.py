"""Hafta 4 - insan dogrulamasi. Projenin TEK gercek referansi.

Girdi : data/annotations/human/validation_<n>_labeled.csv (label_<harf> dolu)
Cikti : reports/results/validation_<n>.json + reports/figures/F18_*.png

IKI MOD, config belirliyor (`validation.annotators`, `validation.reliability`):

  * Birden cok etiketleyici + `reliability: fleiss` -> ozgun tasarim. Referans
    cogunluk uzlasisi, kapi Fleiss kappa.
  * TEK etiketleyici + `reliability: none` -> 2026-09-14'ten beri FIILI durum.
    Yalnizca A teslim etti. Referans A'nin etiketi. Uyum olculemez, o yuzden
    kapi olcutu `passed: null` ve karar INCOMPLETE - ASLA PASS yazilmaz.
    Olcum yontemleri A'nin etiketleri modelle karsilastirilmadan ONCE kaydedildi
    (DECISIONS 2026-09-14, commit 8c697a8).

Olcumler ve cevapladiklari sorular:

  1. FLEISS KAPPA - "gorev tanimi acik mi?" (yalnizca cok etiketleyicide)
     Etiketleyiciler birbiriyle ne kadar uyusuyor. Dusukse sorun detektorde
     degil SEMADA: insanlar bile ayni satiri farkli etiketliyorsa modelden
     tutarlilik beklemek anlamsiz. KAPI olcutu bu (esik `validation.kappa_min`).

  1b. AGIRLIKLI OLCUM - "populasyonda detektor ne kadar dogru?"
     Dogrulama LLM etiketine gore BILEREK dengesiz cekildi; agirliksiz F1
     `household`/`gift_given` yonunde yanlidir. Yalnizca `main` satirlariyla,
     (kategori x LLM sinifi) hucresine ters olasilik agirligi verilerek
     populasyon duzeyi olcum ayrica raporlanir. Bootstrap GA ikisinde de.

  2. LLM vs INSAN - "detektor ne kadar dogru?"
     Sinif bazli precision/recall/F1 + yon yon karisiklik. KAPI DEGIL, olcum:
     esigi hicbir belgede sabitlenmedi ve sonucu gordukten sonra bir esik
     uydurmak, kapiyi sonradan kurmak olur. Raporlanir, karara baglanmaz.

  3. VEKIL vs INSAN UZLASI SI - "LLM gercekten regex'ten iyi mi?"
     Sozcuksel vekilin ILK BAGIMSIZ precision/recall'i. Simdiye kadarki
     0,58 / 0,61 / 0,76 sayilari "yazar destekli on gecis"ti (DECISIONS
     2026-08-25, 2026-08-26); bunlar onlarin yerini alir.

KAPPA'DA SEYREK SINIF KURALI. `validation.min_class_n_for_kappa` (=20) altinda
kalan sinif genel kappa'ya GIRMEZ. Operasyonel karsiligi: o sinifi HERHANGI bir
etiketleyicinin kullandigi SATIRLAR dusurulur - kategori sutununu silmek satir
basina degerlendirici sayisini bozar ve Fleiss'in varsayimini gecersiz kilar.
Kural sonuc gorulmeden yazildi (DECISIONS 2026-08-29). Kisitlanmamis kappa da
raporlanir ki secim seffaf kalsin.

Kullanim:
    python -m gift_contamination.analysis.validation --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..data.labelsheet import LABEL_SOURCE, LABEL_SOURCE_COLUMN, labeled_path
from ..data.sampling import LABELS, validation_path
from ..detection.llm_annotate import annotation_path
from ..detection.schema import CONDITION_OF, CONTAMINATION
from ..utils.io import write_json
from ..utils.logging import get_logger
from . import viz
from .keyword_scan import PROXY_COL

log = get_logger("analysis.validation")

GIFT = "gift_given"
# Uzlasi saglanamayan satirin etiketi. Sessizce bir sinifa itmek uzlasi
# oranini SISIRIR ve tam da olcmek istedigimiz belirsizligi gizler.
TIE = "tie"
FRAME_MAIN = "main"
KENDI_COCUGU = "KENDI_COCUGU"

# `validation.reliability` degerleri. Baska bir deger UYGULANMADI ve gurultulu
# hata verir - sessizce "none" gibi davranmak olculmemis bir seyi olculmus gibi
# gosterirdi.
RELIABILITY_MODES = ("fleiss", "none")


def validation_report_path(cfg: Config, n: int) -> Path:
    return cfg.path("results", f"validation_{n}.json")


def figure_path(cfg: Config, n: int) -> Path:
    ext = cfg.get("eda.figure_format", "png")
    return cfg.path("figures", f"F18_llm_vs_human_{n}.{ext}")


# ------------------------------------------------------------------- yukleme
def label_columns(df: pl.DataFrame) -> list[str]:
    """Etiketleyici kolonlari: `label_A`, `label_B`, ...

    Duz bir `startswith("label_")` YETMIYOR: `labelsheet` ciktisi ayrica
    `label_source` damgasini tasiyor ve o da onekle esleser - dordunca bir
    etiketleyici sanilir ve `load_labels` yanlis yere hata verir (olculdu).
    Etiket TAG'leri `annotator_tags` geregi TEK BUYUK HARF.
    """
    return sorted(
        c for c in df.columns
        if len(c) == len("label_") + 1
        and c.startswith("label_")
        and c[-1].isupper()
    )


def load_labels(cfg: Config, n: int | None = None) -> pl.DataFrame:
    """Doldurulmus dogrulama CSV'sini okur ve eksiksizligini dogrular."""
    n = int(n if n is not None else cfg.get("validation.n"))
    path = labeled_path(validation_path(cfg, n))
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} yok. Once sayfalari toplayin:\n"
            "  python -m gift_contamination.data.labelsheet --validation --ingest"
        )
    df = pl.read_csv(path)

    # `gate1`in backend guard'iyla AYNI gerekce. Elle yazilmis ya da sentetik
    # bir dosya gercek gorunen bir PASS uretebilir ve o rakam rapora "insan
    # dogrulamasi" diye girer. Damga yalnizca `labelsheet --ingest` yolundan
    # basiliyor.
    damga = df[LABEL_SOURCE_COLUMN][0] if LABEL_SOURCE_COLUMN in df.columns else None
    if damga != LABEL_SOURCE:
        raise RuntimeError(
            f"{path.name} '{damga}' damgasi tasiyor, '{LABEL_SOURCE}' bekleniyordu. "
            "Etiketler `labelsheet --validation --ingest` yolundan gelmeli; elle "
            "yazilmis veya sentetik bir dosya Hafta 4 olcumune giremez."
        )

    cols = label_columns(df)
    tags, _ = annotator_mode(cfg)
    want = [f"label_{t}" for t in tags]
    if cols != want:
        raise ValueError(
            f"{len(cols)} etiketleyici kolonu var ({cols}), config {want} bekliyor. "
            "Eksik bir sayfayla hesaplanan kappa yanlis olur; fazla bir sayfa ise "
            "config'te adi gecmeyen birinin etiketini sessizce olcume sokar."
        )

    eksik = df.filter(pl.any_horizontal(pl.col(c).is_null() for c in cols))
    if eksik.height:
        # Fleiss satir basina SABIT degerlendirici sayisi varsayar. Yarim
        # doldurulmus satirlari sessizce tasimak kappa'yi bozar.
        log.warning(
            "%d satirda en az bir etiketleyici bos birakmis - kappa disi kaliyor",
            eksik.height,
        )
    return df


def annotator_mode(cfg: Config) -> tuple[list[str], str]:
    """(etiketleyici harfleri, guvenilirlik modu) - tutarsizsa gurultulu hata.

    Tek etiketleyiciyle uyum OLCULEMEZ: `fleiss` istemek, hesaplanamayacak bir
    kapiyi var gibi gostermek olurdu. Tersine, birden cok etiketleyici varken
    `none` demek elde olan uyum bilgisini atmak olurdu - o da reddediliyor.
    """
    tags = [str(t) for t in cfg.get("validation.annotators")]
    mode = str(cfg.get("validation.reliability"))
    if mode not in RELIABILITY_MODES:
        raise NotImplementedError(
            f"validation.reliability='{mode}' uygulanmadi. Gecerli: {RELIABILITY_MODES}"
        )
    if not tags:
        raise ValueError("validation.annotators bos")
    if len(tags) == 1 and mode != "none":
        raise ValueError(
            f"Tek etiketleyiciyle (A) '{mode}' uyumu hesaplanamaz. "
            "`validation.reliability: none` olmali ve kapi INCOMPLETE kalir."
        )
    if len(tags) > 1 and mode == "none":
        raise ValueError(
            f"{len(tags)} etiketleyici var ama reliability=none. Uyum olculebilirken "
            "atilmaz; `fleiss` kullanin."
        )
    return tags, mode


def reference_labels(df: pl.DataFrame, cols: list[str]) -> pl.Series:
    """Referans etiket: tek etiketleyicide ONUN etiketi, cokta cogunluk uzlasisi."""
    if len(cols) == 1:
        return df[cols[0]].alias("reference")
    return consensus(df, cols).alias("reference")


def consensus(df: pl.DataFrame, cols: list[str]) -> pl.Series:
    """Cogunluk etiketi; cogunluk yoksa `tie`.

    Uc etiketleyicide "cogunluk yok" demek ucunun de farkli sey demesi
    demektir - o satir referans olamaz ve sayisi raporlanir.
    """
    out = []
    for row in df.select(cols).iter_rows():
        vals = [v for v in row if v is not None]
        if len(vals) < len(cols):
            out.append(None)
            continue
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        winners = [k for k, c in counts.items() if c == best]
        out.append(winners[0] if len(winners) == 1 else TIE)
    return pl.Series("consensus", out, dtype=pl.String)


# --------------------------------------------------------------------- kappa
def _rating_table(rows: list[tuple], classes: list[str]) -> np.ndarray:
    """(satir x sinif) sayim matrisi - Fleiss'in bekledigi bicim."""
    index = {c: i for i, c in enumerate(classes)}
    table = np.zeros((len(rows), len(classes)), dtype=int)
    for r, row in enumerate(rows):
        for value in row:
            table[r, index[value]] += 1
    return table


KAPPA_DECIMALS = 4


def _round(value: float | None) -> float | None:
    """Rapora yazilan kappa degerleri. `None` gecerli bir sonuc - korunur."""
    return None if value is None else round(value, KAPPA_DECIMALS)


def fleiss(rows: list[tuple], classes: list[str]) -> float | None:
    """Fleiss' kappa. Satir yoksa veya tek sinif kullanildiysa None.

    Tek sinif kullanildiginda beklenen uyum 1 olur ve kappa 0/0'a duser -
    statsmodels nan dondurur. `None` dondurup raporda gormek, nan'i
    sayiymis gibi tasimaktan iyidir.
    """
    if not rows:
        return None
    from statsmodels.stats.inter_rater import fleiss_kappa  # noqa: PLC0415

    table = _rating_table(rows, classes)
    if (table.sum(axis=0) > 0).sum() < 2:
        return None
    value = float(fleiss_kappa(table, method="fleiss"))
    return None if np.isnan(value) else value


def kappa_report(
    df: pl.DataFrame, cols: list[str], *, min_class_n: int
) -> dict:
    """Genel kappa (seyrek siniflar HARIC) + sinif bazli kappa.

    Seyrek sinifi disarida birakmanin operasyonel karsiligi: o sinifi
    herhangi bir etiketleyicinin kullandigi SATIRLARI dusurmek. Kategori
    sutununu silmek satir basina degerlendirici sayisini bozardi.
    """
    tam = df.filter(~pl.any_horizontal(pl.col(c).is_null() for c in cols))
    rows = list(tam.select(cols).iter_rows())
    classes = list(LABELS)

    kullanim = {c: 0 for c in classes}
    for row in rows:
        for value in row:
            kullanim[value] = kullanim.get(value, 0) + 1

    # HIC KULLANILMAYAN sinif "seyrek" degildir - onun icin dusen satir da yok.
    # Listeye koymak raporu gurultuyle doldurur ve gercekten dislanan sinifi
    # gozden kacirtir.
    seyrek = sorted(c for c in classes if 0 < kullanim[c] < min_class_n)
    tutulan = [c for c in classes if c not in seyrek]
    kalan = [r for r in rows if not (set(r) & set(seyrek))]

    # Rapora YUVARLANMIS, kapiya HAM deger gidiyor. Ikisini ayirmanin sebebi
    # tek basina kozmetik degil: 0,59996 gibi bir deger yuvarlanmis haliyle
    # 0,60 esigini GECERDI. Rapor okunabilir olsun diye yapilan bir islem,
    # kapinin kararini degistiremez.
    ham = fleiss(kalan, tutulan)
    return {
        "n_rows": len(rows),
        "n_incomplete_rows": df.height - len(rows),
        "n_annotators": len(cols),
        "ratings_by_class": kullanim,
        "min_class_n": min_class_n,
        # KAPI degeri bu: seyrek siniflarin gectigi satirlar disarida.
        "kappa": _round(ham),
        # Kapinin karsilastirdigi deger. Yuvarlanmamis.
        "kappa_exact": ham,
        "n_rows_in_kappa": len(kalan),
        "sparse_classes": seyrek,
        "n_rows_dropped_as_sparse": len(rows) - len(kalan),
        # Seffaflik icin: hicbir sey dislanmadan hesaplanan deger.
        "kappa_unrestricted": _round(fleiss(rows, classes)),
        # Bire-karsi-hepsi. Seyrek sinif da burada gorunur - dislanmasi
        # KAPI kararindan, raporlamadan degil.
        "kappa_per_class": {
            c: _round(fleiss([tuple(x == c for x in r) for r in rows], [True, False]))
            for c in classes
            if kullanim[c] > 0
        },
    }


# ---------------------------------------------------------------- kiyaslama
def compare(reference: pl.Series, prediction: pl.Series) -> dict:
    """Sinif bazli precision/recall/F1 + yon yon uyusmazlik.

    `gate1.trial_agreement` ile AYNI bicim - iki rapor yan yana okunabilsin.
    """
    ref = reference.to_list()
    pred = prediction.to_list()
    pairs = [(r, p) for r, p in zip(ref, pred) if r is not None and p is not None]
    if not pairs:
        # Bicim HER IKI dalda ayni: cagiran taraf hangi daldan geldigini
        # bilmek zorunda kalmasin.
        return {
            "n_compared": 0, "accuracy": None, "macro_f1": None,
            "per_class": {}, "disagreements": {},
        }

    classes = sorted({r for r, _ in pairs} | {p for _, p in pairs})
    per_class = {}
    for c in classes:
        tp = sum(1 for r, p in pairs if r == c and p == c)
        n_ref = sum(1 for r, _ in pairs if r == c)
        n_pred = sum(1 for _, p in pairs if p == c)
        precision = round(tp / n_pred, 3) if n_pred else None
        recall = round(tp / n_ref, 3) if n_ref else None
        # F1 = 2TP / (2TP + FP + FN). Onceki hali `if precision and recall`
        # idi: TP=0 olan sinifta precision 0.0 (falsy) oldugu icin F1 None
        # oluyor ve sinif makro-F1'den SESSIZCE dusuyordu - tamamen kacirilan
        # bir sinif ortalamayi YUKSELTIYORDU. Payda sifirsa gercekten tanimsiz.
        payda = 2 * tp + (n_pred - tp) + (n_ref - tp)
        f1 = round(2 * tp / payda, 3) if payda else None
        per_class[c] = {
            "n_reference": n_ref, "n_predicted": n_pred,
            "precision": precision, "recall": recall, "f1": f1,
        }

    yanlis: dict[str, int] = {}
    for r, p in pairs:
        if r != p:
            yanlis[f"{r}->{p}"] = yanlis.get(f"{r}->{p}", 0) + 1

    return {
        "n_compared": len(pairs),
        "accuracy": round(sum(1 for r, p in pairs if r == p) / len(pairs), 4),
        "macro_f1": round(
            float(np.mean([v["f1"] for v in per_class.values() if v["f1"] is not None])), 4
        ) if any(v["f1"] is not None for v in per_class.values()) else None,
        "per_class": per_class,
        # Ikincil anahtar (etiket cifti) SART - bkz. gate1.trial_agreement:
        # esit sayilar siralamasiz kalirsa rapor kosudan kosuya degisir.
        "disagreements": dict(sorted(yanlis.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _proxy_labels(df: pl.DataFrame) -> pl.Series:
    """Sozcuksel vekili bes sinifli semaya cevirir.

    Vekil ikili: ya `gift_given` der ya demez. Demedigi satirlari `self`
    saymak onu HAKSIZ yere cezalandirirdi (household/unclear zaten ayirt
    edemez), o yuzden yalnizca `gift_given` ekseninde kiyasliyoruz:
    isaretledi -> gift_given, isaretlemedi -> not_gift.
    """
    return (
        pl.when(pl.col(PROXY_COL)).then(pl.lit(GIFT)).otherwise(pl.lit("not_gift"))
    ).alias("proxy")


def _binary_gift(series: pl.Series, name: str) -> pl.Series:
    return pl.Series(
        name, [None if v is None else (GIFT if v == GIFT else "not_gift") for v in series]
    )


# ---------------------------------------------- agirlik + bootstrap (2026-09-14)
# Yontem A'nin etiketleri modelle karsilastirilmadan ONCE kaydedildi
# (DECISIONS 2026-09-14). Buradaki hicbir secim sonuca bakilarak yapilmadi.

def _confusion(ref: np.ndarray, pred: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Agirlikli karisiklik matrisi: satir = referans, sutun = tahmin."""
    return np.bincount(ref * k + pred, weights=w, minlength=k * k).reshape(k, k)


def _scores(conf: np.ndarray) -> dict[str, np.ndarray | float]:
    """Karisiklik matrisinden dogruluk, sinif bazli P/R/F1, makro-F1.

    Tanimsiz deger (sifira bolme) NaN - "0" degil. Bir sinifin hic referansi
    yoksa recall'u 0 degil TANIMSIZDIR; 0 yazmak detektoru haksiz cezalandirir.
    """
    tp = np.diag(conf)
    n_pred = conf.sum(axis=0)
    n_ref = conf.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp / n_pred, np.nan)
        recall = np.where(n_ref > 0, tp / n_ref, np.nan)
        # 2TP / (2TP + FP + FN): TP=0 ama sinif tahmin/referansta varsa F1 = 0,
        # tanimsiz DEGIL. Tanimsiz yalnizca sinif ne referansta ne tahminde varsa.
        payda = 2 * tp + (n_pred - tp) + (n_ref - tp)
        f1 = np.where(payda > 0, 2 * tp / payda, np.nan)
    toplam = conf.sum()
    tanimli = ~np.isnan(f1)
    return {
        "accuracy": float(tp.sum() / toplam) if toplam else float("nan"),
        "macro_f1": float(f1[tanimli].mean()) if tanimli.any() else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ref": n_ref,
    }


def population_counts(cfg: Config, categories: list[str]) -> dict[tuple[str, str], int]:
    """(kategori, LLM sinifi) -> `main` cercevesindeki satir sayisi.

    Populasyon: her kategorinin `main` cercevesi (10.000 satir). Dosyasi
    olmayan kategori sozlukte YER ALMAZ; cagiran taraf bunu kontrol eder.
    """
    slug_to_role = {cfg.category_slug(r): r for r in cfg.category_roles()}
    out: dict[tuple[str, str], int] = {}
    for cat in categories:
        role = slug_to_role.get(cat)
        if role is None:
            continue
        path = annotation_path(cfg, role)
        if not path.exists():
            continue
        main = pl.read_parquet(path, columns=["sample_frame", "purchase_type"]).filter(
            pl.col("sample_frame") == FRAME_MAIN
        )
        for k, n in main.group_by("purchase_type").len().iter_rows():
            out[(cat, k)] = int(n)
    return out


def ipw_weights(
    categories: list[str], llm: list[str], pop: dict[tuple[str, str], int],
    *, min_cell_n: int,
) -> tuple[np.ndarray, dict]:
    """Ters olasilik agirligi - onceden kaydedilmis kural.

    Hucre (c, k) icin `w = N_main(c,k) / n_dogrulama(c,k)`. Dogrulama satiri
    `min_cell_n`'den az olan (ya da hic olmayan) hucrenin populasyon kutlesi,
    k sinifinin BUTUN kategorilerdeki dogrulama satirlarina esit dagitilir.
    Hic dogrulama satiri olmayan sinifin kutlesi "kapsanmayan pay" olur.
    """
    n_cell: dict[tuple[str, str], int] = {}
    for c, k in zip(categories, llm):
        n_cell[(c, k)] = n_cell.get((c, k), 0) + 1
    n_class: dict[str, int] = {}
    for (_, k), n in n_cell.items():
        n_class[k] = n_class.get(k, 0) + n

    collapsed: list[str] = []
    artik: dict[str, float] = {}      # sinif -> dagitilacak kutle
    kapsanmayan = 0.0
    for (c, k), big_n in sorted(pop.items()):
        small_n = n_cell.get((c, k), 0)
        if small_n >= min_cell_n:
            continue
        if n_class.get(k, 0) == 0:
            kapsanmayan += big_n
            continue
        artik[k] = artik.get(k, 0.0) + big_n
        collapsed.append(f"{c}/{k} (n={small_n})")

    w = np.zeros(len(llm), dtype=float)
    for i, (c, k) in enumerate(zip(categories, llm)):
        small_n = n_cell[(c, k)]
        if small_n >= min_cell_n:
            w[i] += pop.get((c, k), 0) / small_n
        if k in artik:
            w[i] += artik[k] / n_class[k]

    toplam_pop = float(sum(pop.values()))
    return w, {
        "min_cell_n": min_cell_n,
        "population_rows": int(toplam_pop),
        "collapsed_cells": collapsed,
        "uncovered_share": round(kapsanmayan / toplam_pop, 6) if toplam_pop else None,
    }


def _strata_index(keys: list[tuple]) -> list[np.ndarray]:
    gruplar: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        gruplar.setdefault(key, []).append(i)
    return [np.asarray(v) for _, v in sorted(gruplar.items())]


def bootstrap_scores(
    ref: np.ndarray, pred: np.ndarray, w: np.ndarray, strata: list[np.ndarray],
    *, k: int, n_boot: int, seed: int,
) -> dict[str, np.ndarray]:
    """Katman ICINDE yeniden ornekleme; her tekrarda skorlar yeniden hesaplanir.

    Katmanlar agirlik hucreleriyle ayni oldugu icin hucre boyutlari her tekrarda
    korunur ve agirliklar gecerli kalir.
    """
    rng = np.random.default_rng(seed)
    acc = np.empty(n_boot)
    macro = np.empty(n_boot)
    p = np.empty((n_boot, k))
    r = np.empty((n_boot, k))
    f = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        s = _scores(_confusion(ref[idx], pred[idx], w[idx], k))
        acc[b], macro[b] = s["accuracy"], s["macro_f1"]
        p[b], r[b], f[b] = s["precision"], s["recall"], s["f1"]
    return {"accuracy": acc, "macro_f1": macro, "precision": p, "recall": r, "f1": f}


def _ci(values: np.ndarray, level: float = 0.95) -> list[float] | None:
    ok = values[~np.isnan(values)]
    if not len(ok):
        return None
    alt = (1 - level) / 2 * 100
    return [round(float(np.percentile(ok, alt)), 4),
            round(float(np.percentile(ok, 100 - alt)), 4)]


def _r(x: float) -> float | None:
    return None if x is None or np.isnan(x) else round(float(x), 4)


def scored_measurement(
    ref_labels: list[str], pred_labels: list[str], w: np.ndarray, strata_keys: list[tuple],
    *, n_boot: int, seed: int, min_class_n: int,
) -> dict:
    """Nokta tahmini + bootstrap GA, sinif bazli. Agirliksiz icin `w` = 1."""
    if not ref_labels:
        # Ornek: uc etiketleyicinin her satirda farkli dedigi durumda butun
        # satirlar `tie` olur. Bos kumeden skor uretmek yerine acikca soyle.
        return {"skipped": True, "reason": "karsilastirilacak satir yok", "n_rows": 0}
    classes = list(LABELS)
    k = len(classes)
    pos = {c: i for i, c in enumerate(classes)}
    # dtype ACIK: bos olmayan listede bile varsayilan tipe guvenilmez;
    # `np.bincount` yalnizca tamsayi kabul ediyor.
    ref = np.asarray([pos[v] for v in ref_labels], dtype=np.int64)
    pred = np.asarray([pos[v] for v in pred_labels], dtype=np.int64)
    w = np.asarray(w, dtype=float)

    nokta = _scores(_confusion(ref, pred, w, k))
    boot = bootstrap_scores(
        ref, pred, w, _strata_index(strata_keys), k=k, n_boot=n_boot, seed=seed,
    )
    ham_sayi = np.bincount(ref, minlength=k)

    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "n_reference": int(ham_sayi[i]),
            "weighted_n_reference": _r(nokta["n_ref"][i]),
            # Onceden kayitli kural: az ornekli sinif ISARETLENIR, hesaptan
            # cikarilmaz. Esik kappa'nin seyrek sinif kuraliyla ayni sayi.
            "sparse": bool(ham_sayi[i] < min_class_n),
            "precision": {"value": _r(nokta["precision"][i]), "ci95": _ci(boot["precision"][:, i])},
            "recall": {"value": _r(nokta["recall"][i]), "ci95": _ci(boot["recall"][:, i])},
            "f1": {"value": _r(nokta["f1"][i]), "ci95": _ci(boot["f1"][:, i])},
        }
    return {
        "n_rows": len(ref_labels),
        "accuracy": {"value": _r(nokta["accuracy"]), "ci95": _ci(boot["accuracy"])},
        "macro_f1": {"value": _r(nokta["macro_f1"]), "ci95": _ci(boot["macro_f1"])},
        "per_class": per_class,
        "bootstrap": {"n": n_boot, "seed": seed, "strata": "kategori x LLM sinifi"},
    }


def _binary_prf(ref: np.ndarray, pred: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    tp = float(w[ref & pred].sum())
    fp = float(w[~ref & pred].sum())
    fn = float(w[ref & ~pred].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    payda = 2 * tp + fp + fn
    f = 2 * tp / payda if payda > 0 else float("nan")
    return p, r, f


def binary_axis_measurement(
    ref_labels: list[str], pred_labels: list[str], w: np.ndarray, strata_keys: list[tuple],
    positive: tuple[str, ...], *, n_boot: int, seed: int,
) -> dict:
    """Deneyin KULLANDIGI ikili karar: satir kontaminasyon kumesinde mi, degil mi.

    Bes sinifli makro-F1 deneyin kararini olcmez: C1 yalnizca "gift_given mi"
    diye soruyor, C1b "gift_given/household/received mi". `household <->
    gift_given` karisikligi C1b ekseninde hata DEGILDIR.

    Damitma kapisinin onceden kayitli kurali (DECISIONS 2026-09-14) ogretmenin
    C1 ekseni F1'ini istiyor; o sayi buradan, kodla uretilmis olarak gelir.
    """
    if not ref_labels:
        return {"skipped": True, "reason": "karsilastirilacak satir yok", "n_rows": 0}
    ref = np.asarray([v in positive for v in ref_labels], dtype=bool)
    pred = np.asarray([v in positive for v in pred_labels], dtype=bool)
    w = np.asarray(w, dtype=float)

    nokta = _binary_prf(ref, pred, w)
    rng = np.random.default_rng(seed)
    strata = _strata_index(strata_keys)
    boots = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        boots[b] = _binary_prf(ref[idx], pred[idx], w[idx])
    return {
        "positive_labels": list(positive),
        "n_rows": len(ref_labels),
        "n_reference_positive": int(ref.sum()),
        "precision": {"value": _r(nokta[0]), "ci95": _ci(boots[:, 0])},
        "recall": {"value": _r(nokta[1]), "ci95": _ci(boots[:, 1])},
        "f1": {"value": _r(nokta[2]), "ci95": _ci(boots[:, 2])},
    }


def kendi_cocugu_table(df: pl.DataFrame, tag: str) -> dict | None:
    """`KENDI_COCUGU` isaretli satirlarda insan ve LLM etiketinin dagilimi.

    `household` karari (C1/C1b) kavramsal olarak verildi; bu tablo o sinirin
    insanda ve modelde nerede durdugunu gosterir, karari degistirmez.
    """
    col = f"notes_{tag}"
    if col not in df.columns:
        return None
    flagged = df.filter(pl.col(col) == KENDI_COCUGU)
    if not flagged.height:
        return {"n_flagged": 0}
    ref = f"label_{tag}"
    pairs = dict(
        sorted(
            ((f"{h}->{m}", int(n)) for h, m, n in
             flagged.group_by(ref, "purchase_type").len().iter_rows()),
            key=lambda kv: (-kv[1], kv[0]),
        )
    )
    return {
        "n_flagged": flagged.height,
        "human_label": dict(sorted(flagged.group_by(ref).len().iter_rows())),
        "llm_label": dict(sorted(flagged.group_by("purchase_type").len().iter_rows())),
        "human_to_llm": pairs,
    }


# ------------------------------------------------------------------- figur
def fig_llm_vs_human(
    cfg: Config, per_class: dict, out: Path, *, n_annotators: int = 3
) -> Path:
    """F18 - sinif bazli F1. Baslik OLCUMU yazar, sonuc ilan etmez."""
    viz.apply_style(cfg.get("eda.figure_dpi"))
    fig, ax = viz.plt.subplots(figsize=(7.6, 4.2))

    names = [c for c in LABELS if c in per_class]
    f1 = [(per_class[c]["f1"] or 0) for c in names]
    n = [per_class[c]["n_reference"] for c in names]

    bars = ax.barh(range(len(names)), f1, color=viz.SLOTS[0])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{c}\n(n={m})" for c, m in zip(names, n)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (insan referansina karsi)")
    for bar, value in zip(bars, f1):
        ax.text(min(value + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", fontsize=9)

    referans = (
        "Tek etiketleyicinin (A) etiketi referans alindi; etiket guvenilirligi "
        "OLCULMEDI."
        if n_annotators == 1
        else f"{n_annotators} bagimsiz etiketleyicinin cogunlugu referans alindi."
    )
    viz.titles(
        ax,
        f"F18 · LLM'in sinif bazli F1'i — {sum(n)} satir",
        f"{referans} Bu bir KAPI degil olcumdur: F1 icin hicbir esik sabitlenmedi. "
        "Agirliksiz; dogrulama seti yaygınlık ornegi degildir - `household` ve "
        "`received` bilerek fazla temsil edildi.",
    )
    ax.grid(axis="y", visible=False)
    return viz.save(fig, out, log)


# --------------------------------------------------------------------- kosu
def evaluate(cfg: Config, n: int | None = None) -> dict:
    n = int(n if n is not None else cfg.get("validation.n"))
    tags, mode = annotator_mode(cfg)
    df = load_labels(cfg, n)
    cols = label_columns(df)
    min_class_n = int(cfg.get("validation.min_class_n_for_kappa"))
    threshold = float(cfg.get("validation.kappa_min"))
    n_boot = int(cfg.get("validation.bootstrap_n"))
    seed = int(cfg.get("seed"))

    kappa = None
    if mode == "fleiss":
        kappa = kappa_report(df, cols, min_class_n=min_class_n)
        # Karsilastirma YUVARLANMAMIS deger uzerinden. Bkz. `kappa_report`.
        exact = kappa["kappa_exact"]
        agreement = {
            **kappa,
            "threshold": threshold,
            "passed": None if exact is None else bool(exact >= threshold),
        }
    else:
        # Tek etiketleyici: uyum OLCULEMEZ. `passed: None` -> INCOMPLETE.
        # PASS yazmak olculmemis bir seyi gecmis gibi gosterirdi; FAIL yazmak
        # ise olculmemis bir seyi basarisiz ilan ederdi. Ikisi de yanlis.
        agreement = {
            "skipped": True,
            "passed": None,
            "reason": (
                f"tek etiketleyici ({', '.join(tags)}); etiket guvenilirligi "
                "olculmedi - kullanici karari 2026-09-14 (DECISIONS)"
            ),
            "design": {
                "planned_annotators": 3,
                "planned_statistic": "Fleiss kappa",
                "threshold": threshold,
            },
        }

    df = df.with_columns(reference_labels(df, cols))
    # `tie` satirlari referans olamaz: uzlasi yok demek dogru cevap yok demek.
    # Tek etiketleyicide tie olusmaz.
    usable = df.filter(pl.col("reference").is_not_null() & (pl.col("reference") != TIE))

    vs_llm = compare(usable["reference"], usable["purchase_type"])
    vs_proxy = compare(
        _binary_gift(usable["reference"], "reference"),
        usable.select(_proxy_labels(usable))["proxy"],
    )

    # --- onceden kayitli olcumler (2026-09-14)
    sample_ci = scored_measurement(
        usable["reference"].to_list(), usable["purchase_type"].to_list(),
        np.ones(usable.height),
        list(zip(usable["category"].to_list(), usable["purchase_type"].to_list())),
        n_boot=n_boot, seed=seed, min_class_n=min_class_n,
    )
    sample_ci["note"] = (
        "AGIRLIKSIZ - dogrulama LLM etiketine gore dengesiz cekildi; "
        "populasyon duzeyi icin `llm_vs_human_main_weighted`"
    )

    main_rows = usable.filter(pl.col("sample_frame") == FRAME_MAIN)
    kategoriler = sorted(set(main_rows["category"].to_list()))
    pop = population_counts(cfg, kategoriler)
    eksik = [c for c in kategoriler if not any(key[0] == c for key in pop)]
    w = None
    main_keys = list(zip(main_rows["category"].to_list(), main_rows["purchase_type"].to_list()))
    if not main_rows.height:
        weighted: dict = {"skipped": True, "reason": "dogrulamada `main` satiri yok"}
    elif eksik:
        weighted = {"skipped": True, "reason": f"populasyon sayilari okunamadi: {eksik}"}
    else:
        w, meta = ipw_weights(
            main_rows["category"].to_list(), main_rows["purchase_type"].to_list(), pop,
            min_cell_n=int(cfg.get("validation.min_cell_n")),
        )
        weighted = {
            **scored_measurement(
                main_rows["reference"].to_list(), main_rows["purchase_type"].to_list(), w,
                main_keys, n_boot=n_boot, seed=seed, min_class_n=min_class_n,
            ),
            "population": "dort kategorinin `main` cercevesi (kategori basina esit)",
            "weights": meta,
        }

    # Deneyin ikili kararlari (C1, C1b). Tanim `detection.schema.CONTAMINATION`.
    eksenler: dict[str, dict] = {"sample": {}, "main_weighted": {}}
    for ad, etiketler in CONTAMINATION.items():
        kosul = CONDITION_OF[ad]
        eksenler["sample"][kosul] = binary_axis_measurement(
            usable["reference"].to_list(), usable["purchase_type"].to_list(),
            np.ones(usable.height),
            list(zip(usable["category"].to_list(), usable["purchase_type"].to_list())),
            etiketler, n_boot=n_boot, seed=seed,
        )
        if w is not None:
            eksenler["main_weighted"][kosul] = binary_axis_measurement(
                main_rows["reference"].to_list(), main_rows["purchase_type"].to_list(),
                w, main_keys, etiketler, n_boot=n_boot, seed=seed,
            )

    kendi = {t: kendi_cocugu_table(df, t) for t in tags}

    limitations = []
    if mode == "none":
        emin_degil = sum(
            int((df[f"notes_{t}"] == "EMIN_DEGIL").sum())
            for t in tags if f"notes_{t}" in df.columns
        )
        limitations = [
            "Referans tek etiketleyicinin yargisi; etiket guvenilirligi olculmedi.",
            f"Belirsizlik isareti (EMIN_DEGIL) kullanilan satir: {emin_degil}.",
            "Agirliksiz olcumler katmanlama nedeniyle household/gift_given yonunde "
            "yanli; populasyon icin agirlikli olcum okunmali.",
        ]

    report = {
        "meta": {
            "n": n,
            "annotators": tags,
            "n_annotators": len(cols),
            "reliability": mode,
            "model": df["model"][0] if "model" in df.columns else None,
            "prompt_version": df["prompt_version"][0] if "prompt_version" in df.columns else None,
            "thresholds_fixed": "2026-08-29, configs/base.yaml -> validation",
            "methods_registered": "2026-09-14, DECISIONS (A modelle karsilastirilmadan once)",
        },
        # TEK kapi olcutu. Digerleri olcum.
        "criteria": {"1_annotator_agreement": agreement},
        "measurements": {
            "n_tie": int((df["reference"] == TIE).sum()),
            "n_usable": usable.height,
            "reference_distribution": dict(
                sorted(usable.group_by("reference").len().iter_rows())
            ),
            # DIKKAT: bunlar KAPI DEGIL. F1 icin hicbir esik sabitlenmedi ve
            # sonucu gordukten sonra esik uydurmak kapiyi sonradan kurmaktir.
            "llm_vs_human": vs_llm,
            "llm_vs_human_sample_ci": sample_ci,
            "llm_vs_human_main_weighted": weighted,
            # Deneyin kullandigi ikili kararlar. Damitma kapisi ogretmenin C1
            # F1'ini buradan (`sample`) okur - onceden kayitli kural.
            "contamination_axes": eksenler,
            # Vekilin ILK bagimsiz olcumu. Onceki 0,58/0,61 yazar destekliydi.
            "proxy_vs_human_gift_only": vs_proxy,
            "kendi_cocugu": kendi,
        },
        "limitations": limitations,
        "verdict": (
            "PASS" if agreement["passed"]
            else "INCOMPLETE" if agreement["passed"] is None
            else "FAIL"
        ),
    }

    if vs_llm["per_class"]:
        fig_llm_vs_human(cfg, vs_llm["per_class"], figure_path(cfg, n), n_annotators=len(cols))
    write_json(report, validation_report_path(cfg, n), log)

    log.info("HAFTA 4 -> %s", report["verdict"])
    if kappa is not None:
        log.info("  Fleiss kappa      %s (esik %.2f, %d satir)",
                 "hesaplanamadi" if kappa["kappa"] is None else f"{kappa['kappa']:.4f}",
                 threshold, kappa["n_rows_in_kappa"])
        if kappa["sparse_classes"]:
            log.info("  kappa disi sinif  %s (n < %d) - %d satir dusuruldu",
                     ", ".join(kappa["sparse_classes"]), kappa["min_class_n"],
                     kappa["n_rows_dropped_as_sparse"])
    else:
        log.info("  uyum              OLCULMEDI - %s", agreement["reason"])
    log.info("  LLM dogrulugu     %s (macro F1 %s) - agirliksiz, OLCUM",
             vs_llm["accuracy"], vs_llm["macro_f1"])
    if not weighted.get("skipped"):
        log.info("  agirlikli (main)  dogruluk %s  macro F1 %s",
                 weighted["accuracy"], weighted["macro_f1"])
    for duzey, tablo in eksenler.items():
        for kosul, m in tablo.items():
            if not m.get("skipped"):
                log.info("  eksen %-4s %-14s K=%s D=%s F1=%s %s", kosul, duzey,
                         m["precision"]["value"], m["recall"]["value"],
                         m["f1"]["value"], m["f1"]["ci95"])
    log.info("  vekil (gift ekseni) %s", vs_proxy["per_class"].get(GIFT))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=None, help="Dogrulama seti boyutu")
    args = parser.parse_args(argv)

    evaluate(Config.load(args.config), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
