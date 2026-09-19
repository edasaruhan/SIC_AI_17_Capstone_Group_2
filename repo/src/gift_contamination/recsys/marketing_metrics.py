"""Pazarlama metrikleri M1 / M2 / M3 (Faz 5). Ana ortamda (.venv) kosar; yeniden egitim YOK.

Girdi : data/processed/recbole/<slug>/C0.inter                    (gercek gecmis + etiket)
        data/interim/<slug>_meta.parquet                           (urun alt kategorisi)
        data/processed/recbole/<slug>/peruser/<kosul>_<model>_seed<N>.parquet (top-K listeleri)
Cikti : reports/results/marketing_metrics.json                    (yalnizca toplamlar)

TANIMLAR ONCEDEN KAYITLI (DECISIONS 2026-09-14 "Faz 5 kodu"), deney sonucu gorulmeden:

  gecmis H_u   : C0'daki train + valid satirlari (test urunu gecmis degil). Gecmis HER
                 kosulda C0'in - kosul modelin ne gordugunu degistirir, kullanicinin ne
                 aldigini degil.
  alt kategori : metadata `categories[marketing.subcategory_level]` (Toys: "Puzzles",
                 Grocery: "Beverages"); yoksa null ve hicbir kumeye girmez.
  hediye-yalniz alt kategori S_u : H_u'da o alt kategoride en az bir hediye satiri olan ve
                 HEDIYE OLMAYAN hicbir satiri olmayan alt kategoriler. Hediye = eksenin
                 etiket kumesi (`narrow` = gift_given, `broad` = + household + received).

  M1 israf orani : top-K'nin (K = experiment.topk[0]) S_u'dan gelen payi, S_u bos olmayan
                   kullanicilarda. Karsitliklar (pozitif = kirli liste daha cok israf):
                   dar: C0-C1, C4-C1, C0-C3 · genis: C0-C1b, C4b-C1b.
  M2 yari omur   : son hediye g'nin alt kategorisi hediye-yalnizsa, g'den SONRAKI `self`
                   satir sayisi n. Fazla pay e_u = pay_C0(S_g) - pay_C1(S_g). Kova ortalamasi
                   e(n), n = 0..max-1, e(n) = A*exp(-lambda*n) agirlikli uyum; yari omur =
                   ln2/lambda etkilesim; haftaya ceviri = medyan ardisik etkilesim araligi.
                   A <= 0 ya da uyum olmazsa yari omur TANIMSIZ yazilir (uydurulmaz).
  M3 segment     : C1-C4 ve C1b-C4b kullanici basi metrik farki, |H_u|'nun ucte birlik
                   dilimlerinde (kesim noktalari quantile 1/3, 2/3; esit olan alt dilime).

Kullanici basi metrik ve pay once seed'ler uzerinden ortalanir; GA'lar kullanicilar
yeniden orneklenerek (`experiment.bootstrap_iters`, `experiment.ci`).

Kullanim:
    python -m gift_contamination.recsys.marketing_metrics --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from ..analysis import experiment_stats as es
from ..config import Config, resolve_roles
from ..data.metadata import meta_parquet_path
from ..detection.contamination import CONTAMINATION
from ..utils.io import code_version, read_json, write_json
from ..utils.logging import get_logger
from .atomic import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from .run_experiment import peruser_path, read_condition

log = get_logger("recsys.marketing_metrics")

SELF = "self"
# (a, b, eksen): a - b. Pozitif = a'nin listesi hediye-yalniz alt kategorilere daha cok slot veriyor.
M1_CONTRASTS = (
    ("C0", "C1", "narrow"), ("C4", "C1", "narrow"), ("C0", "C3", "narrow"),
    ("C0", "C1b", "broad"), ("C4b", "C1b", "broad"),
)
M2_PAIR = ("C0", "C1")
# M2'nin birincil degeri sirayi kullanan modelden gelir (DECISIONS "Faz 5 kodu").
SEQUENCE_AWARE_MODELS = frozenset({"SASRec", "GRU4Rec"})
M3_CONTRASTS = (("C1", "C4"), ("C1b", "C4b"))
SEGMENTS = ("short", "medium", "long")
# `preprocess.detect_timestamp_unit` ile ayni esik: medyan > 1e11 ise milisaniye.
MS_THRESHOLD = 1e11


# ------------------------------------------------------------------ girdiler
def subcategories(cfg: Config, role: str) -> pl.DataFrame:
    """item_id -> alt kategori, kategorinin BUTUN urunleri.

    Gecmisteki urunlerle sinirlamak yanlis olurdu: top-K'daki bir urun gecmiste
    gecmiyorsa alt kategorisi bos kalir ve pay sessizce DUSUK sayilirdi.
    """
    seviye = int(cfg.get("marketing.subcategory_level"))
    meta = pl.read_parquet(meta_parquet_path(cfg, role), columns=["parent_asin", "categories"])
    return meta.select(pl.col("parent_asin").alias("item_id"),
                       pl.col("categories").list.get(seviye, null_on_oob=True).alias("subcat"))


def history(c0: pl.DataFrame) -> pl.DataFrame:
    """C0'in train+valid satirlari, kullanici icinde dosya (kronolojik) sirasinda."""
    return (c0.with_row_index("_ord")
            .filter(pl.col("split").is_in([SPLIT_TRAIN, SPLIT_VALID]))
            .select("_ord", pl.col("user_id").cast(pl.String), pl.col("item_id").cast(pl.String),
                    "timestamp", "label"))


def gift_only_subcats(hist: pl.DataFrame, labels: tuple[str, ...]) -> pl.DataFrame:
    """(user_id, subcat): hediye satiri olan ve hediye olmayan satiri OLMAYAN alt kategoriler."""
    return (hist.drop_nulls("subcat")
            .group_by("user_id", "subcat")
            .agg(pl.col("label").is_in(list(labels)).any().alias("_g"),
                 (~pl.col("label").is_in(list(labels))).any().alias("_ng"))
            .filter(pl.col("_g") & ~pl.col("_ng"))
            .select("user_id", "subcat"))


def topk_subcat_share(pu: pl.DataFrame, targets: pl.DataFrame, sub: pl.DataFrame, k: int) -> pl.DataFrame:
    """Kullanici basi: top-k'nin `targets` (user_id, subcat) kumesinden gelen payi."""
    hedef = targets.with_columns(pl.lit(True).alias("_hit"))
    return (pu.select("user_id", pl.col("topk_items").list.head(k).alias("item_id"))
            .explode("item_id")
            .join(sub, on="item_id", how="left")
            .join(hedef, on=["user_id", "subcat"], how="left")
            .group_by("user_id")
            .agg((pl.col("_hit").fill_null(False).sum() / k).alias("share")))


def seed_mean_share(cfg, role, code, model, seeds, targets, sub, k) -> pl.DataFrame:
    parcalar = []
    for seed in seeds:
        pu = pl.read_parquet(peruser_path(cfg, role, code, model, seed), columns=["user_id", "topk_items"])
        parcalar.append(topk_subcat_share(pu, targets, sub, k))
    return pl.concat(parcalar).group_by("user_id").agg(pl.col("share").mean()).sort("user_id")


# ------------------------------------------------------------------ M2 uyumu
# Uyumun lambda alt siniri. Yari omur en fazla ln2 / LAMBDA_MIN olabilir; bu degere dayanan
# bir bootstrap ucu GA'nin ust ucunun SINIRSIZ oldugunu gosterir, bir sure olcumu degil
# (F22 lejanti "∞" yazar).
LAMBDA_MIN = 1e-6


def fit_half_life(n: np.ndarray, excess: np.ndarray, weights: np.ndarray) -> dict:
    """e(n) = A*exp(-lambda*n), agirlikli en kucuk kareler. A<=0 ya da uyum yoksa tanimsiz."""
    from scipy.optimize import curve_fit  # noqa: PLC0415

    n = np.asarray(n, dtype=np.float64)
    e = np.asarray(excess, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(n) < 3 or e[0] <= 0:
        return {"A": None, "lambda": None, "half_life": None, "reason": "en az 3 kova ve pozitif baslangic fazlasi gerekli"}
    try:
        (a, lam), _ = curve_fit(lambda x, a, lam: a * np.exp(-lam * x), n, e, p0=(e[0], 0.5),
                                sigma=1 / np.sqrt(w), bounds=([0, LAMBDA_MIN], [1, 50]), maxfev=10_000)
    except (RuntimeError, ValueError) as exc:
        return {"A": None, "lambda": None, "half_life": None, "reason": f"uyum olmadi: {exc}"}
    return {"A": float(a), "lambda": float(lam), "half_life": float(math.log(2) / lam)}


def _bucket_means(bucket: np.ndarray, values: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    toplam = np.bincount(bucket, weights=values, minlength=n_max + 1)
    adet = np.bincount(bucket, minlength=n_max + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return toplam / adet, adet


def median_gap_days(hist: pl.DataFrame) -> float | None:
    """Kullanici basi ardisik etkilesim araliginin medyani, kullanicilar uzerinden medyan (gun)."""
    if hist.is_empty():
        return None
    ts = hist["timestamp"]
    birim_ms = 1.0 if float(ts.median()) > MS_THRESHOLD else 1000.0
    araliklar = (hist.sort("user_id", "_ord")
                 .with_columns((pl.col("timestamp").diff().over("user_id") * birim_ms / 86_400_000).alias("_gap"))
                 .drop_nulls("_gap")
                 .group_by("user_id").agg(pl.col("_gap").median()))
    return float(araliklar["_gap"].median()) if araliklar.height else None


# ------------------------------------------------------------------ hesap
def evaluate_role_model(cfg: Config, role: str, model: str, runs: dict, c0: pl.DataFrame,
                        sub_all: pl.DataFrame) -> dict:
    k = int(cfg.get("experiment.topk")[0])
    n_boot = int(cfg.get("experiment.bootstrap_iters"))
    ci = float(cfg.get("experiment.ci"))
    base = int(cfg.get("seed"))
    slug = cfg.category_slug(role)
    hist = history(c0).join(sub_all, on="item_id", how="left")
    test_users = (c0.filter((pl.col("split") == SPLIT_TEST) & (pl.col("label") == SELF))
                  .select(pl.col("user_id").cast(pl.String)).unique())
    out: dict = {}

    # ---- M1
    m1: dict = {}
    for a, b, eksen in M1_CONTRASTS:
        anahtar = f"{a}-{b}"
        seeds = es._common_seeds(runs, a, b)
        if not seeds:
            m1[anahtar] = {"axis": eksen, "skipped": "iki kosulun ortak seed'i yok"}
            continue
        hedef = gift_only_subcats(hist, CONTAMINATION[eksen])
        maruz = hedef.select("user_id").unique().join(test_users, on="user_id")
        x = seed_mean_share(cfg, role, a, model, seeds, hedef, sub_all, k).join(maruz, on="user_id").sort("user_id")
        y = seed_mean_share(cfg, role, b, model, seeds, hedef, sub_all, k).join(maruz, on="user_id").sort("user_id")
        xa, ya = es._aligned(x, y, ["share"])
        if not len(xa):
            m1[anahtar] = {"axis": eksen, "skipped": "hediye-yalniz alt kategorisi olan test kullanicisi yok"}
            continue
        s = es.paired_bootstrap(xa, ya, names=["share"], n_boot=n_boot, ci=ci,
                                seed=es._seed_for(base, f"m1/{role}/{model}/{anahtar}"))["share"]
        m1[anahtar] = {"axis": eksen, "seeds": seeds, "n_exposed_users": len(xa),
                       "n_test_users": test_users.height, **s, "direction": es.direction(s["ci"])}
    out["M1_waste_share"] = m1

    # ---- M2
    a, b = M2_PAIR
    seeds = es._common_seeds(runs, a, b)
    n_max = int(cfg.get("marketing.m2_max_self_after_gift"))
    min_users = int(cfg.get("marketing.m2_min_users_per_bucket"))
    if not seeds:
        out["M2_half_life"] = {"skipped": "C0 ve C1'in ortak seed'i yok"}
    else:
        dar = list(CONTAMINATION["narrow"])
        son_hediye = (hist.filter(pl.col("label").is_in(dar)).group_by("user_id")
                      .agg(pl.col("_ord").max().alias("_g_ord"))
                      .join(hist.select("user_id", "_ord", "subcat"), left_on=["user_id", "_g_ord"],
                            right_on=["user_id", "_ord"]))
        sonra = (hist.join(son_hediye.select("user_id", "_g_ord"), on="user_id")
                 .filter((pl.col("_ord") > pl.col("_g_ord")) & (pl.col("label") == SELF))
                 .group_by("user_id").agg(pl.len().alias("n_self_after")))
        kume = (son_hediye.drop_nulls("subcat")
                .join(gift_only_subcats(hist, CONTAMINATION["narrow"]), on=["user_id", "subcat"], how="semi")
                .join(test_users, on="user_id", how="semi")
                .join(sonra, on="user_id", how="left")
                .with_columns(pl.col("n_self_after").fill_null(0))
                .sort("user_id"))
        hedef = kume.select("user_id", "subcat")
        pa = seed_mean_share(cfg, role, a, model, seeds, hedef, sub_all, k).join(kume.select("user_id"), on="user_id").sort("user_id")
        pb = seed_mean_share(cfg, role, b, model, seeds, hedef, sub_all, k).join(kume.select("user_id"), on="user_id").sort("user_id")
        if pa.height != kume.height or not pa["user_id"].equals(kume["user_id"]):
            raise RuntimeError("M2: top-K dosyasinda olmayan test kullanicisi var - dosyalar karismis")
        xa, ya = es._aligned(pa, pb, ["share"])
        n_u = kume["n_self_after"].to_numpy()
        kova = np.minimum(n_u, n_max)
        fazla = (xa - ya).ravel()
        ort, adet = _bucket_means(kova, fazla, n_max)
        ort_a, _ = _bucket_means(kova, xa.ravel(), n_max)
        ort_b, _ = _bucket_means(kova, ya.ravel(), n_max)
        # Uyuma yalnizca TEK bir n'yi temsil eden ve yeterince kalabalik kovalar girer
        # (en ust kova ">= n_max" birlesik, uyuma girmez).
        uygun = np.array([i for i in range(n_max) if adet[i] >= min_users])
        fit = (fit_half_life(uygun, ort[uygun], adet[uygun]) if len(uygun)
               else {"half_life": None, "reason": "yeterli kova yok"})
        boot_hl = []
        if fit.get("half_life") is not None:
            rng = np.random.default_rng(es._seed_for(base, f"m2/{role}/{model}"))
            for _ in range(n_boot):
                idx = rng.integers(0, len(fazla), size=len(fazla))
                bo, badet = _bucket_means(kova[idx], fazla[idx], n_max)
                bu = np.array([i for i in range(n_max) if badet[i] >= min_users])
                if len(bu):
                    f = fit_half_life(bu, bo[bu], badet[bu])
                    if f.get("half_life") is not None:
                        boot_hl.append(f["half_life"])
        gap = median_gap_days(hist.join(test_users, on="user_id", how="semi"))
        hl = fit.get("half_life")
        out["M2_half_life"] = {
            "pair": f"{a}-{b}", "seeds": seeds, "n_users": len(fazla), "k": k,
            "buckets": [{"n_self_after_gift": (f">={n_max}" if i == n_max else i), "n_users": int(adet[i]),
                         "share_a": _f(ort_a[i]), "share_b": _f(ort_b[i]), "excess": _f(ort[i]),
                         "used_in_fit": bool(i in set(uygun.tolist()))} for i in range(n_max + 1)],
            "fit": fit,
            "half_life_interactions_ci": (es._ci(np.array(boot_hl), ci) if len(boot_hl) >= 0.5 * n_boot else None),
            "bootstrap_fit_success_share": round(len(boot_hl) / n_boot, 4) if hl is not None else None,
            "median_gap_days": gap,
            "half_life_weeks": (hl * gap / 7 if hl is not None and gap is not None else None),
            "caveat": "review tarihi alim tarihi degil; hafta cevirisi yaklasik",
            # BPR gecmisin SIRASINI gormez: n'ye bagli azalma beklenmez, birincil deger degil.
            "sequence_aware_model": model in SEQUENCE_AWARE_MODELS,
        }

    # ---- M3
    uzunluk = (hist.group_by("user_id").agg(pl.len().alias("hist_len"))
               .join(test_users, on="user_id", how="semi").sort("user_id"))
    q1, q2 = (float(v) for v in np.quantile(uzunluk["hist_len"].to_numpy(), [1 / 3, 2 / 3]))
    dilim = uzunluk.with_columns(
        pl.when(pl.col("hist_len") <= q1).then(pl.lit("short"))
        .when(pl.col("hist_len") <= q2).then(pl.lit("medium")).otherwise(pl.lit("long")).alias("segment"))
    names = es.metric_keys(cfg)
    m3: dict = {"cutpoints": [q1, q2],
                "segment_sizes": {s: int(dilim.filter(pl.col("segment") == s).height) for s in SEGMENTS}}
    for a, b in M3_CONTRASTS:
        anahtar = f"{a}-{b}"
        seeds = es._common_seeds(runs, a, b)
        if not seeds:
            m3[anahtar] = {"skipped": "iki kosulun ortak seed'i yok"}
            continue
        x = es.peruser_means(cfg, role, a, model, seeds, names).join(dilim, on="user_id").sort("user_id")
        y = es.peruser_means(cfg, role, b, model, seeds, names).join(dilim, on="user_id").sort("user_id")
        seg_out = {}
        for s in SEGMENTS:
            xs, ys = x.filter(pl.col("segment") == s), y.filter(pl.col("segment") == s)
            if xs.is_empty():
                seg_out[s] = {"skipped": "dilim bos"}
                continue
            xa, ya = es._aligned(xs, ys, names)
            r = es.paired_bootstrap(xa, ya, names=names, n_boot=n_boot, ci=ci,
                                    seed=es._seed_for(base, f"m3/{role}/{model}/{anahtar}/{s}"))
            for v in r.values():
                v["interpretation"] = es.interpret_placebo(v["ci"])
            seg_out[s] = {"n_users": len(xa), "metrics": r}
        m3[anahtar] = {"seeds": seeds, "segments": seg_out}
    out["M3_segments"] = m3
    log.info("%s/%s: M1 %d karsitlik, M2 yari omur %s, M3 %s", slug, model, len(m1),
             out["M2_half_life"].get("fit", {}).get("half_life"), m3["segment_sizes"])
    return out


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def evaluate(cfg: Config, roles: list[str]) -> dict:
    gate_yolu = es.gate2_path(cfg)
    gate = read_json(gate_yolu).get("results", {}) if gate_yolu.exists() else {}
    sonuc: dict = {}
    kaynaklar: set = set()
    for role in roles:
        slug = cfg.category_slug(role)
        c0 = None
        for model in cfg.get("experiment.models"):
            anahtar = f"{slug}/{model}"
            runs = es.load_runs(cfg, role, model)
            if not runs:
                sonuc[anahtar] = {"skipped": "hic kosu yok"}
                continue
            kaynaklar |= {r["meta"].get("label_source") for r in runs.values()}
            if c0 is None:
                c0 = read_condition(cfg, role, "C0")
                sub_all = subcategories(cfg, role)
            verdict = gate.get(anahtar, {}).get("verdict")
            sonuc[anahtar] = {
                "gate2_verdict": verdict,
                "interpretable": verdict == "PASS" if verdict else None,
                **evaluate_role_model(cfg, role, model, runs, c0, sub_all),
            }
    raporlanabilir = bool(kaynaklar) and kaynaklar <= es.REPORTABLE_LABEL_SOURCES
    if not raporlanabilir:
        log.warning("etiket kaynagi %s - DUMAN TESTI, sonuc tablosuna giremez", sorted(map(str, kaynaklar)))
    return {
        "meta": {
            "code_version": code_version(),
            "label_sources": sorted(str(k) for k in kaynaklar),
            "reportable": raporlanabilir,
            "definitions_fixed": "2026-09-14, DECISIONS 'Faz 5 kodu' (deney kosulmadan)",
            "subcategory_level": int(cfg.get("marketing.subcategory_level")),
            "k": int(cfg.get("experiment.topk")[0]),
        },
        "results": sonuc,
    }


def output_path(cfg: Config) -> Path:
    return cfg.path("results", "marketing_metrics.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default=None, help="Bos birakilirsa `experiment.categories`.")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    roles = resolve_roles(cfg, args.category) if args.category else list(cfg.get("experiment.categories"))
    write_json(evaluate(cfg, roles), output_path(cfg), log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
