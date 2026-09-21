"""POST-HOC saglamlik (2026-09-21). Ana ortamda (.venv) kosar; RecBole GEREKTIRMEZ.

Girdi : reports/results/experiment_<slug>_<kosul>_<model>_seed<N>.json   (72 kosu raporu)
        reports/results/experiment_stats.json                             (kullanici GA'lari)
        data/processed/recbole/<slug>/<kosul>.inter                       (yalnizca 2. analiz)
        data/processed/recbole/<slug>/peruser/<kosul>_<model>_seed<N>.parquet
Cikti : reports/results/robustness_posthoc.json

BU MODUL ONCEDEN KAYITLI DEGIL. Iki analiz 2026-09-21 denetiminde, 72 kosunun sonuclari
gorulduKTEN SONRA tanimlandi (DECISIONS 2026-09-21 on kaydi). Hicbir birincil sayinin
yerine gecmez; cikti `post_hoc: true` damgasi tasir. Yorum kurali da yok: sayilar
okunur, karar verilmez.

1. SEED DUZEYI. `experiment_stats`in esli bootstrap'i KULLANICILARI yeniden ornekliyor:
   GA "hangi kullanicilar" belirsizligini olcer, "hangi egitim kosusu" belirsizligini
   degil. Burada onceden kayitli her karsitligin her seed'deki farki (kosu raporunun
   kendi test ortalamasindan), ortalamasi, sd'si, isaret uyumu ve sd/sqrt(n_seed)
   degeri kullanici GA'sinin yari genisligiyle yan yana yazilir.

2. MARUZIYET AYRISTIRMASI. Plasebo (C4), C1 kadar satiri KORPUS duzeyinde siler;
   kullanici duzeyinde esleme yok. C1 hediye veren kullanicilarin satirlarini yogun,
   C4 herkesin satirini seyrek siler - iki kosul farkli sayida kullanicinin gecmisine
   dokunur. Test kullanicilari "tedavi (C1) bu kullanicinin bir egitim satirini sildi mi"
   x "plasebo (C4) sildi mi" diye dort gruba ayrilir ve her grupta esli bootstrap
   yapilir (C1b x C4b icin de ayni). Iki kosulun da DOKUNMADIGI kullanicinin kendi
   gecmisi iki kosulda da aynidir: oradaki fark yalnizca modelin ogrendiginden gelir.

Kullanim:
    python -m gift_contamination.analysis.robustness --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, resolve_roles
from ..recsys.atomic import SPLIT_TRAIN
from ..recsys.conditions import REPORTABLE_LABEL_SOURCES, condition_path
from ..utils.io import code_version, read_json, write_json
from ..utils.logging import get_logger
from ..utils.seeding import seed_for
from . import experiment_stats as es

log = get_logger("analysis.robustness")

# Seed duzeyi: experiment_stats'in karsitliklari + Kapi 2'nin plasebo karsitliklari.
SEED_CONTRASTS = tuple((a, b) for a, b, _, _ in es.CONTRASTS) + es.PLACEBO_PAIRS
# Maruziyet: yalnizca plasebo karsitliklari (sorulan soru onlarin eslemesiyle ilgili).
EXPOSURE_PAIRS = tuple((a, b) for a, b, _, plasebo in es.CONTRASTS if plasebo)
# (grup adi, tedavi dokundu mu, plasebo dokundu mu)
GROUPS = (("neither", False, False), ("placebo_only", False, True),
          ("treatment_only", True, False), ("both", True, True))


def output_path(cfg: Config) -> Path:
    return cfg.path("results", "robustness_posthoc.json")


# ------------------------------------------------------------------ 1. seed duzeyi
def _user_ci(stats: dict, slug: str, model: str, a: str, b: str, metric: str) -> list | None:
    """experiment_stats.json'daki kullanici GA'si (Kapi 2 karsitliklari orada yok -> None)."""
    blok = stats.get("results", {}).get(f"{slug}/{model}", {}).get("contrasts", {}).get(f"{a}-{b}", {})
    return blok.get("metrics", {}).get(metric, {}).get("ci")


def seed_level(runs: dict, a: str, b: str, metric: str, user_ci: list | None) -> dict:
    """Bir karsitligin seed basina farki. `runs`: (kosul, seed) -> kosu raporu."""
    seeds = es.common_seeds(runs, a, b)
    if not seeds:
        return {"skipped": "iki kosulun ortak seed'i yok"}
    farklar = {s: float(runs[(a, s)]["test"][metric] - runs[(b, s)]["test"][metric]) for s in seeds}
    v = np.array(list(farklar.values()), dtype=np.float64)
    ort = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else None
    isaret = int(np.sum(np.sign(v) == np.sign(ort))) if ort else 0
    out = {
        "seeds": seeds,
        "per_seed": {str(s): d for s, d in farklar.items()},
        "mean": ort,
        "sd": sd,
        "sd_over_sqrt_n": sd / math.sqrt(len(v)) if sd is not None else None,
        "sign_agreement": f"{isaret}/{len(v)}",
        "user_bootstrap_ci": user_ci,
        "user_ci_half_width": (user_ci[1] - user_ci[0]) / 2 if user_ci else None,
    }
    return out


# ------------------------------------------------------------------ 2. maruziyet
def train_rows_per_user(cfg: Config, role: str, code: str) -> pl.DataFrame:
    """(user_id, n): kosulun EGITIM satiri sayisi. Satiri kalmayan kullanici cikti da yok."""
    yol = condition_path(cfg, role, code)
    if not yol.exists():
        raise FileNotFoundError(f"{yol.name} yok - kosul dosyalari data/processed/recbole altinda olmali")
    return (pl.scan_csv(yol, separator="\t",
                        schema_overrides={"user_id:token": pl.String, "item_id:token": pl.String})
            .filter(pl.col("split") == SPLIT_TRAIN)
            .group_by("user_id:token").len()
            .rename({"user_id:token": "user_id", "len": "n"})
            .collect())


def touched(c0: pl.DataFrame, cond: pl.DataFrame, name: str) -> pl.DataFrame:
    """(user_id, <name>): kosul bu kullanicinin en az bir egitim satirini sildi mi."""
    return (c0.join(cond.rename({"n": "_n"}), on="user_id", how="left")
            .select("user_id", (pl.col("n") > pl.col("_n").fill_null(0)).alias(name)))


def exposure(cfg: Config, role: str, model: str, runs: dict, a: str, b: str, metric: str,
             train_counts: dict[str, pl.DataFrame]) -> dict:
    seeds = es.common_seeds(runs, a, b, "C0")
    if not seeds:
        return {"skipped": "C0 ve iki kosulun ortak seed'i yok"}
    x = es.peruser_means(cfg, role, a, model, seeds, [metric])
    y = es.peruser_means(cfg, role, b, model, seeds, [metric])
    z = es.peruser_means(cfg, role, "C0", model, seeds, [metric])
    bayrak = (touched(train_counts["C0"], train_counts[a], "_ta")
              .join(touched(train_counts["C0"], train_counts[b], "_tb"), on="user_id"))
    d = (x.rename({metric: "_a"}).join(y.rename({metric: "_b"}), on="user_id")
         .join(z.rename({metric: "_c0"}), on="user_id")
         .join(bayrak, on="user_id", how="left")
         # Egitim satiri hic olmayan test kullanicisi C0'da da yok sayilmaz: 5-core
         # her kullaniciya >= 3 egitim satiri birakir, yoksa bolme bozuktur.
         .sort("user_id"))
    if d["_ta"].null_count() or d["_tb"].null_count():
        raise RuntimeError(f"{role}/{model}: C0 egitiminde satiri olmayan test kullanicisi var")
    if d.height != x.height:
        raise RuntimeError(f"{role}/{model}: {a}, {b} ve C0'in kullanici kumeleri farkli")
    n_boot = int(cfg.get("experiment.bootstrap_iters"))
    ci = float(cfg.get("experiment.ci"))
    base = int(cfg.get("seed"))
    gruplar = {}
    for ad, ta, tb in GROUPS:
        g = d.filter((pl.col("_ta") == ta) & (pl.col("_tb") == tb))
        if g.is_empty():
            gruplar[ad] = {"n_users": 0, "skipped": "grup bos"}
            continue
        r = es.paired_bootstrap(g["_a"].to_numpy(), g["_b"].to_numpy(), names=[metric],
                                n_boot=n_boot, ci=ci,
                                seed=seed_for(base, f"posthoc/exposure/{role}/{model}/{a}-{b}/{ad}"))[metric]
        gruplar[ad] = {"n_users": g.height, "share_of_test_users": round(g.height / d.height, 4),
                       "mean_c0": float(g["_c0"].mean()), **r, "direction": es.direction(r["ci"])}
    return {"treatment": a, "placebo": b, "seeds": seeds, "n_test_users": d.height,
            "metric": metric, "groups": gruplar}


# ------------------------------------------------------------------ kosu
def evaluate(cfg: Config, roles: list[str], *, with_exposure: bool = True) -> dict:
    metric = es.metric_key(cfg.get("gate2.metric"))
    stats_yolu = es.stats_path(cfg)
    stats = read_json(stats_yolu) if stats_yolu.exists() else {}
    seed_out: dict = {}
    exp_out: dict = {}
    kaynaklar: set = set()
    for role in roles:
        slug = cfg.category_slug(role)
        sayim = None
        for model in cfg.get("experiment.models"):
            anahtar = f"{slug}/{model}"
            runs = es.load_runs(cfg, role, model)
            if not runs:
                seed_out[anahtar] = {"skipped": "hic kosu yok"}
                continue
            kaynaklar |= {r["meta"].get("label_source") for r in runs.values()}
            seed_out[anahtar] = {
                f"{a}-{b}": seed_level(runs, a, b, metric, _user_ci(stats, slug, model, a, b, metric))
                for a, b in SEED_CONTRASTS
            }
            if not with_exposure:
                continue
            if sayim is None:
                gerekli = {"C0", *(c for pair in EXPOSURE_PAIRS for c in pair)}
                sayim = {c: train_rows_per_user(cfg, role, c) for c in sorted(gerekli)}
            exp_out[anahtar] = {f"{a}-{b}": exposure(cfg, role, model, runs, a, b, metric, sayim)
                                for a, b in EXPOSURE_PAIRS}
            log.info("%s: maruziyet ayristirmasi bitti", anahtar)
    raporlanabilir = bool(kaynaklar) and kaynaklar <= REPORTABLE_LABEL_SOURCES
    return {
        "meta": {
            "post_hoc": True,
            "defined": "2026-09-21, 72 kosunun sonuclari goruldukten SONRA (DECISIONS 2026-09-21 on kaydi)",
            "replaces_primary": False,
            "code_version": code_version(),
            "label_sources": sorted(str(k) for k in kaynaklar),
            "reportable": raporlanabilir,
            "metric": metric,
            "bootstrap_iters": int(cfg.get("experiment.bootstrap_iters")),
            "ci": float(cfg.get("experiment.ci")),
            "note": ("Seed duzeyi: kosu raporlarinin test ortalamalari. Maruziyet: test "
                     "kullanicilari tedavinin/plasebonun egitim satirina dokunup dokunmadigina "
                     "gore dort gruba ayrildi; her grupta esli bootstrap."),
        },
        "seed_level": seed_out,
        "exposure": exp_out if with_exposure else {"skipped": "--no-exposure"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default=None, help="Bos birakilirsa `experiment.categories`.")
    parser.add_argument("--no-exposure", action="store_true",
                        help="Yalnizca seed duzeyi (veri dosyasi gerektirmez).")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    roles = resolve_roles(cfg, args.category) if args.category else list(cfg.get("experiment.categories"))
    write_json(evaluate(cfg, roles, with_exposure=not args.no_exposure), output_path(cfg), log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
