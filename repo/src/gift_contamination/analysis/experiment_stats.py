"""Kapi 2 + deney karsitliklari (Faz 4). Ana ortamda (.venv) kosar; RecBole GEREKTIRMEZ.

Girdi : reports/results/experiment_<slug>_<kosul>_<model>_seed<N>.json   (run_experiment)
        reports/results/condition_<slug>_<kosul>.json                     (conditions)
        data/processed/recbole/<slug>/peruser/<kosul>_<model>_seed<N>.parquet
Cikti : reports/results/gate2.json
        reports/results/experiment_stats.json

HER SEY ONCEDEN KAYITLI (DECISIONS 2026-09-14). Bu modul yeni bir karsitlik,
esik ya da yorum eklemez; kayitli olanlari hesaplar.

KAPI 2 - kurulum gecerli mi (ETKIYI olcmez), kategori x model basina:
  1. test (kullanici, urun) ciftlerinin ozeti butun kosullarda ve seed'lerde AYNI
  2. her kosulun gercek urun evreni C0'in alt kumesi
  3. plasebo tabani gecmiyor: (C4 - C0) ve (C4b - C0) esli bootstrap GA'sinin alt ucu <= 0
  4. C0 metriginin seed'ler arasi degisim katsayisi < `gate2.seed_cv_max`
Olculemeyen olcut `passed: null`; biri null ise karar INCOMPLETE, asla PASS.

KARSITLIKLAR - kullanici basi metrik once seed'ler uzerinden ortalanir, sonra
kullanici ciftleri uzerinden ESLI bootstrap (kullanicilar yeniden orneklenir; butun
metrikler AYNI orneklemle). Yon ne cikarsa ciksin raporlanir:
  RQ2  C1 - C4 (birincil), C1 - C0      RQ3  C3 - C1, C3 - C0
  saglamlik  C1b - C4b                   doz-yanit  (C1 - C4)_Toys - (C1 - C4)_Grocery
Kapi 2 PASS olmayan kategori x model icin sonuc `interpretable: false` damgasi tasir.

Kullanim:
    python -m gift_contamination.analysis.experiment_stats --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, resolve_roles
from ..recsys.conditions import REPORTABLE_LABEL_SOURCES, condition_report_path
from ..recsys.run_experiment import METRIC_ALIASES, experiment_path, peruser_path
from ..utils.io import code_version, read_json, write_json
from ..utils.logging import get_logger
from ..utils.seeding import seed_for

log = get_logger("analysis.experiment_stats")

PLACEBO_PAIRS = (("C4", "C0"), ("C4b", "C0"))
# Kapi 2'nin 1. ve 2. olcutu icin en az bu kosullar kosmus olmali (kesilmez cekirdek).
GATE2_CORE = ("C0", "C1", "C4", "C1b", "C4b")
# (a, b, soru, plasebo karsitligi mi) - a - b. Plasebo karsitliklarina onceden
# kayitli yorum kurali uygulanir.
CONTRASTS = (
    ("C1", "C4", "RQ2", True),
    ("C1", "C0", "RQ2", False),
    ("C3", "C1", "RQ3", False),
    ("C3", "C0", "RQ3", False),
    ("C1b", "C4b", "saglamlik", True),
)
# Doz-yanit: plasebo karsitliginin yuksek ve dusuk hediyeli kategorideki farki.
DOSE_ROLES = ("high", "low")
# Etiket hassasiyeti sonuc tablosunun yaninda durur (yorum kurali).
AXIS_OF = {("C1", "C4"): "C1", ("C1b", "C4b"): "C1b"}
BOOT_CELLS = 5_000_000   # bir bootstrap grubunda en fazla bu kadar hucre (bellek)


# ------------------------------------------------------------------- yardimci
def metric_key(name: str) -> str:
    """Config adi -> kullanici basi kolon adi: 'Recall@10' -> 'recall@10', 'HR@10' -> 'hit@10'."""
    ad, k = name.split("@")
    return f"{METRIC_ALIASES.get(ad, ad).lower()}@{int(k)}"


def metric_keys(cfg: Config) -> list[str]:
    return [metric_key(f"{m}@{k}") for m in cfg.get("experiment.metrics")
            for k in cfg.get("experiment.topk")]


def bootstrap_ci(boot: np.ndarray, ci: float) -> list[float]:
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2], axis=0)
    return [float(lo), float(hi)]


def _boot_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_metrik): kullanicilar yeniden orneklenerek ortalamalar. Butun
    metrikler ayni kullanici orneklemini paylasir."""
    values = values.reshape(len(values), -1)
    m = len(values)
    grup = max(1, BOOT_CELLS // max(m * values.shape[1], 1))
    out = np.empty((n_boot, values.shape[1]))
    for bas in range(0, n_boot, grup):
        n = min(grup, n_boot - bas)
        idx = rng.integers(0, m, size=(n, m))
        out[bas:bas + n] = values[idx].mean(axis=1)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, names: list[str], n_boot: int,
                     ci: float, seed: int) -> dict:
    """Ayni kullanicilar, iki kosul: a - b farkinin ortalamasi ve yuzdelik GA'si."""
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    if a.shape != b.shape:
        raise ValueError(f"esli bootstrap icin sekiller ayni olmali: {a.shape} != {b.shape}")
    boot = _boot_means(a - b, n_boot, np.random.default_rng(seed))
    out = {}
    for j, ad in enumerate(names):
        ort_b = float(b[:, j].mean())
        fark = float((a[:, j] - b[:, j]).mean())
        out[ad] = {
            "mean_a": float(a[:, j].mean()),
            "mean_b": ort_b,
            "diff": fark,
            "ci": bootstrap_ci(boot[:, j], ci),
            "relative_diff": fark / ort_b if ort_b else None,
        }
    return out


def independent_bootstrap_diff(x: np.ndarray, y: np.ndarray, *, names: list[str], n_boot: int,
                               ci: float, seed: int) -> dict:
    """Iki BAGIMSIZ kullanici kumesi (farkli kategoriler): ortalama(x) - ortalama(y)."""
    bx = _boot_means(np.asarray(x, dtype=np.float64), n_boot, np.random.default_rng(seed))
    by = _boot_means(np.asarray(y, dtype=np.float64), n_boot, np.random.default_rng(seed + 1))
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    return {ad: {"effect_x": float(x[:, j].mean()), "effect_y": float(y[:, j].mean()),
                 "diff": float(x[:, j].mean() - y[:, j].mean()),
                 "ci": bootstrap_ci(bx[:, j] - by[:, j], ci)}
            for j, ad in enumerate(names)}


def interpret_placebo(ci: list[float]) -> str:
    """Onceden kayitli yorum kurali (DECISIONS 2026-09-14, Hafta 4 sonuclari)."""
    lo, hi = ci
    if lo > 0:
        return "alt_sinir"          # gercek etkinin alt siniri
    if hi < 0:
        return "negatif"            # oldugu gibi raporlanir, yorumlanmaya zorlanmaz
    return "saptanamadi"            # "etki yok" YAZILMAZ


def direction(ci: list[float]) -> str:
    lo, hi = ci
    return "pozitif" if lo > 0 else "negatif" if hi < 0 else "sifiri_iceriyor"


def verdict(criteria: dict) -> str:
    """Kapi 1 ile ayni desen: olculemeyen olcut PASS'i engeller."""
    kararlar = [c["passed"] for c in criteria.values() if c["passed"] is not None]
    atlanan = [k for k, c in criteria.items() if c["passed"] is None]
    if not all(kararlar):
        return "FAIL"
    return "INCOMPLETE" if atlanan else "PASS"


# ------------------------------------------------------------------- okuma
def load_runs(cfg: Config, role: str, model: str) -> dict[tuple[str, int], dict]:
    """(kosul, seed) -> deney raporu. Yalnizca config'teki kosul ve seed'ler aranir."""
    runs = {}
    for code in cfg.get("experiment.conditions"):
        for seed in cfg.get("seeds"):
            yol = experiment_path(cfg, role, code, model, int(seed))
            if yol.exists():
                runs[(code, int(seed))] = read_json(yol)
    return runs


def peruser_means(cfg: Config, role: str, code: str, model: str, seeds: list[int],
                  names: list[str]) -> pl.DataFrame:
    """Kullanici basi metrik, seed'ler uzerinden ortalanmis. Kullanici kumesi her seed'de ayni olmali."""
    parcalar = []
    kume = None
    for seed in seeds:
        yol = peruser_path(cfg, role, code, model, seed)
        if not yol.exists():
            raise FileNotFoundError(
                f"{yol.name} yok - Kaggle ciktisindaki peruser/ klasoru "
                f"data/processed/recbole/{cfg.category_slug(role)}/peruser/ altina konmali")
        df = pl.read_parquet(yol, columns=["user_id", *names])
        users = set(df["user_id"].to_list())
        if kume is not None and users != kume:
            raise RuntimeError(f"{code}/{model}: seed {seed}'in kullanici kumesi digerlerinden farkli")
        kume = users
        parcalar.append(df)
    return pl.concat(parcalar).group_by("user_id").agg([pl.col(n).mean() for n in names]).sort("user_id")


def aligned_peruser(x: pl.DataFrame, y: pl.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if x.height != y.height or not x["user_id"].equals(y["user_id"]):
        raise RuntimeError("iki kosulun kullanici kumesi farkli - esli karsilastirma yapilamaz (Kapi 2, olcut 1)")
    return x.select(names).to_numpy(), y.select(names).to_numpy()


def common_seeds(runs: dict, *codes: str) -> list[int]:
    ortak = None
    for code in codes:
        s = {seed for (c, seed) in runs if c == code}
        ortak = s if ortak is None else ortak & s
    return sorted(ortak or [])


# ------------------------------------------------------------------- Kapi 2
def gate2(cfg: Config, role: str, model: str, runs: dict) -> dict:
    metric = metric_key(cfg.get("gate2.metric"))
    n_boot = int(cfg.get("gate2.bootstrap_n"))
    ci = float(cfg.get("gate2.ci"))
    base_seed = int(cfg.get("seed"))
    codes = sorted({c for c, _ in runs}, key=list(cfg.get("experiment.conditions")).index)
    eksik_cekirdek = [c for c in GATE2_CORE if c not in codes]
    criteria: dict = {}

    # 1. test ciftleri
    ozetler = {f"{c}/seed{s}": r.get("split", {}).get("test_pairs_sha256") for (c, s), r in runs.items()}
    if eksik_cekirdek or not runs:
        criteria["1_same_test_pairs"] = {"passed": None, "reason": f"kosmamis cekirdek kosul: {eksik_cekirdek}"}
    elif None in ozetler.values():
        criteria["1_same_test_pairs"] = {"passed": None, "reason": "ozeti olmayan kosu var (eski kod surumu)",
                                         "runs_without_hash": sorted(k for k, v in ozetler.items() if v is None)}
    else:
        criteria["1_same_test_pairs"] = {"passed": len(set(ozetler.values())) == 1,
                                         "n_distinct_hashes": len(set(ozetler.values())),
                                         "runs_checked": sorted(ozetler)}

    # 2. evren
    evren = {}
    for c in codes:
        yol = condition_report_path(cfg, role, c)
        evren[c] = read_json(yol).get("universe_is_subset_of_c0") if yol.exists() else None
    if eksik_cekirdek or None in evren.values():
        criteria["2_universe_subset_of_c0"] = {"passed": None, "per_condition": evren,
                                               "reason": "kosmamis cekirdek kosul ya da eksik kosul raporu"}
    else:
        criteria["2_universe_subset_of_c0"] = {"passed": all(evren.values()), "per_condition": evren}

    # 3. plasebo tabani
    karsit = {}
    for a, b in PLACEBO_PAIRS:
        seeds = common_seeds(runs, a, b)
        if not seeds:
            karsit[f"{a}-{b}"] = None
            continue
        x, y = aligned_peruser(peruser_means(cfg, role, a, model, seeds, [metric]),
                        peruser_means(cfg, role, b, model, seeds, [metric]), [metric])
        sonuc = paired_bootstrap(x, y, names=[metric], n_boot=n_boot, ci=ci,
                                 seed=seed_for(base_seed, f"gate2/{role}/{model}/{a}-{b}"))[metric]
        karsit[f"{a}-{b}"] = {**sonuc, "seeds": seeds, "n_users": len(x),
                              "passed": sonuc["ci"][0] <= 0}
    if None in karsit.values():
        criteria["3_placebo_not_better_than_c0"] = {"passed": None, "metric": metric, "contrasts": karsit,
                                                    "reason": "C0, C4 ya da C4b kosmamis"}
    else:
        criteria["3_placebo_not_better_than_c0"] = {
            "passed": all(v["passed"] for v in karsit.values()), "metric": metric, "contrasts": karsit}

    # 4. seed kararliligi
    beklenen = [int(s) for s in cfg.get("seeds")]
    c0 = {s: runs[("C0", s)]["test"].get(metric) for s in beklenen if ("C0", s) in runs}
    esik = float(cfg.get("gate2.seed_cv_max"))
    if len(c0) < len(beklenen) or len(beklenen) < 2 or None in c0.values():
        criteria["4_seed_stability"] = {"passed": None, "metric": metric, "values": c0,
                                        "reason": f"C0 icin beklenen seed'ler {beklenen}, kosan {sorted(c0)}"}
    else:
        v = np.array(list(c0.values()), dtype=np.float64)
        cv = float(v.std(ddof=1) / v.mean()) if v.mean() else None
        criteria["4_seed_stability"] = {"passed": cv is not None and cv < esik, "metric": metric,
                                        "values": c0, "cv": cv, "threshold": esik}

    return {"conditions_found": codes, "criteria": criteria, "verdict": verdict(criteria)}


# ------------------------------------------------------------------- karsitliklar
def contrasts(cfg: Config, role: str, model: str, runs: dict, names: list[str]) -> dict:
    n_boot = int(cfg.get("experiment.bootstrap_iters"))
    ci = float(cfg.get("experiment.ci"))
    base_seed = int(cfg.get("seed"))
    birincil = metric_key(cfg.get("gate2.metric"))
    out = {}
    for a, b, soru, plasebo in CONTRASTS:
        anahtar = f"{a}-{b}"
        seeds = common_seeds(runs, a, b)
        if not seeds:
            out[anahtar] = {"question": soru, "skipped": "iki kosulun ortak seed'i yok"}
            continue
        x, y = aligned_peruser(peruser_means(cfg, role, a, model, seeds, names),
                        peruser_means(cfg, role, b, model, seeds, names), names)
        sonuc = paired_bootstrap(x, y, names=names, n_boot=n_boot, ci=ci,
                                 seed=seed_for(base_seed, f"stats/{role}/{model}/{anahtar}"))
        for ad, s in sonuc.items():
            s["direction"] = direction(s["ci"])
            if plasebo:
                s["interpretation"] = interpret_placebo(s["ci"])
        out[anahtar] = {"question": soru, "placebo_contrast": plasebo, "seeds": seeds,
                        "n_users": len(x), "primary_metric": birincil, "metrics": sonuc}
    return out


def dose_response(cfg: Config, model: str, runs_by_role: dict, names: list[str]) -> dict:
    """Plasebo karsitliginin kullanici basi farki, iki kategori arasinda (bagimsiz orneklem)."""
    yuksek, dusuk = DOSE_ROLES
    if yuksek not in runs_by_role or dusuk not in runs_by_role:
        return {"skipped": f"iki kategori de gerekli: {DOSE_ROLES}"}
    out = {}
    for a, b, _, plasebo in CONTRASTS:
        if not plasebo:
            continue
        farklar = {}
        for role in DOSE_ROLES:
            runs = runs_by_role[role][model]
            seeds = common_seeds(runs, a, b)
            if not seeds:
                break
            x, y = aligned_peruser(peruser_means(cfg, role, a, model, seeds, names),
                            peruser_means(cfg, role, b, model, seeds, names), names)
            farklar[role] = x - y
        if len(farklar) < 2:
            out[f"{a}-{b}"] = {"skipped": "bir kategoride ortak seed yok"}
            continue
        out[f"{a}-{b}"] = {
            "x": cfg.category_slug(yuksek), "y": cfg.category_slug(dusuk),
            "metrics": independent_bootstrap_diff(
                farklar[yuksek], farklar[dusuk], names=names,
                n_boot=int(cfg.get("experiment.bootstrap_iters")), ci=float(cfg.get("experiment.ci")),
                seed=seed_for(int(cfg.get("seed")), f"dose/{model}/{a}-{b}")),
        }
    return out


def label_quality(cfg: Config) -> dict:
    """Iki eksenin olculen etiket kesinligi/duyarliligi (Hafta 4) - sonucun yaninda durur."""
    yol = cfg.path("results", "validation_500.json")
    if not yol.exists():
        return {}
    eksenler = read_json(yol).get("measurements", {}).get("contamination_axes", {}).get("main_weighted", {})
    return {ad: {k: eksenler[ad][k]["value"] for k in ("precision", "recall", "f1")}
            for ad in ("C1", "C1b") if ad in eksenler}


# ------------------------------------------------------------------- kosu
def evaluate(cfg: Config, roles: list[str]) -> tuple[dict, dict]:
    names = metric_keys(cfg)
    models = list(cfg.get("experiment.models"))
    runs_by_role: dict = {}
    gate: dict = {}
    stats: dict = {}
    kaynaklar: set = set()
    for role in roles:
        slug = cfg.category_slug(role)
        runs_by_role[role] = {}
        for model in models:
            runs = load_runs(cfg, role, model)
            runs_by_role[role][model] = runs
            kaynaklar |= {r["meta"].get("label_source") for r in runs.values()}
            anahtar = f"{slug}/{model}"
            if not runs:
                gate[anahtar] = {"verdict": "INCOMPLETE", "reason": "hic kosu yok"}
                stats[anahtar] = {"skipped": "hic kosu yok"}
                continue
            gate[anahtar] = gate2(cfg, role, model, runs)
            stats[anahtar] = {
                "gate2_verdict": gate[anahtar]["verdict"],
                # Kapi 2 PASS degilse sonuc YORUMLANAMAZ (GENEL_BAKIS, Kapi 2).
                "interpretable": gate[anahtar]["verdict"] == "PASS",
                "runs": sorted(f"{c}/seed{s}" for c, s in runs),
                "contrasts": contrasts(cfg, role, model, runs, names),
            }
            log.info("KAPI 2 %s -> %s", anahtar, gate[anahtar]["verdict"])

    raporlanabilir = bool(kaynaklar) and kaynaklar <= REPORTABLE_LABEL_SOURCES
    meta = {
        "code_version": code_version(),
        "label_sources": sorted(str(k) for k in kaynaklar),
        # Vekil/sentetik etiketli kosudan uretilen istatistik DUMAN TESTIDIR.
        "reportable": raporlanabilir,
        "thresholds_fixed": "2026-09-14, configs/base.yaml -> gate2 (deney kosulmadan)",
    }
    gate_rapor = {"meta": meta, "results": gate}
    stats_rapor = {
        "meta": {**meta, "metrics": names, "bootstrap_iters": int(cfg.get("experiment.bootstrap_iters")),
                 "ci": float(cfg.get("experiment.ci")),
                 "method": "kullanici basi metrik seed'ler uzerinden ortalanir; esli bootstrap, yuzdelik GA"},
        "label_quality_population": label_quality(cfg),
        "results": stats,
        "dose_response": {m: dose_response(cfg, m, runs_by_role, names) for m in models},
    }
    if not raporlanabilir:
        log.warning("etiket kaynagi %s - DUMAN TESTI, sonuc tablosuna giremez", meta["label_sources"])
    return gate_rapor, stats_rapor


def gate2_path(cfg: Config) -> Path:
    return cfg.path("results", "gate2.json")


def stats_path(cfg: Config) -> Path:
    return cfg.path("results", "experiment_stats.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default=None,
                        help="Bos birakilirsa `experiment.categories`.")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    roles = (resolve_roles(cfg, args.category) if args.category
             else list(cfg.get("experiment.categories")))
    gate_rapor, stats_rapor = evaluate(cfg, roles)
    write_json(gate_rapor, gate2_path(cfg), log)
    write_json(stats_rapor, stats_path(cfg), log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
