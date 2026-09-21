"""Seed 42 yeniden kosusunun denetimi (2026-09-21). Ana ortamda (.venv) kosar.

Girdi : Kaggle'dan inen bir ya da daha fazla `out/` klasoru (--rerun)
          experiment_*_seed42.json, condition_*.json, peruser/<slug>/*.parquet,
          oturum_ozeti.json
        reports/results/experiment_*.json + data/processed/recbole/<slug>/peruser/ (orijinal)
Cikti : reports/results/reproduction_seed42.json
        reports/results/reproduction/seed42/experiment_*.json   (yalnizca JSON, kopya)

KURALLAR ON KAYITTA (DECISIONS 2026-09-21), YENIDEN KOSU GORULMEDEN yazildi:

  * "BIREBIR AYNI" icin ucu birden gerekir: kullanici basi parquet'in butun metrik
    kolonlari ve top-K listeleri esit, rapordaki `test` metrikleri esit,
    `test_pairs_sha256` esit. Biri tutmazsa fark hucre hucre yazilir ve ORIJINAL 72
    kosu birincil kalir - yeniden kosunun sayilari hicbir sonuc tablosuna girmez.
  * Seed 1337 ve 2024'un epoch sayisi OLCULMEZ, tahmin edilir:
    epochs_42 x fit_s / fit_42 (orijinal kosularin sureleri). "Tahmin" diye yazilir.
  * Herhangi bir kosu tavana degerse ya da tahmin tavanin %90'ini asarsa
    `stop_and_decide: true` yazilir: kullaniciyla karar verilmeden devam edilmez.

Yeniden kosunun raporu bu kopyalar icin `reports/results/reproduction/` ALT
klasorune gider: `experiment_stats.load_runs` tam dosya adiyla koku okudugu icin
oradaki dosyalar hicbir analize girmez (testle kilitli).

Kullanim:
    python -m gift_contamination.analysis.reproduction --config configs/base.yaml \\
        --rerun %USERPROFILE%/kaggle_yukleme/deney_out_R1 %USERPROFILE%/kaggle_yukleme/deney_out_R2
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..recsys.atomic import inter_path, split_path
from ..recsys.conditions import condition_report_path
from ..recsys.run_experiment import experiment_path, peruser_path
from ..utils.io import code_version, read_json, write_json
from ..utils.logging import get_logger

log = get_logger("analysis.reproduction")

NEAR_CAP = 0.9            # tahmin tavanin bu payini asarsa dur (on kayit)
# Yeni kosucunun (86d75f7 ve sonrasi) yazdigi alanlar. Tasimayan bir rapor YENIDEN
# KOSU DEGILDIR - onceki oturumdan out/'a kopyalanmis eski bir dosyadir.
NEW_FIELDS = ("versions", "epochs_trained", "best_epoch")


def output_path(cfg: Config, seed: int) -> Path:
    return cfg.path("results", f"reproduction_seed{seed}.json")


def copies_dir(cfg: Config, seed: int) -> Path:
    return cfg.path("results", "reproduction", f"seed{seed}")


def _sha256(yol: Path) -> str | None:
    if not yol.exists():
        return None
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


# ------------------------------------------------------------------ bulma
def find_reruns(dirs: list[Path], seed: int) -> dict[tuple[str, str, str], tuple[Path, Path]]:
    """(slug, kosul, model) -> (rapor, o raporun geldigi out/ koku). Ayni hucre iki kez gelirse hata."""
    out: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    for d in dirs:
        for yol in sorted(Path(d).rglob(f"experiment_*_seed{seed}.json")):
            meta = read_json(yol).get("meta", {})
            key = (meta.get("category"), meta.get("condition"), meta.get("model"))
            if None in key or meta.get("seed") != seed:
                raise RuntimeError(f"{yol.name}: meta eksik ya da seed {seed} degil")
            if key in out:
                raise RuntimeError(f"{key} iki kez var: {out[key][0]} ve {yol}")
            out[key] = (yol, Path(d))
    return out


def find_peruser(root: Path, slug: str, code: str, model: str, seed: int) -> Path | None:
    """Yalnizca VERILEN out/ kokunun ALTINDA: peruser/<slug>/<kosul>_<model>_seed<N>.parquet."""
    adaylar = sorted(Path(root).rglob(f"peruser/{slug}/{code}_{model}_seed{seed}.parquet"))
    if len(adaylar) > 1:
        raise RuntimeError(f"{slug}/{code}/{model}: birden cok kullanici basi dosya: {adaylar}")
    return adaylar[0] if adaylar else None


# ------------------------------------------------------------------ karsilastirma
def compare_peruser(original: Path, rerun: Path) -> dict:
    """Kullanici basi iki dosya: metrik kolonlari ve top-K listeleri BIREBIR mi."""
    a = pl.read_parquet(original).sort("user_id")
    b = pl.read_parquet(rerun).sort("user_id")
    if a.columns != b.columns or not a["user_id"].equals(b["user_id"]):
        return {"identical": False, "reason": "kolonlar ya da kullanici kumesi farkli",
                "n_users_original": a.height, "n_users_rerun": b.height}
    metrikler = [c for c in a.columns if c not in ("user_id", "topk_items")]
    fark = {c: float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()))) if a.height else 0.0
            for c in metrikler}
    topk_farkli = int((a["topk_items"] != b["topk_items"]).sum()) if "topk_items" in a.columns else None
    return {
        "identical": all(v == 0.0 for v in fark.values()) and not topk_farkli,
        "n_users": a.height,
        "max_abs_diff": fark,
        "n_users_topk_differ": topk_farkli,
    }


def compare_run(cfg: Config, role_of: dict[str, str], key: tuple[str, str, str], rerun_report: Path,
                root: Path, seed: int) -> dict:
    slug, code, model = key
    role = role_of[slug]
    yeni = read_json(rerun_report)
    meta = yeni.get("meta", {})
    out: dict = {"category": slug, "condition": code, "model": model, "seed": seed,
                 "rerun_code_version": meta.get("code_version")}
    eksik = [f for f in NEW_FIELDS if f not in meta]
    if eksik:
        # Onceki oturumdan out/'a kopyalanmis eski rapor - yeniden kosu degil.
        return {**out, "is_rerun": False, "identical": None,
                "reason": f"yeni kosucunun alanlari yok: {eksik} (eski kopya?)"}
    out["is_rerun"] = True
    out["epochs"] = {k: meta.get(k) for k in
                     ("epochs", "epochs_trained", "best_epoch", "epochs_since_best",
                      "stopping_step", "hit_epoch_cap")}
    out["versions"] = meta.get("versions")
    out["rerun_fit_seconds"] = (meta.get("runtime_seconds") or {}).get("fit")

    orijinal_yolu = experiment_path(cfg, role, code, model, seed)
    if not orijinal_yolu.exists():
        return {**out, "identical": None, "reason": f"orijinal rapor yok: {orijinal_yolu.name}"}
    eski = read_json(orijinal_yolu)
    out["original_code_version"] = eski.get("meta", {}).get("code_version")
    out["test_metrics_equal"] = eski.get("test") == yeni.get("test")
    out["test_pairs_equal"] = (eski.get("split", {}).get("test_pairs_sha256")
                               == yeni.get("split", {}).get("test_pairs_sha256"))
    if not out["test_metrics_equal"]:
        out["test_metric_diff"] = {k: (yeni["test"][k] - v if k in yeni.get("test", {}) else None)
                                   for k, v in eski["test"].items()}

    pu_eski = peruser_path(cfg, role, code, model, seed)
    pu_yeni = find_peruser(root, slug, code, model, seed)
    if not pu_eski.exists() or pu_yeni is None:
        out["peruser"] = {"identical": None,
                          "reason": "kullanici basi dosya yok (orijinal ya da yeniden kosu)"}
    else:
        out["peruser"] = compare_peruser(pu_eski, pu_yeni)
    parcalar = (out["test_metrics_equal"], out["test_pairs_equal"], out["peruser"]["identical"])
    out["identical"] = None if None in parcalar else all(parcalar)
    return out


def compare_conditions(cfg: Config, role_of: dict[str, str], dirs: list[Path]) -> dict:
    """Kaggle'da yeniden uretilen kosul raporlari depodakilerle AYNI mi (tam sozluk)."""
    out = {}
    for d in dirs:
        for yol in sorted(Path(d).rglob("condition_*.json")):
            rapor = read_json(yol)
            code = rapor.get("condition")
            slug = next((s for s in role_of if yol.name == f"condition_{s}_{code}.json"), None)
            if slug is None:
                continue
            depo = condition_report_path(cfg, role_of[slug], code)
            out[f"{slug}/{code}"] = depo.exists() and read_json(depo) == rapor
    return out


def check_inputs(cfg: Config, role_of: dict[str, str], dirs: list[Path]) -> dict:
    """Oturum ozetindeki girdi SHA-256'si yereldeki atomic dosyalarla ayni mi."""
    out = {}
    for d in dirs:
        for yol in sorted(Path(d).rglob("oturum_ozeti.json")):
            for slug, ozet in (read_json(yol).get("girdi_sha256") or {}).items():
                if slug not in role_of:
                    continue
                role = role_of[slug]
                out[slug] = {
                    "inter_equal": ozet.get("inter_sha256") == _sha256(inter_path(cfg, role)),
                    "split_equal": ozet.get("split_sha256") == _sha256(split_path(cfg, role)),
                }
    return out


# ------------------------------------------------------------------ epoch tahmini
def estimate_other_seeds(cfg: Config, role_of: dict[str, str], runs: list[dict], seed: int) -> list[dict]:
    """epochs_seed ~ epochs_42 x fit_seed / fit_42 (ORIJINAL kosularin sureleri). TAHMIN."""
    out = []
    for r in runs:
        n42 = (r.get("epochs") or {}).get("epochs_trained")
        if not r.get("is_rerun") or n42 is None:
            continue
        role = role_of[r["category"]]
        fit42 = (read_json(experiment_path(cfg, role, r["condition"], r["model"], seed))
                 .get("meta", {}).get("runtime_seconds", {}).get("fit"))
        for s in cfg.get("seeds"):
            s = int(s)
            if s == seed:
                continue
            yol = experiment_path(cfg, role, r["condition"], r["model"], s)
            if not yol.exists() or not fit42:
                continue
            meta = read_json(yol).get("meta", {})
            fit = meta.get("runtime_seconds", {}).get("fit")
            cap = meta.get("epochs")
            tahmin = n42 * fit / fit42 if fit else None
            out.append({
                "category": r["category"], "condition": r["condition"], "model": r["model"],
                "seed": s, "estimated_epochs": round(tahmin, 1) if tahmin else None,
                "cap": cap, "near_cap": bool(tahmin and cap and tahmin >= NEAR_CAP * cap),
                "basis": f"epochs_{seed}={n42} x fit_{s} / fit_{seed} (orijinal kosular); +-%10 tahmin",
            })
    return out


# ------------------------------------------------------------------ kosu
def evaluate(cfg: Config, dirs: list[Path], *, seed: int = 42, copy: bool = True) -> dict:
    role_of = {cfg.category_slug(r): r for r in cfg.get("experiment.categories")}
    reruns = find_reruns(dirs, seed)
    beklenen = {(cfg.category_slug(r), c, m) for r in cfg.get("experiment.categories")
                for c in cfg.get("experiment.conditions") for m in cfg.get("experiment.models")}
    ciftler = [(compare_run(cfg, role_of, k, yol, kok, seed), yol)
               for k, (yol, kok) in sorted(reruns.items()) if k[0] in role_of]
    runs = [r for r, _ in ciftler]
    tahmin = estimate_other_seeds(cfg, role_of, runs, seed)
    gercek = [r for r in runs if r.get("is_rerun")]
    yeniden = {(r["category"], r["condition"], r["model"]) for r in gercek}
    tavan = [f"{r['category']}/{r['condition']}/{r['model']}" for r in gercek
             if (r.get("epochs") or {}).get("hit_epoch_cap")]
    yakin = [f"{t['category']}/{t['condition']}/{t['model']}/seed{t['seed']}" for t in tahmin if t["near_cap"]]
    ozdes = [r.get("identical") for r in gercek]
    epochlar = [r["epochs"]["epochs_trained"] for r in gercek if r["epochs"].get("epochs_trained")]
    if copy:
        hedef = copies_dir(cfg, seed)
        hedef.mkdir(parents=True, exist_ok=True)
        for r, yol in ciftler:
            if r.get("is_rerun"):
                shutil.copy(yol, hedef / yol.name)
    return {
        "meta": {
            "code_version": code_version(),
            "rules_fixed": "DECISIONS 2026-09-21 on kaydi (yeniden kosu gorulmeden)",
            "seed": seed,
            "identical_means": "peruser metrikleri + top-K esit, test metrikleri esit, test_pairs_sha256 esit",
            "primary_results_unchanged": "orijinal 72 kosu birincil; bu dosya yalnizca denetim",
        },
        "summary": {
            "n_expected": len(beklenen),
            "n_reruns": len(gercek),
            "missing": sorted("/".join(k) for k in beklenen - yeniden),
            "stale_copies": sorted(f"{r['category']}/{r['condition']}/{r['model']}"
                                   for r in runs if not r.get("is_rerun")),
            "n_identical": sum(1 for v in ozdes if v is True),
            "n_different": sum(1 for v in ozdes if v is False),
            "n_undetermined": sum(1 for v in ozdes if v is None),
            "epochs_trained_min_max": [min(epochlar), max(epochlar)] if epochlar else None,
            "runs_hit_cap": tavan,
            "estimates_near_cap": yakin,
            "stop_and_decide": bool(tavan or yakin),
        },
        "inputs": check_inputs(cfg, role_of, dirs),
        "conditions_equal": compare_conditions(cfg, role_of, dirs),
        "runs": runs,
        "estimated_epochs_other_seeds": tahmin,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--rerun", nargs="+", required=True, help="Kaggle'dan inen out/ klasorleri")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    rapor = evaluate(cfg, [Path(d) for d in args.rerun], seed=args.seed)
    write_json(rapor, output_path(cfg, args.seed), log)
    s = rapor["summary"]
    log.info("yeniden kosu: %d/%d | birebir ayni %d, farkli %d, belirsiz %d | tavana degen %s | "
             "tahmini tavana yakin %s", s["n_reruns"], s["n_expected"], s["n_identical"],
             s["n_different"], s["n_undetermined"], s["runs_hit_cap"] or "yok",
             s["estimates_near_cap"] or "yok")
    if s["stop_and_decide"]:
        log.warning("ON KAYIT: tavana degen ya da yaklasan kosu var - kullaniciyla karar verilmeli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
