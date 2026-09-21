"""Seed 42 yeniden kosusunun denetimi: kurallar on kayittaki gibi uygulaniyor mu.

Orijinal kosu dosyalari ve Kaggle'dan inmis gibi bir `out/` klasoru elle kuruluyor.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from gift_contamination.analysis import experiment_stats as es
from gift_contamination.analysis import reproduction as rp
from gift_contamination.config import Config
from gift_contamination.recsys.atomic import inter_path, split_path
from gift_contamination.recsys.conditions import condition_report_path
from gift_contamination.recsys.run_experiment import experiment_path, peruser_path
from gift_contamination.utils.io import write_json

SLUG = "Toys_and_Games"
TEST = {"recall@10": 0.042816, "ndcg@10": 0.02017}


@pytest.fixture
def pcfg(cfg: Config) -> Config:
    d = cfg._data
    d["dataset"]["categories"] = {"high": "raw_review_Toys_and_Games"}
    d["seeds"] = [42, 1337]
    d["experiment"] = {"categories": ["high"], "conditions": ["C0", "C1"], "models": ["BPR"]}
    return cfg


def _peruser(topk_ilk="i1") -> pl.DataFrame:
    return pl.DataFrame({"user_id": ["u1", "u2"], "recall@10": [1.0, 0.0], "ndcg@10": [0.5, 0.0],
                         "topk_items": [[topk_ilk, "i2"], ["i3", "i4"]]})


def _original(cfg, code, seed, fit):
    write_json({"meta": {"category": SLUG, "condition": code, "model": "BPR", "seed": seed,
                         "epochs": 300, "runtime_seconds": {"fit": fit, "test": 1.0},
                         "code_version": "e1c7f89"},
                "split": {"test_pairs_sha256": "abc"}, "test": TEST},
               experiment_path(cfg, "high", code, "BPR", seed))
    yol = peruser_path(cfg, "high", code, "BPR", seed)
    yol.parent.mkdir(parents=True, exist_ok=True)
    _peruser().write_parquet(yol)


def _rerun(out: Path, code, *, epochs=120, since_best=11, new_fields=True, topk_ilk="i1", test=None):
    meta = {"category": SLUG, "condition": code, "model": "BPR", "seed": 42, "epochs": 300,
            "runtime_seconds": {"fit": 95.0, "test": 1.0}, "code_version": "86d75f7"}
    if new_fields:
        meta |= {"versions": {"torch": "2.x"}, "epochs_trained": epochs,
                 "best_epoch": epochs - since_best, "epochs_since_best": since_best,
                 "stopping_step": 10, "hit_epoch_cap": epochs >= 300}
    write_json({"meta": meta, "split": {"test_pairs_sha256": "abc"}, "test": test or TEST},
               out / f"experiment_{SLUG}_{code}_BPR_seed42.json")
    yol = out / "peruser" / SLUG / f"{code}_BPR_seed42.parquet"
    yol.parent.mkdir(parents=True, exist_ok=True)
    _peruser(topk_ilk).write_parquet(yol)


def _world(cfg, tmp_path, **kw) -> Path:
    for code in ("C0", "C1"):
        _original(cfg, code, 42, fit=100.0)
        _original(cfg, code, 1337, fit=200.0)
        write_json({"condition": code, "n_removed": 5}, condition_report_path(cfg, "high", code))
    out = tmp_path / "deney_out_R1"
    out.mkdir()
    for code in ("C0", "C1"):
        _rerun(out, code, **kw.get(code, {}))
        write_json({"condition": code, "n_removed": 5}, out / f"condition_{SLUG}_{code}.json")
    ip = inter_path(cfg, "high")
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text("user_id:token\titem_id:token\n", encoding="utf-8")
    split_path(cfg, "high").write_text("{}", encoding="utf-8")
    (out / "oturum_ozeti.json").write_text(json.dumps({"girdi_sha256": {SLUG: {
        "inter_sha256": rp._sha256(ip), "split_sha256": rp._sha256(split_path(cfg, "high"))}}}),
        encoding="utf-8")
    return out


def test_an_identical_rerun_is_reported_identical_and_epochs_are_estimated(pcfg, tmp_path):
    out = _world(pcfg, tmp_path)

    r = rp.evaluate(pcfg, [out])

    s = r["summary"]
    assert (s["n_expected"], s["n_reruns"], s["n_identical"], s["n_different"]) == (2, 2, 2, 0)
    assert s["missing"] == [] and s["stop_and_decide"] is False
    assert r["inputs"] == {SLUG: {"inter_equal": True, "split_equal": True}}
    assert r["conditions_equal"] == {f"{SLUG}/C0": True, f"{SLUG}/C1": True}
    run = r["runs"][0]
    assert run["epochs"]["epochs_trained"] == 120 and run["epochs"]["best_epoch"] == 109
    # seed 1337 orijinalde iki kat surmus -> 240 epoch TAHMINI
    tahmin = [t for t in r["estimated_epochs_other_seeds"] if t["condition"] == "C0"]
    assert tahmin == [{"category": SLUG, "condition": "C0", "model": "BPR", "seed": 1337,
                       "estimated_epochs": 240.0, "cap": 300, "near_cap": False,
                       "basis": tahmin[0]["basis"]}]


def test_one_different_topk_list_makes_the_cell_not_identical(pcfg, tmp_path):
    out = _world(pcfg, tmp_path, C1={"topk_ilk": "BASKA"})

    r = rp.evaluate(pcfg, [out])

    c1 = next(x for x in r["runs"] if x["condition"] == "C1")
    assert c1["identical"] is False
    assert c1["peruser"]["n_users_topk_differ"] == 1
    assert c1["test_metrics_equal"] is True          # ortalama tutsa bile
    assert r["summary"]["n_different"] == 1


def test_a_different_test_metric_is_reported_with_its_difference(pcfg, tmp_path):
    out = _world(pcfg, tmp_path, C0={"test": {"recall@10": 0.042817, "ndcg@10": 0.02017}})

    c0 = next(x for x in rp.evaluate(pcfg, [out])["runs"] if x["condition"] == "C0")

    assert c0["identical"] is False
    assert c0["test_metric_diff"]["recall@10"] == pytest.approx(1e-6)


def test_an_old_copy_in_out_is_not_mistaken_for_a_rerun(pcfg, tmp_path):
    """Onceki oturumun raporu out/'a kopyalanmissa yeni kosucunun alanlarini tasimaz."""
    out = _world(pcfg, tmp_path, C1={"new_fields": False})

    s = rp.evaluate(pcfg, [out])["summary"]

    assert s["stale_copies"] == [f"{SLUG}/C1/BPR"]
    assert s["n_reruns"] == 1 and s["missing"] == [f"{SLUG}/C1/BPR"]


def test_hitting_the_cap_or_nearing_it_stops_for_a_decision(pcfg, tmp_path):
    out = _world(pcfg, tmp_path, C0={"epochs": 300, "since_best": 2})

    s = rp.evaluate(pcfg, [out])["summary"]

    assert s["runs_hit_cap"] == [f"{SLUG}/C0/BPR"]
    # C0 seed 1337: 300 x 200/100 = 600 -> tavanin %90'i asilir
    assert f"{SLUG}/C0/BPR/seed1337" in s["estimates_near_cap"]
    assert s["stop_and_decide"] is True


def test_copies_live_in_a_subfolder_that_no_analysis_reads(pcfg, tmp_path):
    out = _world(pcfg, tmp_path)
    rp.evaluate(pcfg, [out])
    kopya = rp.copies_dir(pcfg, 42) / f"experiment_{SLUG}_C0_BPR_seed42.json"
    assert kopya.exists()

    experiment_path(pcfg, "high", "C0", "BPR", 42).unlink()

    assert ("C0", 42) not in es.load_runs(pcfg, "high", "BPR")
