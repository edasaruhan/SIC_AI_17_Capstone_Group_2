"""Post-hoc saglamlik (2026-09-21): seed duzeyi ve maruziyet ayristirmasi.

Kosu dosyalari `run_experiment`in, kosul dosyalari `conditions`in yazdigi bicimde elle
kuruluyor; her grubun beklenen farki kurgudan belli (grup icinde SABIT fark -> GA
genisligi sifir, fark tam o sabit).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gift_contamination.analysis import robustness as rb
from gift_contamination.config import Config
from gift_contamination.recsys.conditions import condition_path
from gift_contamination.recsys.run_experiment import experiment_path, peruser_path
from gift_contamination.utils.io import write_json

SEEDS = [42, 1337, 2024]
CONDS = ["C0", "C1", "C4", "C1b", "C4b", "C3"]
# Dort grup, grup basina 5 kullanici. Tedavi (C1) u4-u7 gruplarinin, plasebo (C4)
# u2-u3 ve u6-u7 gruplarinin birer egitim satirini siler.
GRUP_OF = {**{f"u{i:02d}": "neither" for i in range(0, 5)},
           **{f"u{i:02d}": "placebo_only" for i in range(5, 10)},
           **{f"u{i:02d}": "treatment_only" for i in range(10, 15)},
           **{f"u{i:02d}": "both" for i in range(15, 20)}}
FARK = {"neither": 0.10, "placebo_only": 0.20, "treatment_only": -0.05, "both": 0.0}


@pytest.fixture
def rcfg(cfg: Config) -> Config:
    d = cfg._data
    d["dataset"]["categories"] = {"high": "raw_review_Toys_and_Games"}
    d["seeds"] = SEEDS
    d["experiment"] = {"conditions": CONDS, "categories": ["high"], "models": ["BPR"],
                       "topk": [10], "metrics": ["Recall"], "bootstrap_iters": 200, "ci": 0.95}
    d["gate2"] = {"metric": "Recall@10"}
    return cfg


def _write_run(cfg, code, seed, values: dict[str, float], *, aggregate=None):
    users = sorted(values)
    pu = peruser_path(cfg, "high", code, "BPR", seed)
    pu.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": users, "recall@10": [values[u] for u in users]}).write_parquet(pu)
    write_json({"meta": {"condition": code, "model": "BPR", "seed": seed, "label_source": "distilled"},
                "test": {"recall@10": aggregate if aggregate is not None else float(np.mean(list(values.values())))}},
               experiment_path(cfg, "high", code, "BPR", seed))


def _write_condition(cfg, code, removed: set[str]):
    """Her kullanici: 3 egitim + valid + test. `removed` kullanicilarinin ilk egitim satiri silinir."""
    satirlar = []
    for u in GRUP_OF:
        for k in range(3):
            if k == 0 and u in removed:
                continue
            satirlar.append((u, f"i{k}", float(k), "self", "train"))
        satirlar += [(u, "iv", 3.0, "self", "valid"), (u, "it", 4.0, "self", "test")]
    yol = condition_path(cfg, "high", code)
    yol.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(satirlar, schema=["user_id:token", "item_id:token", "timestamp", "label:token", "split"],
                 orient="row").write_csv(yol, separator="\t")


def _world(cfg):
    tedavi = {u for u, g in GRUP_OF.items() if g in ("treatment_only", "both")}
    plasebo = {u for u, g in GRUP_OF.items() if g in ("placebo_only", "both")}
    for code, sil in (("C0", set()), ("C1", tedavi), ("C4", plasebo), ("C1b", tedavi), ("C4b", plasebo)):
        _write_condition(cfg, code, sil)
    base = {u: 0.3 + 0.01 * i for i, u in enumerate(GRUP_OF)}
    for seed in SEEDS:
        _write_run(cfg, "C0", seed, base)
        _write_run(cfg, "C4", seed, base)
        _write_run(cfg, "C1", seed, {u: v + FARK[GRUP_OF[u]] for u, v in base.items()})
        _write_run(cfg, "C4b", seed, base)
        _write_run(cfg, "C1b", seed, {u: v + FARK[GRUP_OF[u]] for u, v in base.items()})
        _write_run(cfg, "C3", seed, base)


def test_seed_level_reports_each_seed_and_its_sign_agreement():
    runs = {("C1", s): {"test": {"recall@10": v}} for s, v in zip(SEEDS, (0.31, 0.33, 0.28))}
    runs |= {("C4", s): {"test": {"recall@10": 0.30}} for s in SEEDS}

    out = rb.seed_level(runs, "C1", "C4", "recall@10", user_ci=[0.0, 0.02])

    assert out["per_seed"] == pytest.approx({"42": 0.01, "1337": 0.03, "2024": -0.02})
    assert out["mean"] == pytest.approx(0.02 / 3)
    assert out["sd"] == pytest.approx(np.std([0.01, 0.03, -0.02], ddof=1))
    assert out["sd_over_sqrt_n"] == pytest.approx(out["sd"] / np.sqrt(3))
    assert out["sign_agreement"] == "2/3"
    assert out["user_ci_half_width"] == pytest.approx(0.01)


def test_seed_level_without_common_seeds_is_skipped_not_invented():
    runs = {("C1", 42): {"test": {"recall@10": 0.3}}, ("C4", 1337): {"test": {"recall@10": 0.3}}}
    assert "skipped" in rb.seed_level(runs, "C1", "C4", "recall@10", None)


def test_exposure_groups_users_by_which_condition_touched_their_training_rows(rcfg):
    _world(rcfg)

    out = rb.evaluate(rcfg, ["high"])

    assert out["meta"]["post_hoc"] is True and out["meta"]["replaces_primary"] is False
    blok = out["exposure"]["Toys_and_Games/BPR"]["C1-C4"]
    assert blok["n_test_users"] == 20
    for ad, fark in FARK.items():
        g = blok["groups"][ad]
        assert g["n_users"] == 5, ad
        assert g["diff"] == pytest.approx(fark, abs=1e-12), ad
        # Grup icinde fark sabit: esli bootstrap GA'si tek noktaya coker.
        assert g["ci"][1] - g["ci"][0] < 1e-9, ad
    assert blok["groups"]["neither"]["direction"] == "pozitif"
    assert blok["groups"]["treatment_only"]["direction"] == "negatif"
    # Seed duzeyi de ayni ciktida; kosu raporlarindan.
    assert out["seed_level"]["Toys_and_Games/BPR"]["C1-C4"]["sign_agreement"] == "3/3"


def test_touched_counts_a_user_who_lost_every_training_row():
    c0 = pl.DataFrame({"user_id": ["a", "b", "c"], "n": [3, 3, 3]})
    cond = pl.DataFrame({"user_id": ["a", "b"], "n": [3, 1]})      # c'nin hic satiri kalmadi

    out = rb.touched(c0, cond, "t").sort("user_id")

    assert out["t"].to_list() == [False, True, True]


def test_seed_level_needs_no_data_files(rcfg):
    for code in CONDS:
        for seed in SEEDS:
            write_json({"meta": {"label_source": "distilled"}, "test": {"recall@10": 0.3}},
                       experiment_path(rcfg, "high", code, "BPR", seed))

    out = rb.evaluate(rcfg, ["high"], with_exposure=False)

    assert out["seed_level"]["Toys_and_Games/BPR"]["C1-C4"]["mean"] == 0.0
    assert out["exposure"] == {"skipped": "--no-exposure"}


def test_exposure_without_condition_files_fails_loudly(rcfg):
    for code in CONDS:
        for seed in SEEDS:
            _write_run(rcfg, code, seed, {"u00": 0.3})

    with pytest.raises(FileNotFoundError):
        rb.evaluate(rcfg, ["high"])
