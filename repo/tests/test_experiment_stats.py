"""Kapi 2 ve esli bootstrap: sentetik kosu dosyalariyla, RecBole'suz.

Kosu dosyalari `run_experiment`in yazdigi bicimde elle kuruluyor (rapor JSON +
kullanici basi parquet + kosul raporu); her senaryonun beklenen sonucu kurgudan belli.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gift_contamination.analysis import experiment_stats as es
from gift_contamination.config import Config
from gift_contamination.recsys.conditions import condition_report_path
from gift_contamination.recsys.run_experiment import experiment_path, peruser_path
from gift_contamination.utils.io import write_json

SEEDS = [42, 1337, 2024]
CORE = ["C0", "C1", "C4", "C1b", "C4b", "C3"]


@pytest.fixture
def ecfg(cfg: Config) -> Config:
    d = cfg._data
    d["dataset"]["categories"] = {"high": "raw_review_Toys_and_Games",
                                  "low": "raw_review_Grocery_and_Gourmet_Food"}
    d["seeds"] = SEEDS
    # Esikler configs/base.yaml ile AYNI; tekrar sayilari testte kucuk.
    d["experiment"] = {"conditions": CORE, "categories": ["high", "low"], "models": ["BPR"],
                       "topk": [10], "metrics": ["Recall"], "bootstrap_iters": 300, "ci": 0.95}
    d["gate2"] = {"metric": "Recall@10", "bootstrap_n": 300, "ci": 0.95, "seed_cv_max": 0.10}
    return cfg


def _base_values(n=400, seed=0):
    return np.random.default_rng(seed).binomial(1, 0.3, size=n).astype(float)


def write_run(cfg, role, code, seed, values, *, model="BPR", sha="ayni", label_source="distilled",
              aggregate=None, universe_ok=True):
    users = [f"u{i:04d}" for i in range(len(values))]
    pu = peruser_path(cfg, role, code, model, seed)
    pu.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": users, "recall@10": values}).write_parquet(pu)
    write_json({
        "meta": {"condition": code, "model": model, "seed": seed, "label_source": label_source,
                 "reportable": label_source in ("llm", "distilled")},
        "split": {"test_pairs_sha256": sha},
        "test": {"recall@10": float(np.mean(values)) if aggregate is None else aggregate},
        "peruser_file": pu.name,
    }, experiment_path(cfg, role, code, model, seed))
    write_json({"condition": code, "universe_is_subset_of_c0": universe_ok},
               condition_report_path(cfg, role, code))


def write_matrix(cfg, role, *, effect_c1=0.05, c4_shift=0.0, drop=(), **kw):
    base = _base_values()
    for seed, agg in zip(SEEDS, (0.30, 0.31, 0.29)):
        vals = {"C0": base, "C4": base + c4_shift, "C1": base + c4_shift + effect_c1,
                "C1b": base, "C4b": base, "C3": base + 0.01}
        for code, v in vals.items():
            if (code, seed) in drop:
                continue
            write_run(cfg, role, code, seed, v, aggregate=agg if code == "C0" else None, **kw)


# ------------------------------------------------------------------ bootstrap
def test_paired_bootstrap_recovers_a_known_difference():
    rng = np.random.default_rng(1)
    b = rng.normal(0.3, 0.2, size=3000)
    a = b + 0.1 + rng.normal(0, 0.2, size=3000)

    out = es.paired_bootstrap(a, b, names=["m"], n_boot=500, ci=0.95, seed=7)["m"]

    assert out["diff"] == pytest.approx(0.1, abs=0.01)
    assert out["ci"][0] < 0.1 < out["ci"][1]
    assert out["ci"][0] > 0
    assert es.paired_bootstrap(a, b, names=["m"], n_boot=500, ci=0.95, seed=7)["m"] == out


def test_paired_bootstrap_resamples_users_not_conditions():
    """ESLI olmasi testle kilitli olmali, yoksa GA'lar sessizce sisirilir.

    Onceki iki test eslesmeyen bir uygulamayla DA geciyordu: b'nin varyansi
    GA'yi 0,1'in etrafinda genisletirdi ama hala sifiri dislardi
    (denetim 2026-09-20). Burada kullanici basi fark SABIT: esli bootstrap
    her tekrarda tam olarak o sabiti bulur, GA genisligi sifirdir. Iki kosulu
    bagimsiz yeniden orneklenen bir uygulamada genislik b'nin varyansini
    tasir ve bu assert duser.
    """
    rng = np.random.default_rng(11)
    b = rng.normal(0.3, 0.5, size=2000)
    a = b + 0.05                          # her kullanicida AYNI fark

    out = es.paired_bootstrap(a, b, names=["m"], n_boot=500, ci=0.95, seed=9)["m"]

    assert out["diff"] == pytest.approx(0.05, abs=1e-12)
    genislik = out["ci"][1] - out["ci"][0]
    assert genislik < 1e-9, f"GA genisligi {genislik:.4f} - kosullar ayni orneklemi paylasmiyor"


def test_paired_bootstrap_without_a_difference_covers_zero():
    rng = np.random.default_rng(2)
    b = rng.normal(0.3, 0.2, size=3000)
    gurultu = rng.normal(0, 0.2, size=3000)
    a = b + gurultu - gurultu.mean()     # ortalama fark TAM sifir

    out = es.paired_bootstrap(a, b, names=["m"], n_boot=500, ci=0.95, seed=3)["m"]

    assert out["diff"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci"][0] < 0 < out["ci"][1]


def test_independent_bootstrap_difference_of_effects():
    rng = np.random.default_rng(4)
    x = rng.normal(0.05, 0.1, size=2000)
    y = rng.normal(0.0, 0.1, size=1500)

    out = es.independent_bootstrap_diff(x, y, names=["m"], n_boot=500, ci=0.95, seed=5)["m"]

    assert out["diff"] == pytest.approx(x.mean() - y.mean())
    assert out["ci"][0] < out["diff"] < out["ci"][1]
    assert out["ci"][0] > 0


def test_preregistered_interpretation_rule():
    assert es.interpret_placebo([0.01, 0.03]) == "alt_sinir"
    assert es.interpret_placebo([-0.01, 0.03]) == "saptanamadi"
    assert es.interpret_placebo([-0.03, -0.01]) == "negatif"


def test_metric_names_follow_the_runner():
    assert es.metric_key("Recall@10") == "recall@10"
    assert es.metric_key("HitRate@20") == "hit@20"
    assert es.metric_key("NDCG@10") == "ndcg@10"


def test_an_unmeasured_criterion_never_passes():
    assert es.verdict({"a": {"passed": True}, "b": {"passed": None}}) == "INCOMPLETE"
    assert es.verdict({"a": {"passed": False}, "b": {"passed": None}}) == "FAIL"
    assert es.verdict({"a": {"passed": True}, "b": {"passed": True}}) == "PASS"


# ------------------------------------------------------------------ Kapi 2
def test_gate2_passes_on_a_valid_setup(ecfg):
    write_matrix(ecfg, "high")

    gate, _ = es.evaluate(ecfg, ["high"])
    g = gate["results"]["Toys_and_Games/BPR"]

    assert g["verdict"] == "PASS", g["criteria"]
    assert g["criteria"]["4_seed_stability"]["cv"] == pytest.approx(np.std([0.30, 0.31, 0.29], ddof=1) / 0.30)
    assert gate["meta"]["reportable"] is True


def test_gate2_fails_when_the_placebo_beats_the_baseline(ecfg):
    write_matrix(ecfg, "high", c4_shift=0.2)

    gate, _ = es.evaluate(ecfg, ["high"])
    g = gate["results"]["Toys_and_Games/BPR"]

    assert g["criteria"]["3_placebo_not_better_than_c0"]["passed"] is False
    assert g["verdict"] == "FAIL"


def test_gate2_fails_when_test_pairs_differ(ecfg):
    write_matrix(ecfg, "high")
    base = _base_values()
    write_run(ecfg, "high", "C1", 42, base + 0.05, sha="baska")

    g = es.evaluate(ecfg, ["high"])[0]["results"]["Toys_and_Games/BPR"]

    assert g["criteria"]["1_same_test_pairs"]["passed"] is False
    assert g["verdict"] == "FAIL"


def test_gate2_is_incomplete_with_a_missing_seed(ecfg):
    write_matrix(ecfg, "high", drop={("C0", 2024)})

    g = es.evaluate(ecfg, ["high"])[0]["results"]["Toys_and_Games/BPR"]

    assert g["criteria"]["4_seed_stability"]["passed"] is None
    assert g["verdict"] == "INCOMPLETE"


def test_gate2_is_incomplete_without_a_core_condition(ecfg):
    write_matrix(ecfg, "high", drop={("C4b", s) for s in SEEDS})

    g = es.evaluate(ecfg, ["high"])[0]["results"]["Toys_and_Games/BPR"]

    assert g["criteria"]["1_same_test_pairs"]["passed"] is None
    assert g["criteria"]["3_placebo_not_better_than_c0"]["passed"] is None
    assert g["verdict"] == "INCOMPLETE"


# ------------------------------------------------------------------ karsitliklar
def test_contrasts_apply_the_rule_and_carry_the_gate(ecfg):
    write_matrix(ecfg, "high", effect_c1=0.05)

    _, stats = es.evaluate(ecfg, ["high"])
    r = stats["results"]["Toys_and_Games/BPR"]
    c1c4 = r["contrasts"]["C1-C4"]["metrics"]["recall@10"]

    assert r["interpretable"] is True
    assert c1c4["diff"] == pytest.approx(0.05)
    assert c1c4["interpretation"] == "alt_sinir"
    assert r["contrasts"]["C1b-C4b"]["metrics"]["recall@10"]["interpretation"] == "saptanamadi"
    assert "interpretation" not in r["contrasts"]["C3-C0"]["metrics"]["recall@10"]
    assert r["contrasts"]["C3-C0"]["metrics"]["recall@10"]["direction"] == "pozitif"
    assert r["contrasts"]["C1-C4"]["seeds"] == SEEDS


def test_contrasts_are_marked_uninterpretable_when_gate2_fails(ecfg):
    write_matrix(ecfg, "high", c4_shift=0.2)

    r = es.evaluate(ecfg, ["high"])[1]["results"]["Toys_and_Games/BPR"]

    assert r["gate2_verdict"] == "FAIL"
    assert r["interpretable"] is False


def test_dose_response_compares_the_two_categories(ecfg):
    write_matrix(ecfg, "high", effect_c1=0.05)
    write_matrix(ecfg, "low", effect_c1=0.0)

    dose = es.evaluate(ecfg, ["high", "low"])[1]["dose_response"]["BPR"]["C1-C4"]

    assert dose["x"] == "Toys_and_Games"
    assert dose["metrics"]["recall@10"]["diff"] == pytest.approx(0.05)


def test_proxy_runs_are_not_reportable(ecfg):
    write_matrix(ecfg, "high", label_source="proxy")

    gate, stats = es.evaluate(ecfg, ["high"])

    assert gate["meta"]["reportable"] is False
    assert stats["meta"]["reportable"] is False


def test_seed_user_sets_must_match(ecfg):
    write_matrix(ecfg, "high")
    write_run(ecfg, "high", "C4", 1337, _base_values(n=399))

    with pytest.raises(RuntimeError, match="kullanici kumesi"):
        es.evaluate(ecfg, ["high"])
