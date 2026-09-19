"""Sonuc figurleri: raporlanmis JSON bicimiyle uretiliyor, girdi yoksa atlaniyor.

Gorsel dogruluk testle olculemez (figurler duman ciktisiyla elle incelendi, DECISIONS);
burada sozlesme sinaniyor: dogru dosyalar, eksik girdide sessiz cokme yok, yol sizmiyor.
"""

from __future__ import annotations

import pytest

from gift_contamination.analysis import result_figures as rf
from gift_contamination.config import Config
from gift_contamination.utils.io import write_json

CI = [-0.01, 0.02]


@pytest.fixture
def fcfg(cfg: Config) -> Config:
    d = cfg._data
    d["dataset"]["categories"] = {"high": "raw_review_Toys_and_Games",
                                  "low": "raw_review_Grocery_and_Gourmet_Food"}
    d["experiment"] = {"categories": ["high", "low"], "models": ["SASRec", "BPR"]}
    d["marketing"] = {"m2_max_self_after_gift": 3}
    return cfg


def _stats(interpretable=True, reportable=True):
    kontrast = {"primary_metric": "recall@10", "metrics": {"recall@10": {"diff": 0.005, "ci": CI}}}
    blok = {"interpretable": interpretable,
            "contrasts": {k: kontrast for k, _ in rf.F21_ROWS}}
    return {"meta": {"reportable": reportable},
            "results": {f"{s}/{m}": blok for s in ("Toys_and_Games", "Grocery_and_Gourmet_Food")
                        for m in ("SASRec", "BPR")}}


def _mm():
    kova = [{"n_self_after_gift": i, "n_users": 50, "excess": 0.08 / 2 ** i, "used_in_fit": True}
            for i in range(3)] + [{"n_self_after_gift": ">=3", "n_users": 0, "excess": None, "used_in_fit": False}]
    blok = {"interpretable": None,
            "M1_waste_share": {k: {"diff": 0.01, "ci": CI} for k, _ in rf.F23_ROWS},
            "M2_half_life": {"buckets": kova, "fit": {"A": 0.08, "lambda": 0.693, "half_life": 1.0},
                             "half_life_weeks": 0.4}}
    return {"meta": {"reportable": True, "k": 10}, "results": {"Toys_and_Games/SASRec": blok}}


def _distill():
    eksen = {"C1": {"f1": {"value": 0.88, "ci95": [0.85, 0.9]}}, "C1b": {"f1": {"value": 0.9, "ci95": [0.88, 0.92]}}}
    return {"meta": {"model": r"C:\Users\someone\models\ModernBERT-base", "backend": "hf"},
            "student_vs_teacher_holdout": {"axes": eksen},
            "student_vs_teacher_holdout_in_kcore": {"n": 0, "skipped": True},
            "student_vs_human": {"axes": eksen},
            "criteria": {"1_c1_axis_vs_teacher": {"threshold": 0.85},
                         "3_c1_axis_vs_human_not_worse_than_teacher": {"teacher_f1": 0.73, "max_drop": 0.05}},
            "verdict": "PASS"}


def test_nothing_is_rendered_without_inputs(fcfg):
    assert rf.render_all(fcfg) == []


def test_all_four_figures_render_from_report_json(fcfg):
    write_json(_distill(), fcfg.path("results", "distill_report_base.json"))
    write_json(_stats(interpretable=False), fcfg.path("results", "experiment_stats.json"))
    write_json(_mm(), fcfg.path("results", "marketing_metrics.json"))

    paths = rf.render_all(fcfg)

    assert sorted(p.name for p in paths) == [
        "F20_distill_fidelity_base.png", "F21_condition_contrasts.png",
        "F22_m2_half_life.png", "F23_m1_waste_share.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_a_figure_without_data_is_skipped_not_drawn_empty(fcfg, tmp_path):
    bos = {"meta": {"reportable": True, "k": 10}, "results": {}}

    assert rf.fig_contrasts(fcfg, bos, tmp_path / "a.png") is None
    assert rf.fig_m1(fcfg, bos, tmp_path / "b.png") is None
    assert rf.fig_m2(fcfg, bos, tmp_path / "c.png") is None
    assert not list(tmp_path.glob("*.png"))


def test_local_model_paths_do_not_leak_into_figures():
    assert rf.display_model_name(r"C:\Users\someone\models\ModernBERT-base") == "ModernBERT-base"
    assert rf.display_model_name("/kaggle/working/models/base/") == "base"
    assert rf.display_model_name("answerdotai/ModernBERT-base") == "answerdotai/ModernBERT-base"


def test_m2_legend_shows_the_ci_and_does_not_read_an_unbounded_one_as_a_duration():
    sinirli = {"fit": {"half_life": 0.657}, "half_life_interactions_ci": [0.527, 0.816],
               "half_life_weeks": 2.39}
    assert rf.m2_label("Toys", sinirli) == "Toys · half-life 0.7 [0.5–0.8] interactions (≈2.4 wk)"

    # GA'nin ust ucu uyumun lambda sinirinda (ln2 / 1e-6): ust uc sinirsiz, hafta yazilmaz.
    sinirsiz = {"fit": {"half_life": 43.26}, "half_life_interactions_ci": [19.43, 693147.18],
                "half_life_weeks": 157.6}
    assert rf.m2_label("Toys", sinirsiz) == "Toys · half-life 43.3 [19.4–∞] interactions"

    assert rf.m2_label("Toys", {"fit": {"half_life": None}}) == "Toys · half-life undefined"
