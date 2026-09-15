"""M1 / M2 / M3: elle kurulmus kucuk senaryolar, beklenen sayilar kurgudan belli.

Kosu dosyalari `run_experiment`in yazdigi bicimde (rapor JSON + kullanici basi parquet),
gecmis `conditions`in yazdigi C0.inter bicimde, alt kategori metadata parquet'inden.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.data.metadata import meta_parquet_path
from gift_contamination.recsys import marketing_metrics as mm
from gift_contamination.recsys.conditions import condition_path
from gift_contamination.recsys.run_experiment import experiment_path, peruser_path
from gift_contamination.utils.io import write_json

K = 20
DAY_MS = 86_400_000
CATS = {"P": "Puzzles", "G": "Games", "D": "Dolls", "B": "Blocks"}


@pytest.fixture
def mcfg(cfg: Config) -> Config:
    d = cfg._data
    d["dataset"]["categories"] = {"high": "raw_review_Toys_and_Games"}
    d["seeds"] = [42]
    d["experiment"] = {"conditions": ["C0", "C1", "C4", "C1b", "C4b", "C3"], "categories": ["high"],
                       "models": ["SASRec"], "topk": [K], "metrics": ["Recall"],
                       "bootstrap_iters": 50, "ci": 0.95}
    d["marketing"] = {"subcategory_level": 1, "m2_max_self_after_gift": 10, "m2_min_users_per_bucket": 1}
    # katalog: her alt kategoriden 20 urun ("P00".."P19")
    rows = [(f"{kod}{i:02d}", ["Toys & Games", ad]) for kod, ad in CATS.items() for i in range(K)]
    meta = meta_parquet_path(cfg, "high")
    meta.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=["parent_asin", "categories"], orient="row").write_parquet(meta)
    return cfg


def write_c0(cfg, histories: dict[str, list[tuple[str, str]]]):
    """histories[user] = [(item, label), ...] kronolojik; sonuna valid + self test eklenir."""
    rows = []
    for u, gecmis in histories.items():
        seq = [(i, lab, "train") for i, lab in gecmis[:-1]] + [(gecmis[-1][0], gecmis[-1][1], "valid")]
        seq.append(("B19", "self", "test"))
        for t, (item, lab, split) in enumerate(seq):
            rows.append((u, item, float(1_600_000_000_000 + t * DAY_MS), lab, split))
    dest = condition_path(cfg, "high", "C0")
    dest.parent.mkdir(parents=True, exist_ok=True)
    (pl.DataFrame(rows, schema=["user_id:token", "item_id:token", "timestamp", "label:token", "split"],
                  orient="row").write_csv(dest, separator="\t"))


def lst(**counts) -> list[str]:
    """lst(P=10, D=10) -> 10 Puzzles + 10 Dolls urunu (toplam K)."""
    out = [f"{kod}{i:02d}" for kod, n in counts.items() for i in range(n)]
    assert len(out) == K
    return out


def write_run(cfg, code, lists: dict[str, list[str]], recall: dict[str, float] | None = None):
    users = sorted(lists)
    pu = peruser_path(cfg, "high", code, "SASRec", 42)
    pu.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": users, f"recall@{K}": [(recall or {}).get(u, 0.0) for u in users],
                  "topk_items": [lists[u] for u in users]}).write_parquet(pu)
    write_json({"meta": {"label_source": "distilled"}, "split": {}, "test": {}},
               experiment_path(cfg, "high", code, "SASRec", 42))


# ------------------------------------------------------------------ M1
def test_m1_counts_only_subcategories_that_entered_through_gifts(mcfg):
    write_c0(mcfg, {
        "u1": [("P00", "gift_given"), ("D00", "self")],                       # S_dar = {Puzzles}
        "u2": [("P01", "gift_given"), ("P02", "self"), ("B00", "self")],      # Puzzles'ta self var -> bos
        "u3": [("G00", "gift_given"), ("G01", "household"), ("D01", "self")],  # dar bos; genis {Games}
    })
    write_run(mcfg, "C0", {"u1": lst(P=10, D=10), "u2": lst(P=20), "u3": lst(G=8, D=12)})
    write_run(mcfg, "C1", {"u1": lst(P=2, D=18), "u2": lst(P=20), "u3": lst(G=8, D=12)})
    write_run(mcfg, "C4", {"u1": lst(P=6, D=14), "u2": lst(P=20), "u3": lst(G=8, D=12)})
    write_run(mcfg, "C3", {"u1": lst(P=4, D=16), "u2": lst(P=20), "u3": lst(G=8, D=12)})
    write_run(mcfg, "C1b", {"u1": lst(P=2, D=18), "u2": lst(P=20), "u3": lst(G=2, D=18)})
    write_run(mcfg, "C4b", {"u1": lst(P=2, D=18), "u2": lst(P=20), "u3": lst(G=2, D=18)})

    m1 = mm.evaluate(mcfg, ["high"])["results"]["Toys_and_Games/SASRec"]["M1_waste_share"]

    assert m1["C0-C1"]["n_exposed_users"] == 1                 # yalnizca u1
    assert m1["C0-C1"]["mean_a"] == pytest.approx(0.5)
    assert m1["C0-C1"]["diff"] == pytest.approx(0.4)
    assert m1["C4-C1"]["diff"] == pytest.approx(0.2)
    assert m1["C0-C3"]["diff"] == pytest.approx(0.3)
    assert m1["C0-C1b"]["n_exposed_users"] == 2                # u1 + u3 (household genis eksende hediye)
    assert m1["C0-C1b"]["diff"] == pytest.approx(((0.5 - 0.1) + (0.4 - 0.1)) / 2)


# ------------------------------------------------------------------ M2
def test_m2_recovers_a_known_half_life(mcfg):
    """Fazla pay 0.8, 0.4, 0.2, 0.1, 0.05 -> A = 0.8, yari omur = 1 etkilesim."""
    gecmisler, c0, c1 = {}, {}, {}
    for n, adet in enumerate((16, 8, 4, 2, 1)):
        for tekrar in range(3):
            u = f"m{n}_{tekrar}"
            gecmisler[u] = [("D00", "self"), ("P00", "gift_given")] + [(f"D{j + 1:02d}", "self") for j in range(n)]
            if n == 0:
                gecmisler[u].append(("B00", "unclear"))   # valid satiri; self degil, n'ye sayilmaz
            c0[u] = lst(P=adet, D=K - adet)
            c1[u] = lst(D=K)
    write_c0(mcfg, gecmisler)
    for code, lists in (("C0", c0), ("C1", c1)):
        write_run(mcfg, code, lists)

    m2 = mm.evaluate(mcfg, ["high"])["results"]["Toys_and_Games/SASRec"]["M2_half_life"]

    assert m2["n_users"] == 15
    assert [b["excess"] for b in m2["buckets"][:5]] == pytest.approx([0.8, 0.4, 0.2, 0.1, 0.05])
    assert m2["fit"]["A"] == pytest.approx(0.8, rel=1e-4)
    assert m2["fit"]["half_life"] == pytest.approx(1.0, rel=1e-4)
    assert m2["median_gap_days"] == pytest.approx(1.0)
    assert m2["half_life_weeks"] == pytest.approx(1 / 7, rel=1e-4)
    assert m2["sequence_aware_model"] is True


def test_m2_half_life_is_undefined_without_excess(mcfg):
    gecmisler, lists = {}, {}
    for n in range(4):
        u = f"z{n}"
        gecmisler[u] = [("P00", "gift_given")] + [(f"D{j:02d}", "self") for j in range(n + 1)]
        lists[u] = lst(P=5, D=15)
    write_c0(mcfg, gecmisler)
    write_run(mcfg, "C0", lists)
    write_run(mcfg, "C1", lists)   # ayni liste: fazla pay 0

    m2 = mm.evaluate(mcfg, ["high"])["results"]["Toys_and_Games/SASRec"]["M2_half_life"]

    assert m2["fit"]["half_life"] is None
    assert m2["half_life_weeks"] is None


def test_fit_half_life_direct():
    n = [0, 1, 2, 3, 4]
    e = [0.5 * math.exp(-0.25 * x) for x in n]

    f = mm.fit_half_life(n, e, [10] * 5)

    assert f["half_life"] == pytest.approx(math.log(2) / 0.25, rel=1e-4)


# ------------------------------------------------------------------ M3
def test_m3_splits_the_placebo_contrast_by_history_length(mcfg):
    gecmisler = {f"s{n}": [(f"D{j:02d}", "self") for j in range(n)] for n in (2, 3, 4, 5, 6, 7)}
    write_c0(mcfg, gecmisler)
    users = sorted(gecmisler)
    liste = {u: lst(D=K) for u in users}
    uzun = {"s6": 1.0, "s7": 1.0}
    write_run(mcfg, "C1", liste, recall=uzun)
    write_run(mcfg, "C4", liste)

    m3 = mm.evaluate(mcfg, ["high"])["results"]["Toys_and_Games/SASRec"]["M3_segments"]

    assert m3["segment_sizes"] == {"short": 2, "medium": 2, "long": 2}
    seg = m3["C1-C4"]["segments"]
    assert seg["long"]["metrics"][f"recall@{K}"]["diff"] == pytest.approx(1.0)
    assert seg["long"]["metrics"][f"recall@{K}"]["interpretation"] == "alt_sinir"
    assert seg["short"]["metrics"][f"recall@{K}"]["interpretation"] == "saptanamadi"
    assert "skipped" in m3["C1b-C4b"]


def test_output_is_reportable_only_for_real_labels(mcfg):
    write_c0(mcfg, {"u1": [("P00", "gift_given"), ("D00", "self")]})
    write_run(mcfg, "C0", {"u1": lst(D=K)})

    out = mm.evaluate(mcfg, ["high"])

    assert out["meta"]["reportable"] is True
    assert out["results"]["Toys_and_Games/SASRec"]["interpretable"] is None   # gate2.json yok
