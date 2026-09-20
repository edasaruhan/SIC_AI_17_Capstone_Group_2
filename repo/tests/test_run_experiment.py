"""Deney kosucusunun torch/RecBole GEREKTIRMEYEN parcalari.

RecBole'la birlikte sinanan kisim (maske + kullanici basi ortalama == RecBole
toplami) kosunun kendisinde KONTROL olarak duruyor: esit degilse kosu hata verir.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from gift_contamination.recsys import run_experiment
from gift_contamination.recsys.conditions import SHADOW_SUFFIX
from gift_contamination.recsys.run_experiment import (
    UserItemMask,
    checkpoint_dir,
    epochs_ran,
    eval_mask_pairs,
    hash_test_pairs,
    per_user_topk_metrics,
    sequential_parts,
    shadow_item_ids,
    stub_tensorboard_if_broken,
)

G = SHADOW_SUFFIX


def _frame(rows):
    """(user, item, split, label) - dosya sirasinda, zaman = sira."""
    return pl.DataFrame(
        [(u, i, float(t), s, lab) for t, (u, i, s, lab) in enumerate(rows)],
        schema=["user_id", "item_id", "timestamp", "split", "label"], orient="row",
    )


def _as_dict(part):
    return {(r["user_id"], r["item_id"]): r["item_id_list"] for r in part.iter_rows(named=True)}


def test_per_user_metrics_match_a_hand_calculation():
    """RecBole 1.2.0 formulleri, elle: dort kullanici, top-3."""
    pos_idx = np.array([
        [1, 0, 0],   # A: tek pozitif 1. sirada
        [0, 1, 0],   # B: tek pozitif 2. sirada
        [0, 0, 0],   # C: isabet yok
        [1, 0, 1],   # D: iki pozitif, 1. ve 3. sirada
    ])
    pos_len = np.array([1, 1, 1, 2])

    m = per_user_topk_metrics(pos_idx, pos_len, [2, 3])

    np.testing.assert_allclose(m["recall@2"], [1, 1, 0, 0.5])
    np.testing.assert_allclose(m["recall@3"], [1, 1, 0, 1])
    np.testing.assert_allclose(m["hit@2"], [1, 1, 0, 1])
    idcg2 = 1 + 1 / np.log2(3)
    np.testing.assert_allclose(m["ndcg@2"], [1, 1 / np.log2(3), 0, 1 / idcg2])
    np.testing.assert_allclose(m["ndcg@3"], [1, 1 / np.log2(3), 0, (1 + 1 / np.log2(4)) / idcg2])


def test_the_runner_imports_without_main_environment_packages():
    """Kosucu `.venv-recbole` altinda calisiyor ve orada pydantic yok.

    2026-09-14'te `conditions` tanimi `detection.schema`'dan (pydantic) okumaya
    baslayinca kosucu RecBole ortaminda ilk satirda cokuyordu; bunu ancak RecBole
    duman testi yakaladi. Taze bir yorumlayicida import zincirine bakiyoruz.
    """
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    kod = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import gift_contamination.recsys.run_experiment; "
        "bad = [m for m in ('pydantic', 'matplotlib', 'statsmodels', 'scipy') if m in sys.modules]; "
        "assert not bad, bad" % src
    )
    sonuc = subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True)
    assert sonuc.returncode == 0, sonuc.stderr[-800:]


def test_a_broken_tensorboard_is_replaced_by_a_silent_writer(monkeypatch):
    """Kaggle 2026-09-15: protobuf uyusmazligi RecBole'un import'unu kiriyordu.

    RecBole `torch.utils.tensorboard`i kosulsuz import ediyor; biz TensorBoard
    kaydini okumuyoruz. Kirik import kosuyu dusurmesin diye yerine hicbir sey
    yapmayan bir yazici konuyor - ve bu RAPORA yaziliyor, sessiz kalmiyor.
    """
    import sys
    import types

    sahte_torch = types.ModuleType("torch")
    sahte_utils = types.ModuleType("torch.utils")
    sahte_torch.utils = sahte_utils
    monkeypatch.setitem(sys.modules, "torch", sahte_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", sahte_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # import'u kirar
    monkeypatch.setattr(run_experiment, "_TB_STUB_SEBEP", None)

    sebep = stub_tensorboard_if_broken()

    assert sebep, "kirik import fark edilmedi"
    yazici = sys.modules["torch.utils.tensorboard"].SummaryWriter("bir/klasor")
    assert yazici.add_scalar("kayip", 1.0, 1) is None  # RecBole'un cagirdigi bicim
    assert yazici.close() is None
    assert sahte_utils.tensorboard is sys.modules["torch.utils.tensorboard"]
    # Ayni surecteki ikinci kosu "ok" degil, ayni sebebi gormeli.
    assert stub_tensorboard_if_broken() == sebep


def test_a_working_tensorboard_is_left_alone(monkeypatch):
    import sys
    import types

    sahte_torch = types.ModuleType("torch")
    sahte_utils = types.ModuleType("torch.utils")
    calisan = types.ModuleType("torch.utils.tensorboard")
    calisan.SummaryWriter = object
    sahte_torch.utils = sahte_utils
    sahte_utils.tensorboard = calisan
    monkeypatch.setitem(sys.modules, "torch", sahte_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", sahte_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", calisan)
    monkeypatch.setattr(run_experiment, "_TB_STUB_SEBEP", None)

    assert stub_tensorboard_if_broken() is None
    assert sys.modules["torch.utils.tensorboard"] is calisan


def test_a_missing_torch_is_not_papered_over(monkeypatch):
    """torch'un kendisi yoksa yama HATAYI GIZLEMEZ - gercek import patlasin."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.utils", None)
    monkeypatch.setattr(run_experiment, "_TB_STUB_SEBEP", None)

    assert stub_tensorboard_if_broken() is None
    assert "torch.utils.tensorboard" not in sys.modules
    assert run_experiment._TB_STUB_SEBEP is None


def test_shadow_ids_are_found_by_suffix_only():
    tokens = ["[PAD]", "B01", f"B01{SHADOW_SUFFIX}", "gift", f"B02{SHADOW_SUFFIX}"]

    assert shadow_item_ids(tokens).tolist() == [2, 4]
    assert shadow_item_ids(["[PAD]", "B01"]).tolist() == []


def test_single_positive_recall_equals_hit():
    """Testte kullanici basi tek pozitif var: Recall@K ve Hit@K ayni sayi olmali."""
    rng = np.random.default_rng(0)
    pos_idx = np.zeros((50, 10), dtype=int)
    for u in range(50):
        r = rng.integers(0, 15)
        if r < 10:
            pos_idx[u, r] = 1

    m = per_user_topk_metrics(pos_idx, np.ones(50), [5, 10])

    for k in (5, 10):
        assert m[f"recall@{k}"] == pytest.approx(m[f"hit@{k}"])


# ------------------------------------------------ sirali benchmark dosyalari
def test_sequential_parts_follow_recbole_augmentation_and_keep_the_last_items():
    df = _frame([
        ("u1", "a", "train", "self"), ("u1", "b", "train", "self"),
        ("u1", "c", "train", "gift_given"), ("u1", "d", "train", "self"),
        ("u1", "e", "valid", "self"), ("u1", "f", "test", "self"),
    ])

    parts, meta = sequential_parts(df, max_len=2)

    # ilk egitim urununun gecmisi yok -> satir yok; gecmis en fazla 2, SONDAN
    assert _as_dict(parts["train"]) == {("u1", "b"): "a", ("u1", "c"): "a b", ("u1", "d"): "b c"}
    assert _as_dict(parts["valid"]) == {("u1", "e"): "c d"}
    assert _as_dict(parts["test"]) == {("u1", "f"): "d e"}
    assert parts["train"]["timestamp"].to_list() == [1.0, 2.0, 3.0]
    assert meta == {"n_valid_dropped_empty_history": 0, "max_item_list_length": 2}


def test_empty_valid_history_is_dropped_and_counted_but_test_pairs_stay():
    """C1 kullanicinin butun egitim satirlarini silmis: valid duser, test kalir."""
    df = _frame([
        ("u2", "g", "valid", "self"), ("u2", "h", "test", "self"),
        ("u3", "x", "train", "self"), ("u3", "y", "train", "self"),
        ("u3", "z", "valid", "self"), ("u3", "w", "test", "gift_given"),
    ])

    parts, meta = sequential_parts(df, max_len=50)

    assert meta["n_valid_dropped_empty_history"] == 1
    assert _as_dict(parts["valid"]) == {("u3", "z"): "x y"}
    # test yalnizca `self`: u3'un hediye test satiri dosyaya girmez
    assert _as_dict(parts["test"]) == {("u2", "h"): "g"}


def test_shadow_tokens_stay_in_the_history_as_their_own_ids():
    df = _frame([
        ("u4", f"p{G}", "train", "gift_given"), ("u4", "q", "train", "self"),
        ("u4", "r", "valid", "self"), ("u4", "s", "test", "self"),
    ])

    parts, _ = sequential_parts(df, max_len=50)

    assert _as_dict(parts["train"]) == {("u4", "q"): f"p{G}"}
    assert _as_dict(parts["test"]) == {("u4", "s"): f"p{G} q r"}


def test_a_test_row_without_history_stops_the_run():
    df = _frame([("u5", "a", "test", "self")])

    with pytest.raises(RuntimeError, match="bos gecmisli test"):
        sequential_parts(df, max_len=50)


# ------------------------------------------ C0 alinmis urun maskesi (DECISIONS)
C0_ROWS = [
    ("u1", "a", "train", "self"), ("u1", "b", "train", "gift_given"),
    ("u1", "c", "valid", "self"), ("u1", "d", "test", "self"),
    ("u2", "a", "train", "gift_given"), ("u2", "e", "train", "self"),
    ("u2", "f", "valid", "self"), ("u2", "g", "test", "self"),
]


def test_mask_pairs_are_the_training_pairs_the_condition_lost():
    c0 = _frame(C0_ROWS)
    c1 = c0.filter(~((pl.col("split") == "train") & (pl.col("label") == "gift_given")))

    assert eval_mask_pairs(c0, c0).height == 0
    assert eval_mask_pairs(c0, c1).rows() == [("u1", "b"), ("u2", "a")]


def test_mask_pairs_for_c3_are_the_real_ids_behind_shadow_tokens():
    c0 = _frame(C0_ROWS)
    c3 = c0.with_columns(
        pl.when((pl.col("split") == "train") & (pl.col("label") == "gift_given"))
        .then(pl.col("item_id") + G).otherwise(pl.col("item_id")).alias("item_id")
    )

    # golge kimlik zaten global maskeleniyor; gercek kimlik bu kullanici icin maskeye girer
    assert eval_mask_pairs(c0, c3).rows() == [("u1", "b"), ("u2", "a")]


def test_mask_pairs_ignore_validation_and_test_rows():
    c0 = _frame(C0_ROWS)
    kosul = c0.filter(pl.col("split") == "train")   # valid/test yok sayilmali

    assert eval_mask_pairs(c0, kosul).height == 0


def test_test_pair_hash_ignores_row_order_and_extra_columns():
    a = pl.DataFrame({"user_id": ["u1", "u2"], "item_id": ["x", "y"], "timestamp": [1.0, 2.0]})
    b = pl.DataFrame({"user_id": ["u2", "u1"], "item_id_list": ["p", "q"], "item_id": ["y", "x"]})
    c = pl.DataFrame({"user_id": ["u1", "u2"], "item_id": ["x", "z"]})

    assert hash_test_pairs(a) == hash_test_pairs(b)
    assert hash_test_pairs(a) != hash_test_pairs(c)


def test_user_item_mask_lookup_returns_batch_rows():
    maske = UserItemMask(np.array([2, 0, 2, 1]), np.array([5, 3, 1, 7]), n_users=4)

    satir, urun = maske.lookup(np.array([2, 3, 0]))

    assert maske.n_pairs == 4
    assert satir.tolist() == [0, 0, 2]
    assert urun.tolist() == [1, 5, 3]
    bos_satir, bos_urun = maske.lookup(np.array([3]))
    assert bos_satir.tolist() == [] and bos_urun.tolist() == []


def test_parallel_runs_never_share_a_checkpoint_dir(cfg):
    # RecBole'un dosya adi yalnizca model + saniye: ayrim klasorden gelmek ZORUNDA.
    klasorler = [checkpoint_dir(cfg, "pilot", k, m, s)
                 for k in ("C0", "C1", "C3") for m in ("SASRec", "BPR") for s in (42, 1337)]

    assert len(set(klasorler)) == len(klasorler)
    assert all(not p.is_absolute() and p.parts[0] == "saved" and len(p.parts) == 2 for p in klasorler)


# --------------------------------------------------------- kac epoch kosuldu
class _Trainer:
    def __init__(self, loss_dict):
        self.train_loss_dict = loss_dict


def test_epochs_trained_counts_the_epochs_recbole_actually_ran():
    """Rapor bugune kadar yalnizca TAVANI yaziyordu, yani 72 kosunun hangisi
    erken durdu soylenemiyordu (SONUCLAR 7)."""
    assert epochs_ran(_Trainer({0: 1.0, 1: 0.8, 2: 0.7}), cap=300) == {
        "epochs_trained": 3, "hit_epoch_cap": False,
    }


def test_hitting_the_cap_is_flagged():
    """Tavana degen kosu 'model hala ogreniyordu, kesildi' demek; mutlak
    sayilar bu bilgi olmadan okunamaz."""
    assert epochs_ran(_Trainer({i: 1.0 for i in range(5)}), cap=5) == {
        "epochs_trained": 5, "hit_epoch_cap": True,
    }


def test_an_unknowable_epoch_count_is_recorded_as_unknown_not_as_zero():
    """Kayit alani, kapi degil: RecBole surumu alani tasimiyorsa kosu durmaz -
    ama 0 yazip 'hic egitilmedi' iddiasinda da bulunmaz."""
    assert epochs_ran(_Trainer({}), cap=300) == {
        "epochs_trained": None, "hit_epoch_cap": None,
    }
    assert epochs_ran(object(), cap=300)["epochs_trained"] is None
