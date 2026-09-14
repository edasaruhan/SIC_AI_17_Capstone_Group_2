"""Sizinti testleri. CLAUDE.md 10 bunu da **kritik** diye isaretliyor.

Sizinti butun metrikleri SISIRIR ve sonucu iyi gosterir - yani kendiliginden
fark edilmez. Uc yolu var ve ucu de burada kapatiliyor:

  1. Zaman sizintisi. Rastgele bolme gelecegi egitime sokar.
  2. Test item'inin egitimde gecmesi.
  3. Test item'inin `self` OLMAMASI. Iddia "hediye alimlari kisinin KENDI
     tercihine dair tahmini bozuyor"; test item'i da hediyeyse olculen sey
     bu degildir (CLAUDE.md 5, kural 2).
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.recsys.atomic import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    assign_splits,
    build_atomic,
    eval_eligible,
    load_atomic,
)
from gift_contamination.recsys.conditions import (
    CONDITIONS,
    SHADOW_SUFFIX,
    apply_condition,
    real_item,
)

SEED = 42


def _sequences() -> pl.DataFrame:
    """Iki kullanici; zaman damgalari KASITLI olarak karisik sirada.

    Girdinin sirali gelmesine guvenen bir bolme burada kirilir.
    """
    rows = [
        ("u1", "i3", 300, 3, "self"),
        ("u1", "i1", 100, 1, "self"),
        ("u1", "i2", 200, 2, "gift_given"),
        ("u1", "i4", 400, 4, "self"),
        ("u2", "i9", 900, 9, "gift_given"),
        ("u2", "i7", 700, 7, "self"),
        ("u2", "i8", 800, 8, "self"),
    ]
    return pl.DataFrame(
        rows,
        schema=["user_id", "item_id", "timestamp", "row_id", "purchase_type"],
        orient="row",
    )


# ------------------------------------------------------------ zaman bazlilik
def test_the_split_is_temporal_not_random():
    """Test satiri kullanicinin EN SON etkilesimi olmali.

    Rastgele bolme gelecegi egitim setine sizdirir ve butun metrikleri sisirir.
    """
    out = assign_splits(_sequences())

    for user in ("u1", "u2"):
        kisi = out.filter(pl.col("user_id") == user)
        test_ts = kisi.filter(pl.col("split") == SPLIT_TEST)["timestamp"].item()
        valid_ts = kisi.filter(pl.col("split") == SPLIT_VALID)["timestamp"].item()
        egitim = kisi.filter(pl.col("split") == SPLIT_TRAIN)["timestamp"].to_list()
        assert test_ts == kisi["timestamp"].max()
        assert valid_ts < test_ts
        assert all(t < valid_ts for t in egitim)


def test_the_split_does_not_depend_on_input_order():
    """Ayni veri, karisik sira -> AYNI bolme.

    Kararsiz siralama ayni veriden iki farkli test seti uretir ve kosullar
    arasi karsilastirmayi sessizce bozar.
    """
    df = _sequences()

    ilk = assign_splits(df).sort("row_id")
    karisik = assign_splits(df.sample(fraction=1.0, seed=7, shuffle=True)).sort("row_id")

    assert ilk.select("row_id", "split").equals(karisik.select("row_id", "split"))


def test_ties_on_timestamp_are_broken_stably():
    """Ayni milisaniyeye dusen iki etkilesimde sira `row_id` ile kararli."""
    esit = pl.DataFrame(
        [
            ("u1", "i1", 100, 1, "self"),
            ("u1", "i2", 100, 2, "self"),
            ("u1", "i3", 100, 3, "self"),
        ],
        schema=["user_id", "item_id", "timestamp", "row_id", "purchase_type"],
        orient="row",
    )

    a = assign_splits(esit)
    b = assign_splits(esit.reverse())

    assert a.sort("row_id").select("row_id", "split").equals(
        b.sort("row_id").select("row_id", "split")
    )


# --------------------------------------------------- test item egitimde mi
@pytest.mark.parametrize("code", CONDITIONS)
def test_the_test_interaction_never_appears_in_training(code: str):
    """Ayni (kullanici, GERCEK urun) cifti hem testte hem egitimde olamaz.

    Gercek kimlik uzerinden: C3'te egitimdeki `i::gift` ile testteki `i` ayni
    urundur - sonek bir sizintiyi gizlememeli.
    """
    df = assign_splits(_sequences()).rename({"purchase_type": "label"})

    out, _ = apply_condition(df, code, seed=SEED)

    def ciftler(split: str) -> set:
        return set(
            out.filter(pl.col("split") == split)
            .select("user_id", real_item(pl.col("item_id")))
            .iter_rows()
        )

    assert not (ciftler(SPLIT_TEST) & ciftler(SPLIT_TRAIN))


def test_shadow_items_never_reach_test_or_validation():
    """C3'un golge kimligi YALNIZCA egitimde. Test urunu hep gercek ve `self`."""
    df = assign_splits(_sequences()).rename({"purchase_type": "label"})
    # u2'nin egitimdeki hediyesi + u1'in egitimdeki hediyesi golge olacak
    out, report = apply_condition(df, "C3", seed=SEED)

    assert report["n_shadow_rows"] > 0
    disari = out.filter(pl.col("split") != SPLIT_TRAIN)
    assert not disari["item_id"].str.ends_with(SHADOW_SUFFIX).any()
    # Degerlendirme kurali golge donusumunden etkilenmiyor
    assert set(eval_eligible(out, "label").to_list()) == set(eval_eligible(df, "label").to_list())


# ------------------------------------------------- test item'i `self` olmali
def test_only_users_whose_last_purchase_is_self_are_evaluated():
    """CLAUDE.md 5, kural 2 - DEGISTIRILEMEZ.

    u2'nin son alimi hediye; degerlendirmeden CIKAR ama evrende kalir.
    """
    df = _sequences().with_columns(
        pl.when((pl.col("user_id") == "u2") & (pl.col("timestamp") == 800))
        .then(pl.lit("gift_given"))
        .otherwise(pl.col("purchase_type"))
        .alias("purchase_type")
    )

    out = assign_splits(df)
    uygun = set(eval_eligible(out).to_list())

    assert uygun == {"u1"}
    # Evrenden DUSMEDI - yalnizca degerlendirmeden cikti
    assert set(out["user_id"].unique()) == {"u1", "u2"}


def test_a_fully_gifted_sequence_is_excluded_from_evaluation():
    """Butun sekansi hediye olan kullanici (CLAUDE.md 9)."""
    hepsi_hediye = pl.DataFrame(
        [
            ("u9", "i1", 100, 1, "gift_given"),
            ("u9", "i2", 200, 2, "gift_given"),
            ("u9", "i3", 300, 3, "gift_given"),
        ],
        schema=["user_id", "item_id", "timestamp", "row_id", "purchase_type"],
        orient="row",
    )

    out = assign_splits(hepsi_hediye)

    assert eval_eligible(out).len() == 0


# ------------------------------------------------------- uctan uca (fixture)
def test_atomic_file_round_trips_with_the_split_column(cfg: Config):
    """`split` dosyaya YAZILMALI.

    Yazilmazsa `conditions` bolmeyi her kosulda yeniden hesaplamak zorunda
    kalir ve C0 ile C1 farkli test setleri uzerinde karsilastirilir.
    """
    from gift_contamination.analysis.keyword_scan import scan_category
    from gift_contamination.data.preprocess import build_clean

    build_clean(cfg, "pilot")
    scan_category(cfg, "pilot")

    build_atomic(cfg, "pilot")
    out = load_atomic(cfg, "pilot")

    assert set(out.columns) == {"user_id", "item_id", "timestamp", "label", "split"}
    assert set(out["split"].unique()) <= {SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST}
    # Her kullanicinin tam olarak bir test satiri var
    test = out.filter(pl.col("split") == SPLIT_TEST)
    assert test.height == out["user_id"].n_unique()
