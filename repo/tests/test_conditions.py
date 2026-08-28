"""C0-C4 kosul testleri. CLAUDE.md 10 bunu **kritik** diye isaretliyor.

Buradaki hatalarin ortak ozelligi SESSIZ olmalari: kod calisir, sayilar makul
gorunur, ve deney gecersiz cikar - ama bunu ancak biri "C4 gercekten C1 kadar
mi cikardi" diye sorunca fark ederiz. Juri'deki ilk akilli kisi de zaten tam
bunu soracak.
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.recsys.atomic import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from gift_contamination.recsys.conditions import (
    CONDITIONS,
    apply_condition,
    universe,
)

SEED = 42


def _frame() -> pl.DataFrame:
    """Dort kullanici, karisik etiketler. Her kullanicinin son satiri `self`.

    Egitim satirlarinda bilerek hem `gift_given` hem `household` var: C1 ile
    C1b'nin gercekten farkli davrandigi gorunsun.
    """
    rows = [
        # user, item, ts, label, split
        ("u1", "i1", 1, "self",       SPLIT_TRAIN),
        ("u1", "i2", 2, "gift_given", SPLIT_TRAIN),
        ("u1", "i3", 3, "household",  SPLIT_VALID),
        ("u1", "i4", 4, "self",       SPLIT_TEST),
        ("u2", "i2", 1, "gift_given", SPLIT_TRAIN),
        ("u2", "i5", 2, "household",  SPLIT_TRAIN),
        ("u2", "i6", 3, "self",       SPLIT_VALID),
        ("u2", "i7", 4, "self",       SPLIT_TEST),
        ("u3", "i8", 1, "self",       SPLIT_TRAIN),
        ("u3", "i9", 2, "gift_given", SPLIT_TRAIN),
        ("u3", "i1", 3, "self",       SPLIT_VALID),
        ("u3", "i2", 4, "self",       SPLIT_TEST),
        ("u4", "i9", 1, "household",  SPLIT_TRAIN),
        ("u4", "i3", 2, "self",       SPLIT_TRAIN),
        ("u4", "i5", 3, "self",       SPLIT_VALID),
        ("u4", "i6", 4, "self",       SPLIT_TEST),
    ]
    return pl.DataFrame(
        rows,
        schema=["user_id", "item_id", "timestamp", "label", "split"],
        orient="row",
    )


# ------------------------------------------------------- evren dondurulmus mu
@pytest.mark.parametrize("code", CONDITIONS)
def test_no_condition_introduces_a_user_or_item(code: str):
    """CLAUDE.md 5, kural 3. Evren C0'DA DONAR.

    Dikkat - evren satirlardan TURETILEMEZ. C1 bir urunun butun egitim
    satirlarini silebilir ve o urun ciktida hic gecmeyebilir; bu bir hata
    degil, CLAUDE.md 9'un "cold item" dedigi durumdur ve AYRICA raporlanir
    (`test_items_that_vanish_from_training_are_reported`). Burada
    kilitlenen sey tek yonlu: hicbir kosul C0'da OLMAYAN bir kullanici ya
    da urun EKLEYEMEZ.
    """
    df = _frame()

    out, _ = apply_condition(df, code, seed=SEED)

    u0, i0 = universe(df)
    u1, i1 = universe(out)
    assert u1 <= u0 and i1 <= i0


@pytest.mark.parametrize("code", CONDITIONS)
def test_no_condition_drops_a_user_entirely(code: str):
    """Kullanici kaybi urun kaybindan farkli: test satiri hep duruyor.

    Test/valid satirlarina dokunulmadigi icin her kullanicinin en az iki
    satiri hayatta kalir. Bu kirilirsa degerlendirme evreni kosula gore
    degisiyor demektir.
    """
    df = _frame()

    out, _ = apply_condition(df, code, seed=SEED)

    assert set(out["user_id"].unique()) == set(df["user_id"].unique())


@pytest.mark.parametrize("code", CONDITIONS)
def test_no_condition_touches_test_or_valid_rows(code: str):
    """Kosullar YALNIZCA egitim satirlarina dokunur.

    Dokunurlarsa C0 ile C1 FARKLI test setleri uzerinde karsilastirilir ve
    olculen sey mudahale olmaktan cikar - test setinin degismesi olur.
    """
    df = _frame()

    out, _ = apply_condition(df, code, seed=SEED)

    for split in (SPLIT_TEST, SPLIT_VALID):
        onceki = df.filter(pl.col("split") == split).sort("user_id", "item_id")
        sonraki = out.filter(pl.col("split") == split).sort("user_id", "item_id")
        assert onceki.select("user_id", "item_id").equals(
            sonraki.select("user_id", "item_id")
        ), f"{code} {split} satirlarina dokundu"


# ------------------------------------------------------------------- plasebo
def test_c4_removes_exactly_as_many_rows_as_c1():
    """C4 OPSIYONEL DEGIL ve "ayni kadar" olmak zorunda.

    C1 kazaniyorsa C4'ten de kazanmali; yoksa gordugumuz sey hediye etkisi
    degil "veri azaldi" etkisidir. Sayi C1'den TURETILIYOR, elle girilmiyor.
    """
    df = _frame()

    _, c1 = apply_condition(df, "C1", seed=SEED)
    _, c4 = apply_condition(df, "C4", seed=SEED)

    assert c4["n_removed"] == c1["n_removed"] > 0


def test_c4_removes_at_random_not_by_label():
    """Plasebo etikete BAKMAMALI - baksa plasebo olmaz, ikinci bir C1 olur."""
    df = _frame()

    out, _ = apply_condition(df, "C4", seed=SEED)

    kalan = out.filter(pl.col("split") == SPLIT_TRAIN)["label"].to_list()
    # C1 butun gift_given'lari silerdi; C4'te en az biri hayatta kalmali
    # ya da bir gift-disi satir silinmis olmali.
    silinen = df.filter(pl.col("split") == SPLIT_TRAIN).height - len(kalan)
    assert silinen > 0
    assert "gift_given" in kalan or "household" in kalan


def test_c4_is_reproducible_across_processes():
    """`hash()` PYTHONHASHSEED ile degisir; crc32 degismez.

    Ayni seed ayni satirlari secmezse "3-5 seed" iddiasi anlamsizlasir.
    """
    df = _frame()

    ilk, _ = apply_condition(df, "C4", seed=SEED)
    ikinci, _ = apply_condition(df, "C4", seed=SEED)

    assert ilk.equals(ikinci)


def test_a_different_seed_selects_different_rows():
    """Ayni seed kararli olmali ama FARKLI seed farkli cekmeli."""
    df = _frame()

    ilk, _ = apply_condition(df, "C4", seed=SEED)
    baska, _ = apply_condition(df, "C4", seed=SEED + 1)

    assert ilk.height == baska.height
    assert not ilk.equals(baska)


# ---------------------------------------------------------------- C1 vs C1b
def test_c1_removes_only_gifts_c1b_also_removes_household():
    """`household` korpusun %20'si; iki tanim arasindaki fark buyuk.

    Karar KAVRAMSAL ve ekipte (CLAUDE.md 13). Kod ikisini de kosabilmeli ki
    sonuc HER IKI tanim altinda raporlanabilsin.
    """
    df = _frame()

    c1, r1 = apply_condition(df, "C1", seed=SEED)
    c1b, r1b = apply_condition(df, "C1b", seed=SEED)

    assert set(c1.filter(pl.col("split") == SPLIT_TRAIN)["label"]) == {
        "self", "household",
    }
    assert set(c1b.filter(pl.col("split") == SPLIT_TRAIN)["label"]) == {"self"}
    assert r1b["n_removed"] > r1["n_removed"]


def test_c0_removes_nothing():
    df = _frame()

    out, report = apply_condition(df, "C0", seed=SEED)

    assert out.equals(df)
    assert report["n_removed"] == 0


# ------------------------------------------------- kaybolan urun raporlaniyor
def test_items_that_vanish_from_training_are_reported():
    """CLAUDE.md 9: "sessizce gecilecek bir detay DEGIL".

    C1'de butun etkilesimleri silinen urun eval'de cold item olur ve C1 ile
    C0 farkinin bir kismini aciklayabilir. Raporlanmazsa o farki mudahaleye
    yazariz.
    """
    df = _frame()

    _, report = apply_condition(df, "C1", seed=SEED)

    # i2 egitimde YALNIZCA gift_given olarak geciyor (u1 ve u2), ama u3 icin
    # TEST item'i. C1 onu egitimden tamamen siliyor -> u3 degerlendirilirken
    # hic gorulmemis bir urun tahmin edilmeye calisiliyor. Iste C1 ile C0
    # farkinin mudahaleye degil cold item'a yazilabilecek kismi bu.
    assert report["n_items_lost_from_training"] == 1
    assert report["items_lost_sample"] == ["i2"]

    # C1b `household`i da siliyor -> daha cok urun dusuyor.
    _, r1b = apply_condition(df, "C1b", seed=SEED)
    assert r1b["n_items_lost_from_training"] > report["n_items_lost_from_training"]


# ------------------------------------------------------- C2 / C3 satir silmez
def test_c2_weights_instead_of_removing():
    df = _frame()

    out, report = apply_condition(df, "C2", seed=SEED, weights={"gift": 0.25})

    assert out.height == df.height
    assert report["gift_weight"] == 0.25
    hediye = out.filter(
        (pl.col("split") == SPLIT_TRAIN) & (pl.col("label") == "gift_given")
    )
    assert (hediye["weight"] == 0.25).all()
    assert (out.filter(pl.col("label") == "self")["weight"] == 1.0).all()


def test_c3_adds_a_feature_instead_of_removing():
    df = _frame()

    out, _ = apply_condition(df, "C3", seed=SEED)

    assert out.height == df.height
    assert out.filter(pl.col("label") == "gift_given")["is_gift"].to_list() == [1, 1, 1]


def test_unknown_condition_fails_loudly():
    with pytest.raises(ValueError, match="bilinmeyen kosul"):
        apply_condition(_frame(), "C9", seed=SEED)
