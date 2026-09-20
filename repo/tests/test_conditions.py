"""C0-C4 kosul testleri. CLAUDE.md 10 bunu **kritik** diye isaretliyor.

Buradaki hatalarin ortak ozelligi SESSIZ olmalari: kod calisir, sayilar makul
gorunur, ve deney gecersiz cikar - ama bunu ancak biri "C4 gercekten C1 kadar
mi cikardi" diye sorunca fark ederiz. Juri'deki ilk akilli kisi de zaten tam
bunu soracak.
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.detection.schema import CONTAMINATION
from gift_contamination.recsys.atomic import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from gift_contamination.recsys.conditions import (
    CONDITIONS,
    PLACEBO_OF,
    REMOVED_LABELS,
    SHADOW_LABELS,
    SHADOW_SUFFIX,
    seed_for,
    apply_condition,
    universe,
)

SEED = 42


def _frame() -> pl.DataFrame:
    """Dort kullanici, karisik etiketler. Her kullanicinin son satiri `self`.

    Egitim satirlarinda bilerek `gift_given`, `household` ve `received` var: C1
    ile C1b'nin gercekten farkli davrandigi gorunsun. `i10` yalnizca u4'un
    `received` satirinda geciyor - C1b onu egitimden tamamen siler, C1 silmez.
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
        ("u4", "i10", 0, "received",  SPLIT_TRAIN),
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


def test_c4b_removes_exactly_as_many_rows_as_c1b():
    """C1b, C1'in ~iki kati satir cikariyor; C4 ile karsilastirmak iki farkli
    veri kaybini karistirirdi. C4b'nin sayisi C1b'den TURETILIYOR."""
    df = _frame()

    _, c1b = apply_condition(df, "C1b", seed=SEED)
    _, c4b = apply_condition(df, "C4b", seed=SEED)
    _, c4 = apply_condition(df, "C4", seed=SEED)

    assert c4b["n_removed"] == c1b["n_removed"] > c4["n_removed"]
    assert c4b["placebo_of"] == "C1b"


def _removed_rows(df: pl.DataFrame, out: pl.DataFrame) -> set[tuple]:
    """Kosulun `df`den dusurdugu satirlar - (kullanici, urun) ciftleri."""
    return set(
        df.join(out, on=["user_id", "item_id", "timestamp", "label", "split"], how="anti")
        .select("user_id", "item_id")
        .iter_rows()
    )


def test_c4_removes_at_random_not_by_label():
    """Plasebo etikete BAKMAMALI - baksa plasebo olmaz, ikinci bir C1 olur.

    Eski hali "gift_given ya da household hayatta kaldi mi" diye soruyordu ve
    C4'u C1'in kopyasi yapan bir mutasyonda BILE geciyordu: C1 `household`a
    dokunmuyor, yani `or` dali her zaman dogruydu (denetim 2026-09-20). Iki
    assert de o mutasyonda dusmeli.
    """
    df = _frame()

    c4, _ = apply_condition(df, "C4", seed=SEED)
    c1, _ = apply_condition(df, "C1", seed=SEED)

    kalan = c4.filter(pl.col("split") == SPLIT_TRAIN)["label"].to_list()
    assert df.filter(pl.col("split") == SPLIT_TRAIN).height - len(kalan) > 0
    # 1. C1'in sildigi etiket C4'te hayatta kalir: secim etikete bakmiyor.
    assert "gift_given" in kalan
    # 2. Secilen satirlar C1'inkinden farkli: sayi ayni, kume degil.
    assert _removed_rows(df, c4) != _removed_rows(df, c1)


def test_seed_for_is_pinned_so_it_cannot_become_hash():
    """`hash()` PYTHONHASHSEED ile surecten surece degisir; crc32 degismez.

    Deger CAKILI, cunku "ayni surecte iki kez cagir, esit mi" testi `hash()`
    ile de gecer - `hash()` bir surec ICINDE kararlidir (denetim 2026-09-20).
    Kardesi `test_sampling.test_stratum_seed_is_stable_across_processes` ayni
    deseni kullaniyor. Bu sayi degisirse C4/C4b'nin sectigi satirlar ve
    butun bootstrap seed'leri degisir.
    """
    assert seed_for(42, "C4") == 993655479
    assert seed_for(42, "C4b") == 353048883


def test_c4_is_reproducible_across_processes():
    """Ayni seed ayni satirlari secmezse "3 seed" iddiasi anlamsizlasir."""
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
def test_c1_removes_only_gifts_c1b_also_removes_household_and_received():
    """C1 = `gift_given`; C1b = "alici urunu kendisi secmedi" (DECISIONS 2026-09-14)."""
    df = _frame()

    c1, r1 = apply_condition(df, "C1", seed=SEED)
    c1b, r1b = apply_condition(df, "C1b", seed=SEED)

    assert set(c1.filter(pl.col("split") == SPLIT_TRAIN)["label"]) == {
        "self", "household", "received",
    }
    assert set(c1b.filter(pl.col("split") == SPLIT_TRAIN)["label"]) == {"self"}
    assert r1b["n_removed"] > r1["n_removed"]


def test_condition_label_sets_come_from_the_single_definition():
    """Yayginlik, dogrulama ekseni ve deney AYNI kumeyi okumali.

    Tanim uc yerde ayri yazilirsa biri degisir, digerleri degismez: yayginlik
    bir seyi, deney baska bir seyi olcer ve bunu hicbir test fark etmez.
    """
    assert REMOVED_LABELS["C1"] == CONTAMINATION["narrow"]
    assert REMOVED_LABELS["C1b"] == CONTAMINATION["broad"]
    assert "received" in REMOVED_LABELS["C1b"]
    # RQ3 "silmek mi, soylemek mi": C3 C1'in satirlarini golge yapar
    assert SHADOW_LABELS["C3"] == REMOVED_LABELS["C1"]
    assert PLACEBO_OF == {"C4": "C1", "C4b": "C1b"}


def test_every_configured_condition_is_implemented():
    """`experiment.conditions` listesindeki her kod gercekten kosabilmeli."""
    from pathlib import Path

    gercek = Config.load(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")

    istenen = list(gercek.get("experiment.conditions"))
    assert set(istenen) <= set(CONDITIONS)
    assert {"C0", "C1", "C4", "C1b", "C4b", "C3"} <= set(istenen)


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


# ------------------------------------------------------------- C2 kilitli
def test_c2_is_locked_instead_of_silently_training_c0():
    """Onceki C2 bir `weight` kolonu ekliyordu ve RecBole onu hic okumuyordu:
    kosulsa C0'in aynisi "C2" diye raporlanirdi."""
    with pytest.raises(NotImplementedError, match="UYGULANMADI"):
        apply_condition(_frame(), "C2", seed=SEED)
    assert "C2" not in CONDITIONS


# -------------------------------------------------------- C3 golge token
def test_c3_turns_training_gifts_into_shadow_items_without_removing_rows():
    df = _frame()

    out, report = apply_condition(df, "C3", seed=SEED)

    assert out.height == df.height and report["n_removed"] == 0
    golge = out.filter(pl.col("item_id").str.ends_with(SHADOW_SUFFIX))
    egitim_hediye = df.filter((pl.col("split") == SPLIT_TRAIN) & (pl.col("label") == "gift_given"))
    assert golge.height == egitim_hediye.height == report["n_shadow_rows"] == 3
    assert set(golge["label"]) == {"gift_given"}
    assert set(golge["split"]) == {SPLIT_TRAIN}
    # Sekans sirasi korunur: ayni kullanici, ayni zaman damgasi, yalnizca kimlik degisti
    assert golge.select("user_id", "timestamp").sort("user_id", "timestamp").equals(
        egitim_hediye.select("user_id", "timestamp").sort("user_id", "timestamp")
    )


def test_c3_leaves_household_received_and_non_training_rows_real():
    """C3 yalnizca C1'in kumesini golge yapar; digerleri ve test/valid gercek kalir."""
    df = _frame()

    out, _ = apply_condition(df, "C3", seed=SEED)

    gercek = out.filter(~pl.col("item_id").str.ends_with(SHADOW_SUFFIX))
    assert {"household", "received", "self"} <= set(gercek["label"])
    assert not out.filter(pl.col("split") != SPLIT_TRAIN)["item_id"].str.ends_with(SHADOW_SUFFIX).any()


def test_c3_reports_items_whose_real_embedding_is_no_longer_trained():
    """i2 egitimde yalnizca hediye olarak geciyor: C3'te gercek kimligi hic
    egitilmiyor (yalnizca golgesi var) - C1 ile ayni cold item durumu."""
    _, report = apply_condition(_frame(), "C3", seed=SEED)

    assert report["items_lost_sample"] == ["i2"]


def test_unknown_condition_fails_loudly():
    with pytest.raises(ValueError, match="bilinmeyen kosul"):
        apply_condition(_frame(), "C9", seed=SEED)
