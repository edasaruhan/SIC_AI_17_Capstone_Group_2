"""Sozcuksel vekilin davranisini kilitleyen testler.

Bu testler dogruluk iddiasi degil, REGRESYON kalkani. Desen dosyasi
(`configs/gift_keywords.yaml`) elle duzenlenebilir bir artefakt; birisi bir desen
eklerken projenin bagli oldugu ayrimlari bozarsa burada patlamali.

Kritik ayrimlar `prompts/gift_detection_v1.md` icinde tanimli:
  - "would make a great gift" SPEKULASYONDUR, hediye degil
  - hediye ALMAK vermek degildir
  - "bought it and my daughter loved it" hediye sozcugu icermez ama HEDIYEDIR
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.analysis.deep_eda import CHARS_PER_TOKEN
from gift_contamination.analysis.eda import gini
from gift_contamination.analysis.keyword_scan import load_patterns
from gift_contamination.analysis.precision_check import wilson
from gift_contamination.config import Config


@pytest.fixture(scope="module")
def pat():
    return load_patterns(Config.load())


def _hits(pat, text: str) -> set[str]:
    """Verilen metnin hangi desen ailelerine takildigi."""
    df = pl.DataFrame({"t": [text]})
    return {
        fam
        for fam, rx in pat.families.items()
        if df.select(pl.col("t").str.contains(rx)).item()
    }


# ------------------------------------------------------- aile ayrimlari
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Acik hediye kaniti
        ("I bought this for my daughter and she loves it", "gift_evidence"),
        ("Got this for my nephew for his birthday", "gift_evidence"),
        ("I gave it to my mom and she was thrilled", "gift_evidence"),
        ("Perfect stocking stuffer, bought as a gift", "gift_evidence"),
        # Hediye sozcugu HIC gecmeyen ama net hediye olan vaka.
        # PROJECT_SPEC bolum 1.3 bunu anahtar kelimenin yanlis negatifi olarak
        # ozellikle ornekliyor; desenlerin bunu yakalamasi gerekiyor.
        ("Bought it and my daughter loved it immediately", "gift_evidence"),
        # Spekulasyon: hediye sozcugu var, alim kendine
        ("This would make a great gift for anyone", "gift_speculative"),
        ("Great gift idea if you are shopping around", "gift_speculative"),
        # Alan, veren degil
        ("I received this as a gift from a coworker", "gift_received"),
        ("This was a gift from my sister", "gift_received"),
    ],
)
def test_family_assignment(pat, text: str, expected: str):
    assert expected in _hits(pat, text), f"{expected!r} yakalanmadi: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "My friend recommended this and I love it",  # tavsiye, hediye degil
        "The colour is nice and it arrived quickly",  # tamamen notr
        "I use this every single day at work",
    ],
)
def test_self_purchases_do_not_match_evidence(pat, text: str):
    assert "gift_evidence" not in _hits(pat, text), f"yanlis pozitif: {text!r}"


def test_speculative_is_not_counted_as_evidence(pat):
    """Projenin tum gerekcesi bu ayrimda; ikisi ayni anda tetiklenmemeli."""
    hits = _hits(pat, "This would make a great gift for the holidays")

    assert "gift_speculative" in hits
    assert "gift_evidence" not in hits


def test_proxy_gives_precedence_to_received(pat):
    """kw_gift_proxy = evidence AND NOT received. Hediye almak vermek degildir."""
    text = "I received this as a gift and my husband loves it too"
    df = pl.DataFrame({"t": [text]}).select(
        **{
            f"kw_{fam}": pl.col("t").str.contains(rx)
            for fam, rx in pat.families.items()
        }
    ).with_columns(
        (pl.col("kw_gift_evidence") & ~pl.col("kw_gift_received")).alias("proxy")
    )

    assert df["kw_gift_received"].item() is True
    assert df["proxy"].item() is False, "alinan hediye vekil sayilmamali"


# --------------------------------------------------- alici / vesile cikarimi
@pytest.mark.parametrize(
    ("text", "recipient"),
    [
        ("bought this for my daughter", "daughter"),
        ("a present for my grandson", "grandson"),
        ("I got it for my coworker", "co-?worker"),
    ],
)
def test_recipient_extraction(pat, text: str, recipient: str):
    got = (
        pl.DataFrame({"t": [text]})
        .select(pl.col("t").str.extract(pat.recipient, 1).str.to_lowercase())
        .item()
    )

    assert got is not None
    # 'co-?worker' gibi desenler icin gercek eslesme 'coworker' olabilir
    assert got.replace("-", "") in recipient.replace("-?", "").replace("-", "")


def test_recipient_is_null_when_absent(pat):
    got = (
        pl.DataFrame({"t": ["the product arrived quickly and works well"]})
        .select(pl.col("t").str.extract(pat.recipient, 1))
        .item()
    )

    assert got is None


@pytest.mark.parametrize(
    ("text", "occasion"),
    [
        ("for her birthday party", "birthday"),
        ("a christmas present", "christmas"),
        ("bought for the wedding", "wedding"),
    ],
)
def test_occasion_extraction(pat, text: str, occasion: str):
    got = (
        pl.DataFrame({"t": [text]})
        .select(pl.col("t").str.extract(pat.occasion, 1).str.to_lowercase())
        .item()
    )

    assert got == occasion


# ------------------------------------------------------- kanit konumu
def test_evidence_position_is_relative_and_ordered(pat):
    """kw_pos_rel metnin basinda kucuk, sonunda buyuk olmali.

    Teslimin bolum 4.8'deki "kesme kaybi" sayisi bu kolona dayaniyor; yon
    yanlissa o bolumun tum sonucu ters doner.
    """
    early = "Bought this for my son. " + "Filler text here. " * 40
    late = "Filler text here. " * 40 + "Bought this for my son."
    rx = pat.families["gift_evidence"]

    pos = (
        pl.DataFrame({"t": [early, late]})
        .select(
            (pl.col("t").str.find(rx) / pl.col("t").str.len_chars()).alias("rel")
        )["rel"]
        .to_list()
    )

    assert pos[0] < 0.05, "bastaki kanit basta cikmali"
    assert pos[1] > 0.90, "sondaki kanit sonda cikmali"


def test_truncation_arithmetic_matches_document():
    """Bolum 4.8'de raporlanan esik: 512 token ~ 2048 karakter."""
    assert 512 * CHARS_PER_TOKEN == 2048
    assert 1024 * CHARS_PER_TOKEN == 4096


# ------------------------------------------------------- istatistik yardimcilari
def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson(35, 60)  # teslimin T4 satiri

    assert lo < 35 / 60 < hi
    assert (round(lo, 3), round(hi, 3)) == (0.457, 0.699)


def test_wilson_handles_degenerate_counts():
    assert wilson(0, 20)[0] == 0.0
    assert wilson(20, 20)[1] == 1.0


def test_gini_bounds():
    import numpy as np

    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-9)
    concentrated = np.array([0.0] * 99 + [100.0])
    assert gini(concentrated) > 0.95
