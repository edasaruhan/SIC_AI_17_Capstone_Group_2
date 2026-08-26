"""Veri butunlugu testleri (CLAUDE.md bolum 10).

Urun kodu testi degil; on islemenin deneyi gecersiz kilacak sekilde yanlis
davranmadigini dogruluyoruz. En kritik olani `test_k_core_cascade`: tek gecislik bir
k-core implementasyonu "5-core" adi altinda 5-core OLMAYAN bir veri uretir ve bu
sessizce gecerse tum recsys sonuclari kirlenir.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from gift_contamination.config import Config
from gift_contamination.data.preprocess import (
    build_clean,
    build_raw_parquet,
    detect_timestamp_unit,
    iterative_k_core,
)

from conftest import EXPECTED_STAGES, TEST_K


# --------------------------------------------------------------------- k-core
def _cascade_frame() -> pl.DataFrame:
    """Tek gecisin yetmedigi kucuk grafik.

    u3-i3 atildiginda u3'un tek etkilesimi kalir; ikinci gecis onu da atmali.
    """
    return pl.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4, 5],
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "parent_asin": ["i1", "i2", "i1", "i2", "i1", "i3"],
        }
    )


def test_k_core_cascade():
    df, iterations = iterative_k_core(_cascade_frame(), TEST_K, "user_id", "parent_asin")

    assert iterations > 1, "kaskad tek gecisde cozulemez; implementasyon iteratif degil"
    assert df.height == 4
    assert set(df["user_id"].unique()) == {"u1", "u2"}
    assert set(df["parent_asin"].unique()) == {"i1", "i2"}


def test_k_core_invariant_holds():
    """Cikti gercekten k-core mu: her kullanici >= k VE her item >= k."""
    df, _ = iterative_k_core(_cascade_frame(), TEST_K, "user_id", "parent_asin")

    assert df.group_by("user_id").len()["len"].min() >= TEST_K
    assert df.group_by("parent_asin").len()["len"].min() >= TEST_K


def test_k_core_empty_result_is_not_an_error():
    """Yogunlugu dusuk kategoride k-core her seyi eleyebilir; patlamamali.

    All_Beauty'de gercekten boyle oluyor (kullanicilarin %0.08'i 5+ etkilesimli).
    """
    sparse = pl.DataFrame(
        {"row_id": [0, 1], "user_id": ["u1", "u2"], "parent_asin": ["i1", "i2"]}
    )
    df, iterations = iterative_k_core(sparse, 5, "user_id", "parent_asin")

    assert df.height == 0
    assert iterations >= 1


# ------------------------------------------------------------------ huni akisi
def test_funnel_stage_counts(cfg: Config):
    """Her filtre tam olarak beklenen satirlari eliyor mu."""
    import json

    build_clean(cfg, "pilot")
    funnel = json.loads(
        (cfg.path("results") / "preprocess_funnel_Test_Cat.json").read_text(
            encoding="utf-8"
        )
    )

    for stage, expected in EXPECTED_STAGES.items():
        assert funnel[stage]["rows"] == expected, f"{stage} beklenenden farkli"


def test_dedup_keeps_earliest_timestamp(cfg: Config):
    """Ayni user+item iki kez yorumlanmissa en erken timestamp tutulur (CLAUDE.md bolum 9)."""
    clean_path, _ = build_clean(cfg, "pilot")
    clean = pl.read_parquet(clean_path)

    row = clean.filter((pl.col("user_id") == "u1") & (pl.col("parent_asin") == "i1"))
    assert row.height == 1, "dedup calismadi"
    assert row["timestamp"].item() == 1600000000000, "gec olan kayit tutulmus"


def test_short_and_unverified_rows_removed(cfg: Config):
    clean_path, _ = build_clean(cfg, "pilot")
    clean = pl.read_parquet(clean_path)

    assert "u4" not in clean["user_id"].to_list(), "verified_purchase=false satir kalmis"
    assert "u5" not in clean["user_id"].to_list(), "5 kelimeden kisa satir kalmis"
    assert clean["n_words"].min() >= 5


def test_sequences_are_chronological(cfg: Config):
    """seq_pos kullanici icinde zaman sirasini takip etmeli."""
    clean_path, _ = build_clean(cfg, "pilot")
    clean = pl.read_parquet(clean_path)

    for (_user,), group in clean.group_by(["user_id"]):
        ordered = group.sort("seq_pos")
        assert ordered["seq_pos"].to_list() == list(range(ordered.height))
        assert ordered["timestamp"].is_sorted(), "sekans kronolojik degil"


def test_kcore_output_is_really_k_core(cfg: Config):
    """Yazilan kcore parquet'i dosya seviyesinde de k-core mu."""
    _, kcore_path = build_clean(cfg, "pilot")
    kcore = pl.read_parquet(kcore_path)

    assert kcore.height == EXPECTED_STAGES[f"05_k_core_{TEST_K}"]
    assert kcore.group_by("user_id").len()["len"].min() >= TEST_K
    assert kcore.group_by("parent_asin").len()["len"].min() >= TEST_K


def test_clean_is_not_k_cored(cfg: Config):
    """clean korpusu k-core UYGULANMAMIS olmali; annotation onun uzerinde yurur."""
    clean_path, kcore_path = build_clean(cfg, "pilot")

    assert pl.read_parquet(clean_path).height > pl.read_parquet(kcore_path).height


# ------------------------------------------------------------------- timestamp
@pytest.mark.parametrize(
    ("configured", "expected"), [("s", "s"), ("ms", "ms"), ("auto", "ms")]
)
def test_timestamp_unit_detection(cfg: Config, configured: str, expected: str):
    """Fixture ms cinsinden; auto modda ms tespit edilmeli, acik ayar ezmeli."""
    raw = build_raw_parquet(cfg, "pilot")

    assert detect_timestamp_unit(raw, configured) == expected


def test_timestamps_land_in_expected_decade(cfg: Config):
    """ms/s karisirsa tarihler 1970'e ya da uzak gelecege dusar; erken yakala."""
    clean_path, _ = build_clean(cfg, "pilot")
    clean = pl.read_parquet(clean_path)

    assert clean["ts"].dt.year().min() >= 2020
    assert clean["ts"].dt.year().max() <= 2024


# ---------------------------------------------------------------- idempotency
def test_rerun_is_idempotent(cfg: Config):
    """Cikti varsa yeniden hesaplanmamali (CLAUDE.md bolum 7, birinci yari)."""
    clean_path, _ = build_clean(cfg, "pilot")
    first = pl.read_parquet(clean_path)

    clean_path_again, _ = build_clean(cfg, "pilot")
    second = pl.read_parquet(clean_path_again)

    assert clean_path == clean_path_again
    assert first.equals(second)


def test_recompute_is_deterministic(cfg: Config):
    """`--force` ile yeniden hesaplama AYNI sonucu vermeli.

    Idempotency iddiasinin asil onemli yarisi. Ustteki test `should_skip`
    yoluna girdigi icin yeniden hesaplamayi hic sinamiyor - sort kararliligi
    ya da seed yonetimi bozulsa oradan gecerdi (2026-08-27 denetimi).
    """
    c1, k1 = build_clean(cfg, "pilot", force=True)
    clean_a, kcore_a = pl.read_parquet(c1), pl.read_parquet(k1)

    c2, k2 = build_clean(cfg, "pilot", force=True)

    assert clean_a.equals(pl.read_parquet(c2)), "clean yeniden hesaplamada farkli"
    assert kcore_a.equals(pl.read_parquet(k2)), "kcore yeniden hesaplamada farkli"


# --------------------------------------------------------------------- gizlilik
def test_funnel_json_carries_no_absolute_path(cfg: Config):
    """Huni JSON'i commit ediliyor; mutlak yol kullanici adini sizdirir.

    2026-08-27 denetimi: dokuz commit edilmis dosyada
    `C:\\Users\\<ad>\\...` yaziyordu. Duzeltme `utils.io.relative_to_repo`.
    """
    import json

    build_clean(cfg, "pilot", force=True)
    raw = (cfg.path("results") / "preprocess_funnel_Test_Cat.json").read_text(
        encoding="utf-8"
    )
    funnel = json.loads(raw)

    # Fixture repo kokunun DISINDA (pytest tmpdir) oldugu icin yardimci dogru
    # sekilde cıplak dosya adina dusuyor. Gercek kosuda `data/interim/...` olur.
    assert "Users" not in raw, "kullanici adi sizmis"
    for value in [funnel["meta"]["source"], *funnel["meta"]["outputs"].values()]:
        assert not Path(value).is_absolute(), f"mutlak yol yazilmis: {value}"
        assert ":" not in value, f"surucu harfi yazilmis: {value}"
