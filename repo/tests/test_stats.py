"""Paylasilan `wilson_interval` ve onu saran iki cagri noktasi.

`wilson` iki modulde iki kez yazilmisti ve ikisi de commit edilen bir JSON'a
sayi yaziyordu (`prevalence.json`, `keyword_precision.json`). Iki kopyanin
sorunu birinde duzeltilen bir hatanin digerinde yasamasi. Buradaki testler
formulun TEK kaldigini ve iki cagri noktasinin kendi sunum kararini
koruduklarini kilitliyor (denetim 2026-09-20).
"""

from __future__ import annotations

import inspect
import math

import pytest

from gift_contamination.analysis.precision_check import wilson as precision_wilson
from gift_contamination.analysis.prevalence import wilson as prevalence_wilson
from gift_contamination.utils.stats import Z95, wilson_interval


def test_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(25, 100)

    assert lo < 0.25 < hi


def test_interval_is_undefined_without_a_sample():
    assert wilson_interval(0, 0) == (None, None)
    assert wilson_interval(3, -1) == (None, None)


@pytest.mark.parametrize("k,n", [(0, 20), (20, 20), (1, 3), (599, 100000)])
def test_core_never_leaves_the_unit_interval_by_more_than_rounding(k: int, n: int):
    """Wilson'un varlik sebebi bu: Wald araligi kucuk oranlarda negatife
    dusuyordu (`received` %0,6 seviyesinde)."""
    lo, hi = wilson_interval(k, n)

    assert -1e-12 <= lo <= hi <= 1 + 1e-12


def test_the_core_matches_a_hand_computed_interval():
    k, n, z = 25, 100, Z95
    p = k / n
    d = 1 + z * z / n
    merkez = (p + z * z / (2 * n)) / d
    yari = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d

    assert wilson_interval(k, n) == pytest.approx((merkez - yari, merkez + yari))


def test_both_callers_go_through_the_one_formula():
    """Kopyalardan biri geri gelirse burasi duser."""
    lo, hi = wilson_interval(35, 60, z=1.96)
    assert precision_wilson(35, 60) == (max(0.0, lo), min(1.0, hi))

    lo, hi = wilson_interval(25, 100)
    assert prevalence_wilson(25, 100) == (round(100 * lo, 2), round(100 * hi, 2))


def test_the_two_z_defaults_stay_where_the_published_numbers_were_made():
    """Iki cagri noktasi ayni z'yi KULLANMIYOR (1,959963985 ve 1,96) ve
    `prevalence.json` ile `keyword_precision.json` o degerlerle uretildi.
    Birlestirme sirasinda sessizce esitlemek yayinlanmis bir araligi
    degistirirdi - sonuc goruldukten sonra tanim degismez (CLAUDE.md 4)."""
    assert inspect.signature(prevalence_wilson).parameters["z"].default == 1.959963985
    assert inspect.signature(precision_wilson).parameters["z"].default == 1.96


def test_presentation_differs_on_purpose():
    """Biri YUZDE ve yuvarlanmis, digeri ORAN ve kirpilmis."""
    assert prevalence_wilson(25, 100) == (17.55, 34.3)
    assert precision_wilson(0, 20)[0] == 0.0
    assert precision_wilson(20, 20)[1] == 1.0
    assert prevalence_wilson(0, 0) == (None, None)
    assert all(math.isnan(x) for x in precision_wilson(0, 0))
