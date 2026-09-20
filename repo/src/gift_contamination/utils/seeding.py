"""Surecler arasi KARARLI seed turetimi.

Deneyin yeniden uretilebilirlik iddiasi tek bir `seed: 42` degerine dayaniyor,
ama her rastgele secim kendi seed'ini istiyor (kosul, metrik, model, kova...).
Bu turetim `hash()` ile YAPILAMAZ: Python'un string hash'i PYTHONHASHSEED ile
sureclere gore degisiyor, yani ayni komut iki kosuda baska satirlari secerdi ve
fark ancak sayilara bakilarak sezilirdi. `zlib.crc32` sabit.

Ayni desen `data/sampling.py` icinde de var (`_stratum_seed`); buradaki surum
kosul kodu, istatistik ve pazarlama metrikleri tarafindan paylasiliyor. Uc
modulde uc kopya yerine tek yer (denetim 2026-09-20).
"""

from __future__ import annotations

import zlib

MODULUS = 2**31 - 1


def seed_for(base: int, key: str) -> int:
    """`base` ile `key`den tureyen kararli seed.

    DEGERLER YAYINLANMIS SONUCLARA BAGLI: `experiment_stats.json`,
    `gate2.json`, `marketing_metrics.json` ve 16 `condition_*.json` bu
    fonksiyonun ciktisiyla uretildi. Formul degisirse o sayilar da degisir -
    `tests/test_stats_seeding.py` birkac degeri caktigi icin sessizce olmaz.
    """
    return (base + zlib.crc32(str(key).encode("utf-8"))) % MODULUS
