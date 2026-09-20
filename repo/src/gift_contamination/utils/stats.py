"""Paylasilan istatistik cekirdegi.

Neden ayri bir modul: `wilson` iki yerde iki kez yazilmisti
(`analysis/prevalence.py`, `analysis/precision_check.py`). Ayni formulun iki
kopyasi, birinde duzeltilen bir hatanin digerinde yasamaya devam etmesi
demektir - ve ikisi de commit edilen bir JSON'a sayi yaziyor (denetim
2026-09-20).

DIKKAT - iki cagri noktasinin `z` VARSAYILANI AYNI DEGIL: prevalence
1,959963985 kullaniyor, precision_check 1,96. Fark dorduncu basamakta ve
`prevalence.json` ile `keyword_precision.json` o degerlerle uretildi. Sonuc
goruldukten sonra tanim degistirilmiyor, bu yuzden burada TEK bir varsayilan
dayatilmiyor: ortak olan yalnizca formul.
"""

from __future__ import annotations

import math

Z95 = 1.959963985


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | tuple[None, None]:
    """Wilson skor araligi, ORAN olarak (0-1); orneklem yoksa (None, None).

    Normal (Wald) yaklasim p +- z*sqrt(p(1-p)/n) kucuk oranlarda birim
    araligin disina tasiyor - `received` %0,6 seviyesinde ve orada alt sinir
    negatife dusuyordu. Wilson bu kusuru tasimiyor.

    Kirpma BURADA YAPILMIYOR: cagiran taraf ya yuzdeye cevirip yuvarliyor
    (prevalence) ya da [0, 1]'e kirpiyor (precision_check). Sunum kararini
    cekirdege gommek, iki cagri noktasindan birinin ciktisini degistirirdi.
    """
    if n <= 0:
        return (None, None)
    p = k / n
    payda = 1 + z * z / n
    merkez = (p + z * z / (2 * n)) / payda
    yari = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / payda
    return (merkez - yari, merkez + yari)
