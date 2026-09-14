"""KONTAMINASYON TANIMLARI - TEK dogruluk kaynagi (karar 2026-09-14).

Yayginlik (`analysis.prevalence`), dogrulamanin ikili eksenleri
(`analysis.validation`), damitma kapisi ve deney kosullari (`recsys.conditions`)
AYNI kumeyi kullanmali: ayri yazilmis bir tanim sessizce ayrisir ve "C1b'nin
etiket kalitesi" ile "C1b kosulunun cikardigi satirlar" farkli seyler olur.

  dar   -> C1  (birincil): literaturdeki hediye
  genis -> C1b (saglamlik): alici urunu kendisi SECMEDI

NEDEN AYRI MODUL. Deney kosucusu `.venv-recbole` altinda kosuyor ve orada
pydantic yok; `detection.schema` pydantic istiyor. Tanim bagimliliksiz durur,
`schema` onu yeniden disa aktarir ve degerlerin `PurchaseType`'ta oldugunu
import aninda dogrular.
"""

from __future__ import annotations

CONTAMINATION: dict[str, tuple[str, ...]] = {
    "narrow": ("gift_given",),
    "broad": ("gift_given", "household", "received"),
}
# Tanimin deney kosulundaki adi.
CONDITION_OF = {"narrow": "C1", "broad": "C1b"}
