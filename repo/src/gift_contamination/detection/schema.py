"""Etiket semasinin Pydantic karsiligi (v2).

CLAUDE.md bolum 8, kural 3: LLM ciktisi regex ile parse EDILMEZ. vLLM guided
decoding `configs/annotation_schema.json` semasini uygular, bu modul de donen
nesneyi dogrular. Iki katman ayni sozlesmeyi iki farkli yerde zorlar.

IKI DOGRULUK KAYNAGI SORUNU. Enum degerleri hem JSON semasinda hem burada
yaziyor. Sessizce ayrisirlarsa guided decoding'in urettigi deger Pydantic'te
gecersiz olur ve bunu ancak annotation kosusunun ortasinda fark ederiz.
`tests/test_schema.py` bu esitligi kilitliyor - biri degisip digeri degismezse
test kirmizi yanar.

Sema v2 gerekcesi (2026-08-26): `grandchild` ayri bir sinif, cunku torun ayri
hanede yasar (hediye) ama kendi cocugu `household` olabilir. Toys_and_Games'te
adi gecen alicilarin %26.2'si torun. Ayrinti: docs/DECISIONS.md.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..config import REPO_ROOT, Config


class PurchaseType(StrEnum):
    SELF = "self"
    GIFT_GIVEN = "gift_given"
    HOUSEHOLD = "household"
    UNCLEAR = "unclear"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Recipient(StrEnum):
    CHILD = "child"
    GRANDCHILD = "grandchild"          # ayri hane -> hediye. Toys alicilarinin %26.2'si.
    PARTNER = "partner"                # es/sevgili/nisanli
    PARENT = "parent"
    SIBLING = "sibling"
    EXTENDED_FAMILY = "extended_family"  # yegen/kuzen/hala/amca
    FRIEND = "friend"                  # arkadas/komsu/is arkadasi
    OTHER = "other"
    UNKNOWN = "unknown"


class Occasion(StrEnum):
    BIRTHDAY = "birthday"
    CHRISTMAS = "christmas"
    WEDDING = "wedding"
    GRADUATION = "graduation"
    ANNIVERSARY = "anniversary"
    BABY_SHOWER = "baby_shower"
    VALENTINES_DAY = "valentines_day"
    MOTHERS_DAY = "mothers_day"
    FATHERS_DAY = "fathers_day"
    OTHER = "other"
    NONE = "none"
    UNKNOWN = "unknown"                # vesile metinden yalnizca %20-29 cikarilabiliyor


class GiftAnnotation(BaseModel):
    """Tek bir review icin detektor ciktisi."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    purchase_type: PurchaseType
    confidence: Confidence
    recipient: Recipient | None = None
    occasion: Occasion
    evidence_span: str = Field(
        description="Review metninden BIREBIR alinti; dogrulanmazsa kayit unclear'a duser."
    )


# ------------------------------------------------------------------ span kontrolu
def normalise_whitespace(text: str) -> str:
    """Ardisik bosluklari tekile indirger. Buyuk/kucuk harf KORUNUR.

    LLM bosluk ve satir sonunu normalize edebilir; bu meshru. Harf buyuklugunu
    degistirmesi ise "birebir alinti" iddiasini zayiflatir, o yuzden orada tolerans
    yok.
    """
    return " ".join(text.split())


def span_is_verbatim(span: str, review_text: str) -> bool:
    if not span.strip():
        return False
    return normalise_whitespace(span) in normalise_whitespace(review_text)


def downgrade_if_span_missing(
    annotation: GiftAnnotation, review_text: str
) -> tuple[GiftAnnotation, bool]:
    """Dogrulanamayan kanit span'i olan kaydi `unclear`'a dusurur.

    CLAUDE.md bolum 9: "evidence_span metinde yok -> unclear'a dusur, sayaci
    artir, orani raporla. Sessiz atlamak yok." Ikinci donus degeri o sayac icin.

    Zaten `unclear` olan kayit dusurulmez: prompt, kanit bulunamadiginda tam
    olarak bunu (unclear + bos span) uretmeyi soyluyor, yani bu bir basarisizlik
    degil dogru davranis. Onu da saymak hata oranini yapay olarak sisirirdi.
    """
    if annotation.purchase_type is PurchaseType.UNCLEAR:
        return annotation, False
    if span_is_verbatim(annotation.evidence_span, review_text):
        return annotation, False
    return (
        annotation.model_copy(
            update={
                "purchase_type": PurchaseType.UNCLEAR,
                "confidence": Confidence.LOW,
            }
        ),
        True,
    )


# ------------------------------------------------------------------ JSON semasi
def schema_json_path(cfg: Config | None = None) -> Path:
    """Guided decoding'e verilecek JSON semasinin yolu.

    cfg verilmezse varsayilan config okunur; yol burada tekrarlanmaz.
    Gerekce icin bkz. `prompting.prompt_path`.
    """
    raw = Path((cfg or Config.load()).get("detection.schema_path"))
    return raw if raw.is_absolute() else REPO_ROOT / raw


def load_json_schema(cfg: Config | None = None) -> dict:
    """vLLM `guided_json` bu dosyayi alir; Pydantic donen nesneyi dogrular."""
    with schema_json_path(cfg).open(encoding="utf-8") as fh:
        return json.load(fh)


def json_schema_enums(schema: dict) -> dict[str, list]:
    """Semadaki alan -> enum degerleri. Parite testi bunu kullaniyor."""
    return {
        field: spec["enum"]
        for field, spec in schema["properties"].items()
        if "enum" in spec
    }
