"""Etiket semasi testleri.

Iki isi var. Birincisi CLAUDE.md bolum 10'un istedigi "gecerli/gecersiz JSON dogru
ayriliyor mu" kontrolu. Ikincisi ve daha kritigi PARITE: enum degerleri hem
`configs/annotation_schema.json` icinde (guided decoding onu kullaniyor) hem
`detection/schema.py` icinde (Pydantic onu kullaniyor) yaziyor. Ayrisirlarsa
model gecerli bir deger uretir, Pydantic reddeder ve bunu ancak 40-60K'lik
kosunun ortasinda fark ederiz.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gift_contamination.detection.schema import (
    Confidence,
    GiftAnnotation,
    Occasion,
    PurchaseType,
    Recipient,
    downgrade_if_span_missing,
    json_schema_enums,
    load_json_schema,
    normalise_whitespace,
    span_is_verbatim,
)

VALID = {
    "purchase_type": "gift_given",
    "confidence": "high",
    "recipient": "grandchild",
    "occasion": "birthday",
    "evidence_span": "bought this for my grandson",
}


# ---------------------------------------------------------------- parite
@pytest.mark.parametrize(
    ("field", "enum_cls"),
    [
        ("purchase_type", PurchaseType),
        ("confidence", Confidence),
        ("recipient", Recipient),
        ("occasion", Occasion),
    ],
)
def test_enum_parity_with_json_schema(field: str, enum_cls):
    """Pydantic enum'lari JSON semasindakiyle BIREBIR ayni olmali.

    `recipient` semada `null` da kabul ediyor; o Pydantic tarafinda `| None`
    olarak ifade edildigi icin karsilastirmadan cikariliyor.
    """
    from_json = [v for v in json_schema_enums(load_json_schema())[field] if v is not None]

    assert from_json == [e.value for e in enum_cls], (
        f"'{field}' JSON semasi ile Pydantic arasinda ayristi. "
        "Ikisini birden guncelleyin (docs/DECISIONS.md'ye de yazin)."
    )


def test_default_schema_path_comes_from_the_config_not_a_literal():
    """Bkz. test_prompting.py'deki esi - ayni surukleme hatasi sinifi."""
    from gift_contamination.config import Config
    from gift_contamination.detection.schema import schema_json_path

    assert schema_json_path() == schema_json_path(Config.load())


def test_schema_file_requires_evidence_span():
    """evidence_span zorunlu kalmali: halusinasyonu kisan tek mekanizma."""
    schema = load_json_schema()

    assert "evidence_span" in schema["required"]
    assert schema["additionalProperties"] is False


# ------------------------------------------------------------- dogrulama
def test_valid_annotation_parses():
    ann = GiftAnnotation(**VALID)

    assert ann.purchase_type is PurchaseType.GIFT_GIVEN
    assert ann.recipient is Recipient.GRANDCHILD


def test_null_recipient_is_allowed():
    ann = GiftAnnotation(**{**VALID, "recipient": None})

    assert ann.recipient is None


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("purchase_type", "gift"),          # sema disi deger
        ("confidence", "very_high"),
        ("recipient", "colleague"),         # v1'de vardi, v2'de kaldirildi
        ("occasion", "halloween"),
    ],
)
def test_invalid_enum_value_is_rejected(field: str, bad: str):
    with pytest.raises(ValidationError):
        GiftAnnotation(**{**VALID, field: bad})


def test_extra_field_is_rejected():
    """Model uydurma alan eklerse sessizce yutulmamali."""
    with pytest.raises(ValidationError):
        GiftAnnotation(**{**VALID, "reasoning": "because"})


def test_missing_required_field_is_rejected():
    incomplete = {k: v for k, v in VALID.items() if k != "occasion"}

    with pytest.raises(ValidationError):
        GiftAnnotation(**incomplete)


# ------------------------------------------------------------- span kontrolu
def test_whitespace_is_normalised_but_case_is_not():
    assert normalise_whitespace("a  b\n c") == "a b c"
    assert span_is_verbatim("my  grandson", "for my grandson today")
    assert not span_is_verbatim("My Grandson", "for my grandson today")


def test_verbatim_span_is_not_downgraded():
    ann = GiftAnnotation(**VALID)

    out, downgraded = downgrade_if_span_missing(ann, "I bought this for my grandson.")

    assert downgraded is False
    assert out.purchase_type is PurchaseType.GIFT_GIVEN


def test_hallucinated_span_is_downgraded():
    ann = GiftAnnotation(**VALID)

    out, downgraded = downgrade_if_span_missing(ann, "The colour is nice.")

    assert downgraded is True
    assert out.purchase_type is PurchaseType.UNCLEAR
    assert out.confidence is Confidence.LOW
    # Kanit korunur: hata analizinde modelin ne uydurdugunu gormek gerekiyor.
    assert out.evidence_span == VALID["evidence_span"]


def test_empty_span_on_a_confident_label_is_downgraded():
    ann = GiftAnnotation(**{**VALID, "evidence_span": ""})

    out, downgraded = downgrade_if_span_missing(ann, "anything at all")

    assert downgraded is True
    assert out.purchase_type is PurchaseType.UNCLEAR


def test_already_unclear_with_empty_span_is_not_counted_as_a_failure():
    """Prompt kanit bulunamayinca tam olarak bunu uretmeyi soyluyor.

    Dogru davranisi hata sayacina yazmak, span basarisizlik oranini yapay olarak
    sisirir ve raporlanan sayiyi anlamsiz kilar.
    """
    ann = GiftAnnotation(
        purchase_type="unclear", confidence="low", recipient=None,
        occasion="unknown", evidence_span="",
    )

    out, downgraded = downgrade_if_span_missing(ann, "Great product!")

    assert downgraded is False
    assert out == ann
