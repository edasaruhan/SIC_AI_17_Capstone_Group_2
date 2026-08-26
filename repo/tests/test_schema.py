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

    v3'ten beri hicbir alanda `null` yok - tum degerler string. Filtre yine de
    duruyor ki `null` sessizce geri eklenirse test bunu yakalasin degil, bariz
    olsun diye: geri eklenirse Pydantic tarafi eslesmez ve test kirmizi yanar.
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


def test_null_recipient_is_rejected():
    """v3: `null` kaldirildi, tek bir "bilinmiyor" yolu var.

    Enum'da hem `null` hem `"unknown"` vardi ve hangisinin ne zaman
    kullanilacagi hicbir yerde yazmiyordu; 40.000 satirda model ikisi arasinda
    rastgele gidip gelirdi ve analiz karisirdi.
    """
    with pytest.raises(ValidationError):
        GiftAnnotation(**{**VALID, "recipient": None})

    assert GiftAnnotation(**{**VALID, "recipient": "unknown"}).recipient is Recipient.UNKNOWN


def test_received_is_its_own_class():
    """v3: hediye ALAN, `self`'e katlanmiyor.

    `household` zaten "hediye degil ama kendi tercihi de degil" diye ayri
    tutuluyor; hediye alan da tam olarak bu durumda. Bir kez `self` yazilirsa
    bilgi geri gelmez ve C1/C2/C3 sonradan karar veremez.
    """
    ann = GiftAnnotation(**{**VALID, "purchase_type": "received"})

    assert ann.purchase_type is PurchaseType.RECEIVED
    assert PurchaseType.RECEIVED != PurchaseType.SELF


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
        purchase_type="unclear", confidence="low", recipient="unknown",
        occasion="unknown", evidence_span="",
    )

    out, downgraded = downgrade_if_span_missing(ann, "Great product!")

    assert downgraded is False
    assert out == ann
