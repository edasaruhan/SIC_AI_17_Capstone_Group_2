"""Prompt yukleme ve render testleri.

Bu modulun tek isi sessiz basarisizligi engellemek. 40-60K'lik bir kosuda bos bir
SYSTEM prompt'u ya da doldurulmamis bir `{text}` yer tutucusu ile devam etmek,
hatanin ancak sonuclara bakildiginda anlasilmasi demektir - ve o noktada Kaggle
kotasi harcanmis olur.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gift_contamination.detection.prompting import (
    REQUIRED_FIELDS,
    load_prompt,
    prompt_path,
    render,
)

MINIMAL = """# test_prompt

**Versiyon:** v9 · **Model:** test

## SYSTEM

```
You classify reviews.
```

## USER

```
Product: {product_title}
Category: {category}
Title: {title}
Review: {text}
```
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "prompt.md"
    p.write_text(body, encoding="utf-8")
    return p


# ------------------------------------------------------------ gercek dosya
def test_real_prompt_file_loads():
    """Config'in gosterdigi prompt sozlesmeye uyuyor mu.

    Versiyon numarasi sabitlenmiyor: prompt bumplendiginde bu testin degismesi
    gerekseydi test, degisimi yakalamak yerine degisime ayak uydururdu. Sabit
    olan sey SOZLESME - versiyon satiri var, SYSTEM dolu, kaynak config'ten geliyor.
    """
    prompt = load_prompt()

    assert re.fullmatch(r"v\d+", prompt.version), f"beklenmeyen versiyon: {prompt.version}"
    assert "gift" in prompt.system.lower()
    assert prompt.source == prompt_path()
    assert prompt.source.exists()


def test_real_prompt_documents_the_critical_distinctions():
    """Projenin tum gerekcesi bu ayrimlarda; prompt'tan dusurulurse yakala.

    `tests/test_keyword_scan.py` ayni uc ayrimi kilitliyor: spekulasyon hediye
    degil, alinan hediye verilen degil, hediye sozcugu olmadan da hediye olabilir.
    """
    system = load_prompt().system.lower()

    # Ornek CUMLELERE degil KURALLARA bakiliyor: ornekler gercek review metniyle
    # ortusmemek icin degistirilebiliyor, kurallar degismemeli.
    assert "realized" in system and "suggested" in system, "gerceklesmis/onerilmis ayrimi yok"
    # v3'te ifade "receiving is not giving" oldu ve etiket `self` degil `received`;
    # kural ayni, sozler degisti - test soze degil kurala bakiyor.
    assert "receiving is not giving" in system, "alan/veren ayrimi dusmus"
    assert "grandchild" in system, "sema revizyonu prompt'a yansimamis"


def test_prompt_keeps_the_rules_learned_from_the_trial_pass():
    """200 satirlik deneme gecisinin urettigi uc kural prompt'ta durmali.

    Ucu de vekilin olculen hatalarindan cikti (kesinlik 0,610 / duyarlilik 0,762).
    Biri dusurulurse LLM ayni hatayi 40.000 satirda tekrarlar ve bunu ancak Hafta
    4'un dogrulamasinda fark ederiz.
    """
    system = load_prompt().system.lower()

    # 1. Satin alma fiili olmadan da gerceklesmis hediye olabilir
    assert "no purchase verb is required" in system
    # 2. Alicinin tepkisi tek basina kanittir (vekilin en buyuk kor noktasi)
    assert "reaction" in system
    # 3. Kanitsiz spekulasyon self degil unclear
    assert "is not evidence of a" in system, "spekulatif -> unclear kurali dusmus"


def test_render_fills_every_placeholder():
    out = render("Wooden Puzzle", "Toys_and_Games", "Nice", "bought for my grandson")

    assert "Wooden Puzzle" in out
    assert "Toys_and_Games" in out
    assert "bought for my grandson" in out
    assert "{" not in out


def test_messages_carry_system_and_user_roles():
    msgs = load_prompt().messages(
        product_title="Wooden Puzzle", category="Toys", title="T", text="body"
    )

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == load_prompt().system


# --------------------------------------------------------------- ayrıştırma
def test_version_is_read_from_the_file(tmp_path: Path):
    prompt = load_prompt(path=_write(tmp_path, MINIMAL))

    assert prompt.version == "v9"


def test_missing_version_line_raises(tmp_path: Path):
    body = MINIMAL.replace("**Versiyon:** v9 · **Model:** test", "no version here")

    with pytest.raises(ValueError, match="Versiyon"):
        load_prompt(path=_write(tmp_path, body))


def test_missing_system_block_raises(tmp_path: Path):
    body = MINIMAL.replace("## SYSTEM", "## NOTES")

    with pytest.raises(ValueError, match="SYSTEM"):
        load_prompt(path=_write(tmp_path, body))


def test_unexpected_placeholder_set_raises(tmp_path: Path):
    """USER blogunda beklenmeyen bir alan varsa render'da degil YUKLEMEDE patla."""
    body = MINIMAL.replace("Review: {text}", "Review: {text}\nExtra: {rating}")

    with pytest.raises(ValueError, match="yer tutucular"):
        load_prompt(path=_write(tmp_path, body))


def test_missing_placeholder_raises(tmp_path: Path):
    body = MINIMAL.replace("Review: {text}", "Review: (yok)")

    with pytest.raises(ValueError, match="yer tutucular"):
        load_prompt(path=_write(tmp_path, body))


def test_required_fields_match_the_renderer_signature():
    """REQUIRED_FIELDS ile render() imzasi ayrisirsa yukleme kontrolu anlamsizlasir."""
    import inspect

    from gift_contamination.detection.prompting import Prompt

    params = set(inspect.signature(Prompt.render).parameters) - {"self"}

    assert params == set(REQUIRED_FIELDS)


def test_default_path_comes_from_the_config_not_a_literal():
    """cfg verilmeden cagrildiginda da config'in gosterdigi dosya donmeli.

    Eskiden dosya adi burada ikinci kez yaziliydi: config v2'ye gectiginde
    `load_prompt()` v1'i okumaya devam ediyordu ve fark yalnizca annotation
    ciktisindaki `prompt_version` alanina bakilinca anlasilabilirdi.
    """
    from gift_contamination.config import Config

    assert prompt_path() == prompt_path(Config.load())


def test_nonexistent_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt(path=tmp_path / "yok.md")
