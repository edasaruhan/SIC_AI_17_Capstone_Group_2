"""Prompt yukleme ve render testleri.

Bu modulun tek isi sessiz basarisizligi engellemek. 40-60K'lik bir kosuda bos bir
SYSTEM prompt'u ya da doldurulmamis bir `{text}` yer tutucusu ile devam etmek,
hatanin ancak sonuclara bakildiginda anlasilmasi demektir - ve o noktada Kaggle
kotasi harcanmis olur.
"""

from __future__ import annotations

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
    """Repodaki v1 dosyasi sozlesmeye uyuyor mu."""
    prompt = load_prompt()

    assert prompt.version == "v1"
    assert "gift" in prompt.system.lower()
    assert prompt.source == prompt_path()


def test_real_prompt_documents_the_critical_distinctions():
    """Projenin tum gerekcesi bu ayrimlarda; prompt'tan dusurulurse yakala.

    `prompts/gift_detection_v1.md` ve `tests/test_keyword_scan.py` ayni uc
    ayrimi kilitliyor: spekulasyon hediye degil, alinan hediye verilen degil,
    hediye sozcugu olmadan da hediye olabilir.
    """
    system = load_prompt().system.lower()

    assert "would make a great gift" in system
    assert "receiving a gift is not giving one" in system
    assert "grandchild" in system, "v2 sema revizyonu prompt'a yansimamis"


def test_render_fills_every_placeholder():
    out = render("Toys_and_Games", "Nice", "bought for my grandson")

    assert "Toys_and_Games" in out
    assert "bought for my grandson" in out
    assert "{" not in out


def test_messages_carry_system_and_user_roles():
    msgs = load_prompt().messages(category="Toys", title="T", text="body")

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


def test_nonexistent_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prompt(path=tmp_path / "yok.md")
