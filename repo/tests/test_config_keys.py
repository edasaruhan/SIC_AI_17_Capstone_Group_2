"""`configs/base.yaml`da ÖLÜ ANAHTAR bırakmıyoruz.

base.yaml'in ilk satiri "Kodda hardcode parametre YOK" diyor; bunun tersi de
gecerli olmali - config'te de KODSUZ parametre olmamali. Kimsenin okumadigi
bir anahtar sessiz bir tuzaktir: birisi onu duzeltir, hicbir sey degismez ve
degismedigini fark etmesinin bir yolu yoktur.

Bu tam olarak yasandi (denetim 2026-08-29): `analysis.expected_peaks` olcume
gore duzeltilmis ve DECISIONS'a yazilmisti, ama `eda.py` ayni degeri kendi
icinde `(12, 1)` olarak tasiyordu ve anahtari hic okumuyordu.

KURAL: bir anahtar ya kodda okunur, ya da kendi satirinda / iceren blogunda
"HENÜZ OKUNMUYOR" diye isaretlenir. Ucuncu secenek yok.

Not: "okunuyor mu" kontrolu metin aramasidir ve MUHAFAZAKARDIR - yanlislikla
"okunuyor" diyebilir (ayni adi tasiyan baska bir degisken yuzunden), ama
okunan bir anahtari "olu" diye isaretlemez. Yani bu test yanlis alarm vermez.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "base.yaml"
MARK = ("HENÜZ OKUNMUYOR", "HENUZ OKUNMUYOR")


def _leaf_paths(node: dict, prefix: str = "") -> list[tuple[str, str]]:
    """(tam.yol, son_anahtar) ciftleri - yalnizca yaprak dugumler."""
    out: list[tuple[str, str]] = []
    for key, value in node.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_leaf_paths(value, full))
        else:
            out.append((full, key))
    return out


def _source_text() -> str:
    parts = []
    for folder in ("src", "scripts"):
        for path in (REPO / folder).rglob("*.py"):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _marked_keys(raw: str) -> set[str]:
    """İşaretli anahtarlar: kendi satırında ya da içeren bloğunda not olanlar.

    Girinti takip edilerek bir bloğun üstündeki not, bloğun tüm alt
    anahtarlarına miras kalır.
    """
    marked: set[str] = set()
    lines = raw.splitlines()
    # (girinti, hala_gecerli_mi) yigini
    stack: list[tuple[int, bool]] = []
    pending = False

    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if any(m in line for m in MARK):
                pending = True
            continue

        match = re.match(r"(\s*)([A-Za-z_][\w-]*)\s*:", line)
        if not match:
            continue
        indent, key = len(match.group(1)), match.group(2)

        while stack and stack[-1][0] >= indent:
            stack.pop()

        inherited = bool(stack and stack[-1][1])
        own = pending or any(m in line for m in MARK)
        if inherited or own:
            marked.add(key)
        stack.append((indent, inherited or own))
        pending = False

    return marked


def test_every_config_key_is_read_by_code_or_marked_unread():
    raw = CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)
    source = _source_text()
    marked = _marked_keys(raw)

    unread_and_unmarked = []
    for full, leaf in _leaf_paths(config):
        read = any(
            token in source
            for token in (f'"{full}"', f"'{full}'", f'"{leaf}"', f"'{leaf}'")
        )
        if not read and leaf not in marked:
            unread_and_unmarked.append(full)

    assert not unread_and_unmarked, (
        "Bu anahtarlar hiçbir yerde okunmuyor ve 'HENÜZ OKUNMUYOR' diye de "
        "işaretlenmemiş. Ya okuyan kodu yazın, ya anahtarı silin, ya da "
        f"neden durduğunu yazın: {unread_and_unmarked}"
    )


def test_the_marker_scanner_actually_detects_an_unmarked_key():
    """Tarayıcının kendisi çalışıyor mu — testin yanlışlıkla hep geçmediğini gösterir."""
    isaretsiz = _marked_keys("alpha:\n  beta: 1\n")
    assert "beta" not in isaretsiz

    isaretli = _marked_keys("# HENÜZ OKUNMUYOR\nalpha:\n  beta: 1\n")
    assert "beta" in isaretli, "blok notu alt anahtarlara miras kalmalı"
    assert "alpha" in isaretli

    satir_ici = _marked_keys("alpha:\n  beta: 1   # HENÜZ OKUNMUYOR\n")
    assert "beta" in satir_ici
    assert "alpha" not in satir_ici, "kardeş bloğa sızmamalı"
