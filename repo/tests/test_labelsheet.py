"""Etiketleme sayfasi gidis-donus testleri.

Bu modulun engelledigi hata sinifi tek cumlede: insan 2-3 saat etiketler, sonra
dosya sessizce bozulmus olur. Uc yol var - kodlama (Turkce karakterler cp1254'e
duser), ayrac (Turkce yerel ayarda `;`), ve satir kaybi (Excel'de yanlislikla
silinen/siralanan satir). Ucu de burada test ediliyor.

Dorduncusu daha ince: `kw_gift_proxy` kolonu sayfada gorunurse etiketleyen kisi
vekilin karari ne ise ona yaslanir, insan-vekil karsilastirmasi da tautoloji
haline gelir. `test_export_hides_the_bias_columns` bunu kilitliyor.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from gift_contamination.data.labelsheet import (
    BLIND_COLUMNS,
    EXPORT_COLUMNS,
    export,
    ingest,
    labeled_path,
    sheet_path,
)
from gift_contamination.data.sampling import LABELS

# Turkce karakterler kasitli: cp1254'e dusen bir gidis-donus bunlarda kirilir.
TEXTS = [
    "Bought this for my grandson's birthday.",
    "Great product, I use it every day.",
    "Cok guzel bir urun - kizim icin aldim, sasirdi.",
    "Would make a great gift for anyone.",
]


@pytest.fixture()
def trial_csv(tmp_path: Path) -> Path:
    """sampling.build_trial ciktisiyla ayni kolon duzenine sahip kucuk CSV."""
    path = tmp_path / "prompt_trial_4.csv"
    pl.DataFrame(
        {
            "trial_id": [1, 2, 3, 4],
            "category": ["Toys_and_Games", "All_Beauty", "Toys_and_Games", "Video_Games"],
            "row_id": [11, 22, 33, 44],
            "title": ["Perfect", "Nice", "Harika", "Five Stars"],
            "text": TEXTS,
            "label": ["", "", "", ""],
            "notes": ["", "", "", ""],
            "trial_stratum": ["proxy", "unflagged", "proxy", "speculative"],
            "sample_frame": ["boost", "main", "main", "main"],
            "kw_gift_proxy": [True, False, True, False],
        }
    ).write_csv(path)
    return path


def _fill(path: Path, labels: list[str | None]) -> None:
    """xlsx'in etiket kolonunu doldurur - insanin Excel'de yaptigi sey."""
    import openpyxl

    book = openpyxl.load_workbook(sheet_path(path))
    sheet = book.active
    col = EXPORT_COLUMNS.index("label") + 1
    for row, value in enumerate(labels, start=2):
        sheet.cell(row=row, column=col, value=value)
    book.save(sheet_path(path))


# ------------------------------------------------------------------- export
def test_export_hides_the_bias_columns(trial_csv: Path):
    """Anahtar kelime karari sayfada GORUNMEMELI.

    Gorunurse etiketler vekilin hatalarini tekrarlar ve Hafta 4'teki
    insan-vekil karsilastirmasi kendi kendini dogrulayan bir olcume doner.
    """
    import openpyxl

    export(trial_csv)

    sheet = openpyxl.load_workbook(sheet_path(trial_csv)).active
    header = [c.value for c in sheet[1]]
    assert header == EXPORT_COLUMNS
    for hidden in BLIND_COLUMNS:
        assert hidden not in header


def test_export_offers_only_valid_labels(trial_csv: Path):
    """Acilir menu sozluk disi etiketi kaynagında engeller."""
    import openpyxl

    export(trial_csv)

    sheet = openpyxl.load_workbook(sheet_path(trial_csv)).active
    sources = " ".join(dv.formula1 for dv in sheet.data_validations.dataValidation)
    for label in LABELS:
        assert label in sources


def test_export_is_idempotent(trial_csv: Path):
    first = export(trial_csv)
    stamp = first.stat().st_mtime_ns

    export(trial_csv)

    assert first.stat().st_mtime_ns == stamp


# ------------------------------------------------------------------- ingest
def test_round_trip_preserves_text_exactly(trial_csv: Path):
    """Asil sigorta: Turkce karakterler ve metin birebir geri gelmeli."""
    export(trial_csv)
    _fill(trial_csv, ["gift_given", "self", "household", "self"])

    dest, _ = ingest(trial_csv)
    out = pl.read_csv(dest)

    assert out["text"].to_list() == TEXTS
    assert out["title"].to_list() == ["Perfect", "Nice", "Harika", "Five Stars"]
    assert dest == labeled_path(trial_csv)


def test_round_trip_keeps_the_blind_columns(trial_csv: Path):
    """Sayfada gosterilmeyen kolonlar ciktida geri gelmeli - Hafta 4 onlari kullaniyor."""
    export(trial_csv)
    _fill(trial_csv, ["gift_given", "self", "household", "self"])

    dest, _ = ingest(trial_csv)
    out = pl.read_csv(dest)

    assert out.columns == pl.read_csv(trial_csv).columns
    assert out["kw_gift_proxy"].to_list() == [True, False, True, False]
    assert out["row_id"].to_list() == [11, 22, 33, 44]


def test_ingest_rejects_a_label_outside_the_dictionary(trial_csv: Path):
    """Acilir menu atlatilabilir (yapistirma); ikinci savunma hatti burasi."""
    export(trial_csv)
    _fill(trial_csv, ["gift", "self", "household", "self"])

    with pytest.raises(ValueError, match="sozluk disi"):
        ingest(trial_csv)


def test_ingest_rejects_a_deleted_row(trial_csv: Path):
    """Excel'de yanlislikla silinen satir sessizce kaybolmamali."""
    import openpyxl

    export(trial_csv)
    book = openpyxl.load_workbook(sheet_path(trial_csv))
    book.active.delete_rows(3)
    book.save(sheet_path(trial_csv))

    with pytest.raises(ValueError, match="trial_id kumesi degismis"):
        ingest(trial_csv)


def test_partial_labelling_is_allowed_and_counted(trial_csv: Path):
    """Etiketleme boluneble - yarim dosya hata degil, rapor edilecek durum."""
    export(trial_csv)
    _fill(trial_csv, ["gift_given", None, "self", None])

    _, report = ingest(trial_csv)

    assert report["n_labeled"] == 2
    assert report["n_total"] == 4
    assert report["by_label"] == {"gift_given": 1, "self": 1}


def test_report_compares_the_human_against_the_keyword_proxy(trial_csv: Path):
    """Prompt v2'nin girdisi bu tablo: vekil nerede yaniliyor, nerede kaciriyor."""
    export(trial_csv)
    # 1: proxy=True  & gift_given -> dogru
    # 2: proxy=False & self       -> ikisi de hayir
    # 3: proxy=True  & self       -> vekil yanlis pozitif
    # 4: proxy=False & gift_given -> vekil kacirdi
    _fill(trial_csv, ["gift_given", "self", "self", "gift_given"])

    _, report = ingest(trial_csv)

    assert report["vs_keyword_proxy"] == {
        "proxy_dogru": 1,
        "proxy_yanlis_pozitif": 1,
        "proxy_kacirdi": 1,
        "ikisi_de_hayir": 1,
    }


def test_ingest_without_export_fails_loudly(trial_csv: Path):
    with pytest.raises(FileNotFoundError, match="once --export"):
        ingest(trial_csv)
