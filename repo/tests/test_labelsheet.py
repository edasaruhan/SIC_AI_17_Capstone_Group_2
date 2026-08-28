"""Etiketleme sayfasi gidis-donus testleri.

Bu modulun engelledigi hata sinifi tek cumlede: insan 2-3 saat etiketler, sonra
dosya sessizce bozulmus olur. Uc yol var - kodlama (Turkce karakterler cp1254'e
duser), ayrac (Turkce yerel ayarda `;`), ve satir kaybi (Excel'de yanlislikla
silinen/siralanan satir). Ucu de burada test ediliyor.

Dorduncusu daha ince: `kw_gift_proxy` kolonu sayfada gorunurse etiketleyen kisi
vekilin karari ne ise ona yaslanir, insan-vekil karsilastirmasi da tautoloji
haline gelir. `test_export_hides_the_bias_columns` bunu kilitliyor.

Besincisi Hafta 4'un ta kendisi: LLM'in cevabi (`purchase_type` ve arkadaslari)
sayfaya sizarsa olculen sey detektorun DOGRULUGU olmaktan cikar, telkine uyum
olur - ve o rakam raporda "insan dogrulamasi" diye gecer.
`test_export_hides_the_llm_answer` bunu kilitliyor.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from gift_contamination.data.labelsheet import (
    BLIND_COLUMNS,
    EXPORT_COLUMNS,
    export,
    export_columns,
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
    # `export` her zaman liste doner - tek etiketleyicide de, uc kisilik
    # Hafta 4 akisinda da ayni imza.
    (first,) = export(trial_csv)
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


# ------------------------------------------- Hafta 4: uc etiketleyici + korleme


@pytest.fixture()
def validation_csv_file(tmp_path: Path) -> Path:
    """`sampling.build_validation` ciktisiyla ayni kolon duzenine sahip kucuk CSV."""
    path = tmp_path / "validation_4.csv"
    pl.DataFrame(
        {
            "val_id": [1, 2, 3, 4],
            "category": ["Toys_and_Games", "All_Beauty", "Toys_and_Games", "Video_Games"],
            "row_id": [11, 22, 33, 44],
            "title": ["Perfect", "Nice", "Harika", "Five Stars"],
            "text": TEXTS,
            "label": ["", "", "", ""],
            "notes": ["", "", "", ""],
            "val_stratum": ["gift_given", "self", "household", "unclear"],
            "sample_frame": ["boost", "main", "main", "main"],
            "kw_gift_proxy": [True, False, True, False],
            # LLM'in kendi cevabi - sayfaya GECMEMELI
            "purchase_type": ["gift_given", "self", "household", "unclear"],
            "confidence": ["high", "high", "medium", "low"],
            "recipient": ["grandchild", "unknown", "child", "unknown"],
            "occasion": ["birthday", "none", "none", "unknown"],
            "evidence_span": ["for my grandson", "", "my kids", ""],
        }
    ).write_csv(path)
    return path


def _fill_tagged(path: Path, tag: str, labels: list[str | None]) -> None:
    import openpyxl

    book = openpyxl.load_workbook(sheet_path(path, tag))
    sheet = book.active
    col = export_columns("val_id").index("label") + 1
    for row, value in enumerate(labels, start=2):
        sheet.cell(row=row, column=col, value=value)
    book.save(sheet_path(path, tag))


def test_export_hides_the_llm_answer(validation_csv_file: Path):
    """Hafta 4'un TEK isi detektorun dogrulugunu olcmek.

    Etiketleyen kisi modelin cevabini gorurse olculen sey doğruluk degil
    telkine uyum olur - ve o rakam raporda "insan dogrulamasi" diye gecer.
    `evidence_span` etiketlemeyi hizlandirirdi ama modelin gerekcesini de
    gosterir; bagimsizlik hiza tercih ediliyor.
    """
    import openpyxl

    export(validation_csv_file, annotators=1)

    sheet = openpyxl.load_workbook(sheet_path(validation_csv_file)).active
    header = [c.value for c in next(sheet.iter_rows(max_row=1))]
    assert header == export_columns("val_id")
    for sizinti in (
        "purchase_type", "confidence", "recipient", "occasion",
        "evidence_span", "val_stratum", "kw_gift_proxy",
    ):
        assert sizinti not in header
        assert sizinti in BLIND_COLUMNS


def test_three_annotators_get_identical_sheets(validation_csv_file: Path):
    """Uc dosya AYNI satirlari AYNI sirada tasimali - kappa hizalamasi buna bagli."""
    import openpyxl

    written = export(validation_csv_file, annotators=3)

    assert [p.name for p in written] == [
        "validation_4_A.xlsx", "validation_4_B.xlsx", "validation_4_C.xlsx",
    ]
    goruntuler = []
    for tag in ("A", "B", "C"):
        sheet = openpyxl.load_workbook(sheet_path(validation_csv_file, tag)).active
        goruntuler.append([r[:5] for r in sheet.iter_rows(values_only=True)])
    assert goruntuler[0] == goruntuler[1] == goruntuler[2]


def test_ingest_keeps_the_three_label_columns_apart(validation_csv_file: Path):
    """Uc etiketleyicinin cevabi AYRI kolonlarda gelmeli.

    Tek bir `label` kolonuna katlamak uyusmazligi yok eder ve Fleiss kappa
    hesaplanamaz hale gelir.
    """
    export(validation_csv_file, annotators=3)
    _fill_tagged(validation_csv_file, "A", ["gift_given", "self", "household", "self"])
    _fill_tagged(validation_csv_file, "B", ["gift_given", "self", "gift_given", "self"])
    _fill_tagged(validation_csv_file, "C", ["gift_given", "self", "household", "unclear"])

    dest, report = ingest(validation_csv_file, annotators=3)
    out = pl.read_csv(dest)

    assert out["label_A"].to_list() == ["gift_given", "self", "household", "self"]
    assert out["label_B"].to_list() == ["gift_given", "self", "gift_given", "self"]
    assert out["label_C"].to_list() == ["gift_given", "self", "household", "unclear"]
    # Kor kolonlar geri gelmis olmali - analiz onlari kullaniyor
    assert out["purchase_type"].to_list() == ["gift_given", "self", "household", "unclear"]
    assert set(report["by_annotator"]) == {"A", "B", "C"}
    assert report["by_annotator"]["A"]["n_labeled"] == 4


def test_a_missing_annotator_sheet_fails_loudly(validation_csv_file: Path):
    """Iki kisi doldurup ucuncusu unutursa kappa yanlis hesaplanir.

    Sessizce iki kisiyle devam etmek yerine hata veriyoruz.
    """
    export(validation_csv_file, annotators=3)
    sheet_path(validation_csv_file, "C").unlink()

    with pytest.raises(FileNotFoundError, match="_C.xlsx"):
        ingest(validation_csv_file, annotators=3)


def test_id_column_must_be_unambiguous(tmp_path: Path):
    """Iki kimlik kolonu birden varsa hangisinin anahtar oldugu belirsizdir."""
    path = tmp_path / "karisik.csv"
    pl.DataFrame({
        "trial_id": [1], "val_id": [1], "category": ["x"],
        "title": ["t"], "text": ["metin metin"], "label": [""], "notes": [""],
    }).write_csv(path)

    with pytest.raises(ValueError, match="belirsiz"):
        export(path)
