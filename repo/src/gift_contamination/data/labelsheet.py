"""Elle etiketleme sayfasi: CSV -> xlsx -> CSV gidis donusu (Hafta 2, Hafta 4).

Neden xlsx? Etiketleme dosyasi CSV olarak uretiliyor ama insanin onu Excel'de
acip kaydetmesi iki sessiz hataya aciktir: Turkce Windows yerel ayarinda liste
ayraci `;` oldugu icin virgullu dosya tek kolona duser, ve Excel BOM'suz UTF-8'i
taniyamayip cp1254 ile geri yazar - review metnindeki karakterler bozulur.
Ikisi de ancak saatler suren etiketleme bittikten sonra fark edilir. xlsx'te
ayrac ve kodlama diye bir kavram yok.

ONYARGI KONTROLU: `trial_stratum`, `sample_frame` ve `kw_gift_proxy` kolonlari
sayfaya HIC yazilmaz. Etiketleyen kisi anahtar kelimenin ne dedigini gormeden
karar verir; aksi halde etiketler vekilin hatalarini tekrarlar ve
insan-vekil karsilastirmasi anlamini yitirir. Kolonlar `ingest` sirasinda
`trial_id` uzerinden geri eklenir.

Kullanim:
    python -m gift_contamination.data.labelsheet --export
    # ... Excel'de doldur, kaydet ...
    python -m gift_contamination.data.labelsheet --ingest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..config import Config
from ..data.sampling import LABELS
from ..utils.io import should_skip
from ..utils.logging import get_logger, log_output

log = get_logger("data.labelsheet")

# Etiketleyene GOSTERILEN kolonlar. Liste kasitli olarak kisa.
EXPORT_COLUMNS = ["trial_id", "category", "title", "text", "label", "notes"]

# Sayfaya yazilmayan, ingest'te geri eklenen kolonlar (onyargi kontrolu).
BLIND_COLUMNS = ["row_id", "trial_stratum", "sample_frame", "kw_gift_proxy"]

_WIDTHS = {"trial_id": 8, "category": 16, "title": 34, "text": 96, "label": 14, "notes": 30}


def sheet_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".xlsx")


def labeled_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_labeled.csv")


def default_csv(cfg: Config) -> Path:
    """En son uretilen prompt deneme CSV'si."""
    found = sorted(
        p
        for p in cfg.path("human").glob("prompt_trial_*.csv")
        if not p.stem.endswith("_labeled")
    )
    if not found:
        raise FileNotFoundError(
            "prompt_trial_*.csv bulunamadi. Once uretin:\n"
            "  python -m gift_contamination.data.sampling --trial 200 --category all"
        )
    return found[-1]


# --------------------------------------------------------------------- export
def export(csv_path: Path, *, force: bool = False) -> Path:
    """CSV'yi acilir menulu, dondurulmus baslikli bir xlsx'e cevirir."""
    dest = sheet_path(csv_path)
    if should_skip(dest, force, log):
        return dest

    df = pl.read_csv(csv_path).select(EXPORT_COLUMNS)

    import xlsxwriter

    book = xlsxwriter.Workbook(str(dest), {"strings_to_urls": False})
    sheet = book.add_worksheet("etiketleme")

    head = book.add_format({"bold": True, "bg_color": "#DDDDDD", "border": 1})
    wrap = book.add_format({"text_wrap": True, "valign": "top"})
    plain = book.add_format({"valign": "top"})
    entry = book.add_format({"valign": "top", "bg_color": "#FFF7CC", "border": 1})

    for col, name in enumerate(EXPORT_COLUMNS):
        sheet.write(0, col, name, head)
        fmt = wrap if name in ("title", "text") else (entry if name in ("label", "notes") else plain)
        sheet.set_column(col, col, _WIDTHS[name], fmt)

    for row, record in enumerate(df.iter_rows(named=True), start=1):
        for col, name in enumerate(EXPORT_COLUMNS):
            value = record[name]
            sheet.write(row, col, "" if value is None else value)

    label_col = EXPORT_COLUMNS.index("label")
    sheet.data_validation(
        1,
        label_col,
        df.height,
        label_col,
        {
            "validate": "list",
            "source": list(LABELS),
            "error_title": "Gecersiz etiket",
            "error_message": "Yalnizca: " + ", ".join(LABELS),
        },
    )
    sheet.freeze_panes(1, 0)
    sheet.autofilter(0, 0, df.height, len(EXPORT_COLUMNS) - 1)
    book.close()

    log_output(log, dest, n_rows=df.height)
    log.info("etiket kolonu acilir menulu; onyargi kolonlari sayfada YOK")
    return dest


# --------------------------------------------------------------------- ingest
def ingest(csv_path: Path, *, force: bool = False) -> tuple[Path, dict]:
    """Doldurulmus xlsx'i okur, dogrular ve UTF-8 CSV olarak geri yazar."""
    src = sheet_path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"{src} yok - once --export calistirin")

    import openpyxl

    book = openpyxl.load_workbook(src, data_only=True)
    rows = list(book.active.iter_rows(values_only=True))
    header = [str(c) if c is not None else "" for c in rows[0]]
    if header != EXPORT_COLUMNS:
        raise ValueError(f"basliklar degismis: {header} != {EXPORT_COLUMNS}")

    records = [dict(zip(header, r)) for r in rows[1:] if r[0] is not None]
    filled = pl.DataFrame(
        records,
        schema={
            "trial_id": pl.Int64,
            "category": pl.String,
            "title": pl.String,
            "text": pl.String,
            "label": pl.String,
            "notes": pl.String,
        },
    ).with_columns(pl.col("label").str.strip_chars().replace("", None))

    source = pl.read_csv(csv_path)
    report = _validate(filled, source)

    merged = (
        source.drop("label", "notes")
        .join(filled.select("trial_id", "label", "notes"), on="trial_id", how="left")
        .select(source.columns)
    )
    dest = labeled_path(csv_path)
    if should_skip(dest, force, log):
        return dest, report
    merged.write_csv(dest)
    log_output(log, dest, n_rows=merged.height)
    return dest, report


def _validate(filled: pl.DataFrame, source: pl.DataFrame) -> dict:
    """Satir kaybi, tekrar ve sozluk disi etiket arar; ilerleme raporu doner."""
    missing = set(source["trial_id"]) - set(filled["trial_id"])
    extra = set(filled["trial_id"]) - set(source["trial_id"])
    if missing or extra:
        raise ValueError(
            "trial_id kumesi degismis - satir silinmis veya eklenmis. "
            f"eksik={sorted(missing)[:5]} fazla={sorted(extra)[:5]}"
        )
    if filled["trial_id"].n_unique() != filled.height:
        raise ValueError("tekrar eden trial_id var - satir kopyalanmis")

    bad = filled.filter(
        pl.col("label").is_not_null() & ~pl.col("label").is_in(list(LABELS))
    ).select("trial_id", "label")
    if bad.height:
        raise ValueError(
            f"{bad.height} satirda sozluk disi etiket var. Gecerli: {', '.join(LABELS)}\n{bad.head(10)}"
        )

    done = filled.filter(pl.col("label").is_not_null())
    report = {
        "n_total": filled.height,
        "n_labeled": done.height,
        "by_label": dict(done.group_by("label").len().sort("len", descending=True).iter_rows()),
    }
    if done.height:
        # Insan ile sozcuksel vekil nerede ayrisiyor: prompt v2'nin asil girdisi.
        joined = done.join(source.select("trial_id", "kw_gift_proxy"), on="trial_id")
        gift = pl.col("label") == "gift_given"
        proxy = pl.col("kw_gift_proxy")
        report["vs_keyword_proxy"] = {
            "proxy_dogru": joined.filter(proxy & gift).height,
            "proxy_yanlis_pozitif": joined.filter(proxy & ~gift).height,
            "proxy_kacirdi": joined.filter(~proxy & gift).height,
            "ikisi_de_hayir": joined.filter(~proxy & ~gift).height,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--csv", type=Path, help="Kaynak CSV (varsayilan: en son prompt_trial_*.csv)")
    parser.add_argument("--export", action="store_true", help="xlsx etiketleme sayfasi uret")
    parser.add_argument("--ingest", action="store_true", help="Doldurulmus xlsx'i CSV'ye geri yaz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.export == args.ingest:
        parser.error("--export veya --ingest secin")

    cfg = Config.load(args.config)
    csv_path = args.csv or default_csv(cfg)

    if args.export:
        export(csv_path, force=args.force)
        return 0

    _, report = ingest(csv_path, force=args.force)
    log.info("etiketlenen: %d / %d", report["n_labeled"], report["n_total"])
    for label, n in report["by_label"].items():
        log.info("  %-12s %4d", label, n)
    if "vs_keyword_proxy" in report:
        log.info("sozcuksel vekile karsi: %s", report["vs_keyword_proxy"])
    if report["n_labeled"] < report["n_total"]:
        log.warning("%d satir bos - etiketleme yarim", report["n_total"] - report["n_labeled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
