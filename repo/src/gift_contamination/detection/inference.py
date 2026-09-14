"""Hafta 5 - Tam korpus cikarimi: ogrenci 5-core'un her satirini etiketler.

Girdi : data/interim/<slug>_kcore.parquet, data/interim/<slug>_meta.parquet
        models/distill/<model>/ + reports/results/distill_report_<model>.json
Cikti : data/processed/distill/infer_<slug>.parquet  (prepare; METIN TASIR, git'e girmez)
        data/annotations/<slug>_inferred.parquet     (etiket + olasilik, metin yok)
        reports/results/inference_<slug>.json

IKI ADIM, IKI MAKINE (damitmayla ayni):
  prepare - yerelde, CPU. 5-core satirlarini ogrencinin girdi bicimine cevirir
            (ogretmenin gordugu dort alan).
  run     - Kaggle'da, GPU. Parca parca etiketler; oturum koparsa kaldigi
            yerden devam eder (bitmis parca yeniden hesaplanmaz).

KAPI. Gercek model (`hf`) yalnizca sadakat kapisi PASS ise kosar - kapiyi
gecmemis bir ogrencinin etiketiyle deney kurmak, kapiyi kurmamakla ayni sey.
Stub ogrenci test icindir; ciktisi `label_source: stub` damgasi tasir ve deney
onu raporlanabilir kaynak olarak KABUL ETMEZ.

Kullanim:
    python -m gift_contamination.detection.inference prepare --category high
    python -m gift_contamination.detection.inference run --category high   # Kaggle, GPU
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from ..config import Config, resolve_roles
from ..data.metadata import load_join_frame
from ..data.preprocess import kcore_parquet_path
from ..data.sampling import LABELS
from ..utils.io import read_json, should_skip, write_json
from ..utils.logging import get_logger, log_output
from .distill import (
    MODEL_KEYS,
    distill_dir,
    load_student,
    report_path as distill_report_path,
    student_text,
)
from .llm_annotate import annotation_path, code_version
from .schema import CONDITION_OF, CONTAMINATION

log = get_logger("detection.inference")

# Muhendislik ayari, arastirma parametresi degil: parca basina satir. Kaggle
# oturumu koparsa en fazla bu kadar satirin isi kaybolur.
DEFAULT_CHUNK_ROWS = 200_000
LABEL_SOURCE_REAL = "distilled"
LABEL_SOURCE_STUB = "stub"


def input_path(cfg: Config, role: str) -> Path:
    return distill_dir(cfg) / f"infer_{cfg.category_slug(role)}.parquet"


def parts_dir(cfg: Config, role: str) -> Path:
    return distill_dir(cfg) / f"infer_parts_{cfg.category_slug(role)}"


def inferred_path(cfg: Config, role: str) -> Path:
    return cfg.path("annotations", f"{cfg.category_slug(role)}_inferred.parquet")


def inference_report_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"inference_{cfg.category_slug(role)}.json")


# ------------------------------------------------------------------ prepare
def prepare_inputs(cfg: Config, role: str, *, force: bool = False) -> Path:
    """5-core satirlari -> (row_id, month, text). Urun adi metadata'dan LEFT join."""
    dest = input_path(cfg, role)
    if should_skip(dest, force, log):
        return dest
    src = kcore_parquet_path(cfg, role)
    if not src.exists():
        raise FileNotFoundError(f"{src.name} yok - once preprocess")

    kcore = pl.read_parquet(src, columns=["row_id", "month", "title", "text", "parent_asin"])
    if not kcore.height:
        raise RuntimeError(
            f"{cfg.category_slug(role)} 5-core'u BOS - bu kategori deneye giremez "
            "(All_Beauty icin olculdu)."
        )
    df = kcore.join(load_join_frame(cfg, role), on="parent_asin", how="left")
    if df.height != kcore.height:
        raise RuntimeError(
            f"metadata join satir cogaltti ({kcore.height} -> {df.height}); "
            "parent_asin metadata'da tekrar ediyor olabilir"
        )
    slug = cfg.category_slug(role)
    texts = [
        student_text(slug, p, t, x)
        for p, t, x in df.select("product_title", "title", "text").iter_rows()
    ]
    out = df.select("row_id", "month").with_columns(pl.Series("text", texts, dtype=pl.String))

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    n_missing = int(df["product_title"].fill_null("").str.strip_chars().eq("").sum())
    log_output(log, dest, n_rows=out.height)
    log.info("%s: %s/%s satirda urun adi yok (girdide '(unknown)')",
             slug, f"{n_missing:,}", f"{out.height:,}")
    return dest


# ---------------------------------------------------------------------- run
def _gate(cfg: Config, model_key: str, backend: str) -> str:
    """Kapi kontrolu; donen deger raporlanacak kapi karari."""
    path = distill_report_path(cfg, model_key)
    if not path.exists():
        if backend == "stub":
            return "NONE"
        raise FileNotFoundError("distill_report.json yok - once `distill train`")
    rapor = read_json(path)
    if rapor["meta"]["model_key"] != model_key:
        raise RuntimeError(
            f"sadakat raporu '{rapor['meta']['model_key']}' modeli icin, "
            f"'{model_key}' isteniyor - rapor dosyasi elle tasinmis olabilir."
        )
    if backend != "stub" and rapor["verdict"] != "PASS":
        raise RuntimeError(
            f"sadakat kapisi {rapor['verdict']}. Kapiyi gecmemis ogrencinin etiketi "
            "deneye giremez (DECISIONS 2026-09-14)."
        )
    return str(rapor["verdict"])


def seasonality(df: pl.DataFrame, cfg: Config) -> dict:
    """Etiketsiz dis gecerlilik: Aralik-Ocak / yaz orani, C1 ve C1b ekseninde.

    Kapi 1'in birinci olcutunun aynisi (esik 1,25). Burada KAPI DEGIL - 5-core'da
    ogrencinin mevsimsel imzayi koruyup korumadigini gosteren bir sagdama.
    """
    peak = list(cfg.get("gate1.peak_months"))
    trough = list(cfg.get("gate1.trough_months"))
    out = {"peak_months": peak, "trough_months": trough,
           "gate1_reference_min": cfg.get("gate1.seasonality_ratio_min")}
    for ad, etiketler in CONTAMINATION.items():
        isaret = pl.col("purchase_type").is_in(list(etiketler))
        tepe = df.filter(pl.col("month").is_in(peak)).select(isaret.mean()).item()
        cukur = df.filter(pl.col("month").is_in(trough)).select(isaret.mean()).item()
        out[CONDITION_OF[ad]] = {
            "rate_peak": None if tepe is None else round(float(tepe), 5),
            "rate_trough": None if cukur is None else round(float(cukur), 5),
            "ratio": (round(float(tepe) / float(cukur), 3) if tepe is not None and cukur else None),
        }
    return out


def run(
    cfg: Config, role: str, *, model_key: str = "base", backend: str = "hf",
    fp16: bool = True, chunk_rows: int = DEFAULT_CHUNK_ROWS, force: bool = False,
) -> Path:
    dest = inferred_path(cfg, role)
    if should_skip(dest, force, log):
        return dest
    src = input_path(cfg, role)
    if not src.exists():
        raise FileNotFoundError(f"{src.name} yok - once `inference prepare --category {role}`")
    kapi = _gate(cfg, model_key, backend)

    student = load_student(cfg, model_key, backend, fp16=fp16)
    parts = parts_dir(cfg, role)
    parts.mkdir(parents=True, exist_ok=True)
    girdi = pl.scan_parquet(src)
    n = girdi.select(pl.len()).collect().item()

    t0 = time.perf_counter()
    n_hesaplanan = 0
    for i, start in enumerate(range(0, n, chunk_rows)):
        part = parts / f"part_{i:05d}.parquet"
        if part.exists() and not force:
            continue   # kesinti sonrasi devam: bitmis parca yeniden hesaplanmaz
        chunk = girdi.slice(start, chunk_rows).select("row_id", "text").collect()
        probs = student.predict_proba(chunk["text"].to_list())
        pl.DataFrame({
            "row_id": chunk["row_id"],
            "purchase_type": [LABELS[j] for j in probs.argmax(axis=1)],
            **{f"p_{lab}": pl.Series(probs[:, k], dtype=pl.Float32) for k, lab in enumerate(LABELS)},
        }).write_parquet(part)
        n_hesaplanan += chunk.height
        gecen = time.perf_counter() - t0
        log.info("%s: parca %d bitti (%s/%s satir, %.0f satir/sn)", role, i,
                 f"{min(start + chunk_rows, n):,}", f"{n:,}", n_hesaplanan / max(gecen, 1e-9))
    gecen = time.perf_counter() - t0

    beklenen = (n + chunk_rows - 1) // chunk_rows
    dosyalar = sorted(parts.glob("part_*.parquet"))
    if len(dosyalar) != beklenen:
        raise RuntimeError(
            f"{len(dosyalar)} parca var, {beklenen} bekleniyordu - farkli chunk_rows ile "
            "yarim kalmis bir kosu olabilir. Parca klasorunu silip yeniden baslatin."
        )
    etiket = pl.concat([pl.read_parquet(p) for p in dosyalar], how="vertical")
    if etiket.height != n or etiket["row_id"].n_unique() != n:
        raise RuntimeError(
            f"cikti {etiket.height} satir ({etiket['row_id'].n_unique()} benzersiz), "
            f"girdi {n}. Deney her 5-core satiri icin TAM OLARAK bir etiket bekliyor."
        )
    kaynak = LABEL_SOURCE_STUB if student.name == "stub" else LABEL_SOURCE_REAL
    etiket = etiket.with_columns(
        pl.lit(kaynak).alias("label_source"),
        pl.lit(student.model_name).alias("model"),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    etiket.write_parquet(dest)
    log_output(log, dest, n_rows=etiket.height)

    aylik = etiket.select("row_id", "purchase_type").join(
        girdi.select("row_id", "month").collect(), on="row_id", how="left"
    )
    ortak = _overlap_with_llm(cfg, role, etiket)
    write_json(
        {
            "category": cfg.category_slug(role),
            "n_rows": etiket.height,
            "model": student.model_name,
            "model_key": model_key,
            "label_source": kaynak,
            "distill_gate_verdict": kapi,
            "code_version": code_version(),
            # Yalnizca BU kosuda hesaplanan parcalar; devam eden kosuda eksik gorunur.
            "throughput": {
                "rows_computed_this_run": n_hesaplanan,
                "elapsed_s": round(gecen, 1),
                "rows_per_s": round(n_hesaplanan / gecen, 1) if n_hesaplanan and gecen else None,
            },
            "label_share_pct": {
                k: round(100 * v / etiket.height, 3)
                for k, v in sorted(etiket.group_by("purchase_type").len().iter_rows())
            },
            "contamination_share_pct": {
                CONDITION_OF[ad]: round(
                    100 * etiket.filter(pl.col("purchase_type").is_in(list(e))).height
                    / etiket.height, 3)
                for ad, e in CONTAMINATION.items()
            },
            "seasonality": seasonality(aylik, cfg),
            "overlap_with_llm": ortak,
        },
        inference_report_path(cfg, role),
        log,
    )
    return dest


def _overlap_with_llm(cfg: Config, role: str, etiket: pl.DataFrame) -> dict:
    """5-core'da LLM'in de etiketledigi satirlarda uyum - IYIMSER, kapi degil.

    Bu satirlarin cogu ogrencinin EGITIM verisindeydi; uyum yuksek cikmasi
    beklenir ve bir basari olcusu olarak okunmamali. Tarafsiz sayi damitma
    raporundaki `student_vs_teacher_holdout_in_kcore`.
    """
    path = annotation_path(cfg, role)
    if not path.exists():
        return {"skipped": True, "reason": "LLM etiketi yok"}
    llm = pl.read_parquet(path, columns=["row_id", "purchase_type"]).rename(
        {"purchase_type": "llm"}
    )
    j = etiket.select("row_id", "purchase_type").join(llm, on="row_id", how="inner")
    if not j.height:
        return {"n": 0}
    return {
        "n": j.height,
        "agreement": round(float((j["purchase_type"] == j["llm"]).mean()), 4),
        "note": "iyimser: bu satirlarin cogu egitimdeydi",
    }


# -------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("prepare", "run"))
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default="high")
    parser.add_argument("--model", default="base", choices=tuple(MODEL_KEYS))
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        if args.step == "prepare":
            prepare_inputs(cfg, role, force=args.force)
        else:
            run(cfg, role, model_key=args.model, backend=args.backend,
                fp16=not args.no_fp16, chunk_rows=args.chunk_rows, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
