"""Hafta 5 - Damitma: LLM etiketlerinden kucuk ve hizli bir ogrenci.

Girdi : data/annotations/<slug>_llm.parquet            (LLM etiketleri, 4 kategori)
        data/interim/<slug>_annotation_sample.parquet  (review metni + urun adi)
        data/annotations/human/validation_500_labeled.csv (A'nin etiketleri)
        reports/results/validation_500.json            (ogretmenin C1 F1'i)
Cikti : data/processed/distill/bundle.parquet  (egitim + ayrilmis; METIN TASIR, git'e girmez)
        data/processed/distill/human.parquet   (500 insan satiri; METIN TASIR, git'e girmez)
        reports/results/distill_bundle.json    (sayilar, metin yok)
        models/distill/<model>/                (git'e girmez)
        reports/results/distill_report_<base|fallback>.json  (sadakat kapisi, model basina)

NEDEN DAMITMA. Deney korpusu (Toys + Grocery 5-core) 4,6 milyon satir. LLM
~3,3 satir/sn ile bunu 16 gunde etiketlerdi. 47.200 LLM etiketiyle egitilmis
bir siniflandirici ayni isi dakikalar icinde yapar.

IKI ADIM, IKI MAKINE:
  prepare  - yerelde, CPU. Egitim setini kurar; dogrulama satirlarini DISLAR.
  train    - Kaggle'da, GPU. Egitir ve sadakat kapisini olcer.

SIZINTI KURALI - DEGISTIRILEMEZ. 500 dogrulama satiri 47.200'un ICINDE. Ogrenci
onlari egitimde gorurse "ogrenci insana karsi" olcumu kendi egitim verisini
ezbere okumasini olcer. Dislama `(category, row_id)` CIFTI uzerinden - row_id
her kategoride 0'dan basliyor (`sampling._excluded_pairs` ile ayni gerekce).

GIRDI OGRETMENINKIYLE AYNI. LLM dort alan gordu: urun adi, kategori, review
basligi ve metni (6.000 karakter). Ogrenci de ayni dortunu goruyor - farkli
bir girdi, sadakat olcumunu model farkiyla girdi farkini karistiran bir sayiya
cevirirdi.

SECIM YOK. Sabit sayida epoch egitilir, ara kontrol noktasi SECILMEZ. Ayrilmis
kume uzerinden model secmek, kapiyi ayni kume uzerinde olcunce iyimser bir sayi
uretirdi.

SADAKAT KAPISI (configs/base.yaml -> distill.fidelity) HICBIR MODEL EGITILMEDEN
yazildi (DECISIONS 2026-09-14, commit 8c697a8).

Kullanim:
    python -m gift_contamination.detection.distill prepare
    python -m gift_contamination.detection.distill train                  # Kaggle, GPU
    python -m gift_contamination.detection.distill train --model fallback # kapi kacarsa bir kez
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..data.preprocess import kcore_parquet_path
from ..data.sampling import LABELS, _stratum_seed, annotation_sample_path
from ..utils.io import read_json, should_skip, write_json
from ..utils.logging import get_logger, log_output
from .llm_annotate import UNKNOWN_PRODUCT, _truncate, annotation_path, code_version
from .schema import CONDITION_OF, CONTAMINATION

log = get_logger("detection.distill")

SPLIT_TRAIN = "train"
SPLIT_HOLDOUT = "holdout"
MODEL_KEYS = {"base": "distill.base_model", "fallback": "distill.fallback_model"}
BACKENDS = ("hf", "stub")

# Muhendislik ayari, arastirma parametresi degil: tahminde bir grubun tasiyabilecegi
# en fazla token (satir x gruptaki en uzun satir). Sabit satirli grup, uzunluga gore
# siralanmis listenin sonunda 256 x 1024 token'lik bir grup kurar; T4'te FlashAttention
# yok ve SDPA matematik cekirdegine duserse yalnizca dikkat matrisi ~6 GB tutar.
# Butce kisa satirlarda grubu buyutur, uzunlarda kucultur - sonucu degistirmez.
PREDICT_MAX_BATCH_TOKENS = 65_536
PREDICT_MAX_BATCH_ROWS = 1_024


# ------------------------------------------------------------------- yollar
def distill_dir(cfg: Config) -> Path:
    return cfg.path("processed", "distill")


def bundle_path(cfg: Config) -> Path:
    return distill_dir(cfg) / "bundle.parquet"


def human_path(cfg: Config) -> Path:
    return distill_dir(cfg) / "human.parquet"


def bundle_report_path(cfg: Config) -> Path:
    return cfg.path("results", "distill_bundle.json")


def report_path(cfg: Config, model_key: str = "base") -> Path:
    """Model basina AYRI rapor: yedek modelin kosusu ana modelin FAIL kanitini ezmemeli."""
    return cfg.path("results", f"distill_report_{model_key}.json")


def model_dir(cfg: Config, model_key: str) -> Path:
    name = str(cfg.get(MODEL_KEYS[model_key])).replace("/", "__")
    return cfg.path("models", "distill", name)


# -------------------------------------------------------------------- girdi
def student_text(category: str, product_title: str | None, title: str | None,
                 text: str | None) -> str:
    """Ogretmenin gordugu dort alan, ayni kirpmayla (`llm_annotate._truncate`)."""
    body, _ = _truncate(text or "")
    urun = (product_title or "").strip() or UNKNOWN_PRODUCT
    return (
        f"Category: {category}\n"
        f"Product: {urun}\n"
        f"Title: {(title or '').strip()}\n"
        f"Review: {body}"
    )


def _texts(df: pl.DataFrame) -> list[str]:
    return [
        student_text(c, p, t, x)
        for c, p, t, x in df.select("category", "product_title", "title", "text").iter_rows()
    ]


# ------------------------------------------------------------------ prepare
def _validation_frame(cfg: Config) -> pl.DataFrame:
    """A'nin etiketleri + LLM etiketi, `(category, row_id)` anahtariyla."""
    from ..analysis.validation import label_columns, load_labels, reference_labels  # noqa: PLC0415

    df = load_labels(cfg)
    df = df.with_columns(reference_labels(df, label_columns(df)))
    return df.select("val_id", "category", "row_id", "reference", "purchase_type")


def _stratified_split(df: pl.DataFrame, *, fraction: float, seed: int) -> pl.Series:
    """(kategori x etiket) katmani icinde deterministik ayirma.

    Katman basina turetilmis seed: tek bir seed butun katmanlarda ayni konumlari
    secerdi (2026-08-27 denetimindeki hatanin aynisi).
    """
    split = np.full(df.height, SPLIT_TRAIN, dtype=object)
    for (cat, lab), grp in df.with_row_index("_i").group_by(
        "category", "label", maintain_order=True
    ):
        idx = grp["_i"].to_numpy()
        n_hold = int(round(len(idx) * fraction))
        if n_hold == 0 or len(idx) < 2:
            continue
        rng = np.random.default_rng(_stratum_seed(seed, f"distill|{cat}|{lab}"))
        split[rng.choice(idx, size=n_hold, replace=False)] = SPLIT_HOLDOUT
    return pl.Series("split", split.tolist(), dtype=pl.String)


def prepare(cfg: Config, roles: list[str], *, force: bool = False) -> Path:
    """Egitim + ayrilmis kume ve 500 insan satirini kurar. Yerelde, CPU."""
    dest = bundle_path(cfg)
    if should_skip(dest, force, log):
        return dest

    parts = []
    for role in roles:
        path = annotation_path(cfg, role)
        if not path.exists():
            raise FileNotFoundError(f"{path.name} yok - once LLM etiketleri gerekli")
        labels = pl.read_parquet(
            path, columns=["row_id", "category", "sample_frame", "purchase_type",
                           "parse_ok", "backend"],
        )
        if labels["backend"][0] != "vllm":
            raise RuntimeError(
                f"{path.name} '{labels['backend'][0]}' backend'iyle uretilmis. "
                "Kuru kosu ciktisi ogrenciye ogretmen olamaz."
            )
        sample = pl.read_parquet(
            annotation_sample_path(cfg, role),
            columns=["row_id", "category", "product_title", "title", "text"],
        )
        joined = labels.join(sample, on=["row_id", "category"], how="inner")
        if joined.height != labels.height:
            raise RuntimeError(
                f"{role}: {labels.height} etiketin {joined.height} tanesi metinle eslesti - "
                "row_id kaymis olabilir."
            )
        # row_id ham okumada atanir ve clean -> kcore boyunca korunur: ayni kimlik uzayi.
        kpath = kcore_parquet_path(cfg, role)
        kcore_ids = (
            pl.scan_parquet(kpath).select("row_id").collect()
            if kpath.exists() else pl.DataFrame(schema={"row_id": labels["row_id"].dtype})
        )
        joined = joined.join(
            kcore_ids.with_columns(pl.lit(True).alias("in_kcore")), on="row_id", how="left"
        ).with_columns(pl.col("in_kcore").fill_null(False))
        if joined.height != labels.height:
            raise RuntimeError(f"{role}: 5-core join'i satir cogaltti - row_id benzersiz degil")
        parts.append(joined)

    df = pl.concat(parts, how="vertical_relaxed")
    n_total = df.height

    # --- SIZINTI KURALI: dogrulama satirlari egitime GIREMEZ.
    # Yalnizca hazirlanan kategorilerin dogrulama satirlari: diger kategorinin
    # satiri zaten burada yok, onu "dislanamadi" saymak yanlis alarm olurdu.
    slugs = [cfg.category_slug(r) for r in roles]
    val = _validation_frame(cfg).filter(pl.col("category").is_in(slugs))
    pairs = val.select("category", "row_id")
    df = df.join(pairs, on=["category", "row_id"], how="anti")
    n_excluded_val = n_total - df.height
    if n_excluded_val != val.height:
        raise RuntimeError(
            f"{val.height} dogrulama satirindan {n_excluded_val} tanesi dislanabildi. "
            "Eksik dislama = ogrencinin sinav sorularini egitimde gormesi."
        )

    n_before_parse = df.height
    df = df.filter(pl.col("parse_ok"))
    n_parse_fail = n_before_parse - df.height

    df = df.rename({"purchase_type": "label"})
    df = df.with_columns(
        _stratified_split(df, fraction=float(cfg.get("distill.val_fraction")),
                          seed=int(cfg.get("seed")))
    )
    bundle = df.select(
        "category", "row_id", "sample_frame", "in_kcore", "split", "label",
        pl.Series("text", _texts(df), dtype=pl.String),
    )

    # --- 500 insan satiri: ogrenciyle ayni girdi bicimi, A'nin ve LLM'in etiketi.
    human_src = pl.concat([
        pl.read_parquet(
            annotation_sample_path(cfg, r),
            columns=["row_id", "category", "product_title", "title", "text"],
        )
        for r in roles
    ], how="vertical_relaxed")
    human = val.join(human_src, on=["category", "row_id"], how="left")
    if human["text"].null_count():
        raise RuntimeError("dogrulama satirlarinin bir kismi ornekte bulunamadi")
    human = human.select(
        "val_id", "category", "row_id",
        pl.col("reference").alias("human_label"),
        pl.col("purchase_type").alias("teacher_label"),
        pl.Series("text", _texts(human), dtype=pl.String),
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_parquet(dest)
    human.write_parquet(human_path(cfg))
    log_output(log, dest, n_rows=bundle.height)
    log_output(log, human_path(cfg), n_rows=human.height)

    write_json(
        {
            "code_version": code_version(),
            "categories": [cfg.category_slug(r) for r in roles],
            "n_llm_labels": n_total,
            "n_excluded_validation_rows": n_excluded_val,
            "n_excluded_parse_failures": n_parse_fail,
            "n_rows": bundle.height,
            "split_counts": dict(sorted(bundle.group_by("split").len().iter_rows())),
            "label_counts_train": dict(sorted(
                bundle.filter(pl.col("split") == SPLIT_TRAIN).group_by("label").len().iter_rows()
            )),
            "n_holdout_in_kcore": int(
                bundle.filter((pl.col("split") == SPLIT_HOLDOUT) & pl.col("in_kcore")).height
            ),
            "n_human_rows": human.height,
            "seed": int(cfg.get("seed")),
            "note": (
                "Metin TASIYAN parquet'ler git'e girmez. Dogrulama satirlari "
                "(category,row_id) cifti uzerinden dislandi - sayi raporda."
            ),
        },
        bundle_report_path(cfg),
        log,
    )
    return dest


# ------------------------------------------------------------------ ogrenci
class StubStudent:
    """GPU'suz kuru kosu. Modeli taklit eder, boru hattini ETMEZ.

    Egitim metinlerini ezberler; gormedigi metinde anahtar kelimeye bakar.
    Amaci dogru etiket degil: veri boru hattinin, sizinti kuralinin ve kapi
    mantiginin model indirmeden sinanabilmesi.
    """

    name = "stub"

    def __init__(self, cfg: Config, model_key: str) -> None:
        self.memory: dict[str, str] = {}
        self.model_name = f"stub:{cfg.get(MODEL_KEYS[model_key])}"
        log.warning("STUB ogrenci: tahminler taklit, sonuc ANALIZ EDILEMEZ")

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.memory = dict(zip(texts, labels))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = np.full((len(texts), len(LABELS)), 0.02)
        for i, t in enumerate(texts):
            lab = self.memory.get(t)
            if lab is None:
                low = t.lower()
                lab = "gift_given" if any(w in low for w in ("gift", "present", "birthday")) else "self"
            out[i, LABELS.index(lab)] = 1 - 0.02 * (len(LABELS) - 1)
        return out

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "stub.json").write_text(json.dumps({"model": self.model_name}), encoding="utf-8")

    @classmethod
    def load(cls, cfg: Config, model_key: str, _path: Path) -> StubStudent:
        return cls(cfg, model_key)


def token_batches(lengths: list[int] | np.ndarray, *, max_tokens: int,
                  max_rows: int) -> list[np.ndarray]:
    """Uzunluga gore siralanmis indeks gruplari; her grup `satir x en uzun <= max_tokens`.

    Tek basina butceyi asan satir kendi grubunda gider (kirpma zaten
    `max_length`'te yapildi; burada satir dusurulmez).
    """
    order = np.argsort(np.asarray(lengths), kind="stable")
    out: list[np.ndarray] = []
    start = 0
    for i in range(len(order)):
        n = i - start + 1
        if n > max_rows or (n > 1 and n * int(lengths[order[i]]) > max_tokens):
            out.append(order[start:i])
            start = i
    if start < len(order):
        out.append(order[start:])
    return out


def _training_arguments(fields: set[str], **base) -> dict:
    """TrainingArguments'i kurulu transformers surumune gore kurar.

    Kaggle imajindaki surum sabit degil. v5 `warmup_ratio`'yu `warmup_steps`'e
    (kesirli deger) katladi, `group_by_length`'i `train_sampling_strategy`'ye
    tasidi; eski surumde de `eval_strategy` `evaluation_strategy` idi. Yanlis ad
    Kaggle'da model indirildikten SONRA TypeError ile patlardi.
    """
    kw = {k: v for k, v in base.items() if k not in ("warmup_ratio", "group_by_length",
                                                      "eval_strategy")}
    if "warmup_ratio" in fields:
        kw["warmup_ratio"] = base["warmup_ratio"]
    else:
        kw["warmup_steps"] = base["warmup_ratio"]
    if "train_sampling_strategy" in fields:
        kw["train_sampling_strategy"] = "group_by_length" if base["group_by_length"] else "random"
    else:
        kw["group_by_length"] = base["group_by_length"]
    kw["eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"] = base["eval_strategy"]
    return kw


class HFStudent:
    """transformers ile ince ayar. Yalnizca GPU'lu ortamda (Kaggle) kosar.

    T4 (Turing): bfloat16 yok -> fp16. FlashAttention yok, yani ModernBERT'in
    dolgu kaldirmasi da yok: 32 x 1024 token'lik bir grup T4'u tasirir. Bu yuzden
    `distill.gradient_checkpointing` (sonucu degistirmez, ~%30 yavaslatir).
    fp16'da kayip NaN'a donerse `--no-fp16` ile fp32'ye gecilir.

    TEK GPU. Iki kart gorunurse Trainer DataParallel'e gecer ve etkin grup boyu
    config'teki `batch_size`in iki katina cikar - raporda `effective_batch_size`.
    Kaggle betigi egitimi `CUDA_VISIBLE_DEVICES=0` ile kosar.
    """

    name = "hf"

    def __init__(self, cfg: Config, model_key: str, *, fp16: bool = True) -> None:
        self.cfg = cfg
        self.model_name = str(cfg.get(MODEL_KEYS[model_key]))
        self.fp16 = fp16
        self.tokenizer = None
        self.model = None
        self.train_info: dict = {}

    def _load(self, source: str) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        kwargs = {
            "num_labels": len(LABELS),
            "id2label": dict(enumerate(LABELS)),
            "label2id": {lab: i for i, lab in enumerate(LABELS)},
        }
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                source, attn_implementation="sdpa", **kwargs
            )
        except (ValueError, TypeError, ImportError):
            # SDPA'yi desteklemeyen mimari (ör. DeBERTa-v3): varsayilan dikkat.
            self.model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _encode(self, texts: list[str]):
        return self.tokenizer(
            texts, truncation=True, max_length=int(self.cfg.get("distill.max_length"))
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        import dataclasses  # noqa: PLC0415
        import time  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        set_seed(int(self.cfg.get("seed")))
        self._load(self.model_name)
        ds = Dataset.from_dict({"text": texts, "label": [LABELS.index(x) for x in labels]})
        ds = ds.map(lambda b: self._encode(b["text"]), batched=True, remove_columns=["text"])
        checkpointing = bool(self.cfg.get("distill.gradient_checkpointing"))
        kw = _training_arguments(
            {f.name for f in dataclasses.fields(TrainingArguments)},
            output_dir=str(self.cfg.path("models", "distill", "_ckpt")),
            per_device_train_batch_size=int(self.cfg.get("distill.batch_size")),
            learning_rate=float(self.cfg.get("distill.lr")),
            num_train_epochs=float(self.cfg.get("distill.epochs")),
            warmup_ratio=float(self.cfg.get("distill.warmup_ratio")),
            weight_decay=float(self.cfg.get("distill.weight_decay")),
            fp16=bool(self.fp16 and torch.cuda.is_available()),
            gradient_checkpointing=checkpointing,
            group_by_length=True,
            logging_steps=100,
            # SECIM YOK: ara kontrol noktasi kaydedilmez, degerlendirilmez.
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=int(self.cfg.get("seed")),
        )
        args = TrainingArguments(**kw)
        collator = DataCollatorWithPadding(self.tokenizer)
        try:
            trainer = Trainer(model=self.model, args=args, train_dataset=ds,
                              data_collator=collator, processing_class=self.tokenizer)
        except TypeError:  # transformers < 4.46
            trainer = Trainer(model=self.model, args=args, train_dataset=ds,
                              data_collator=collator, tokenizer=self.tokenizer)
        # Sozlesme disi alanlar egitimden SONRA okunmuyor: 10 dakikalik bir egitimin
        # sonunda AttributeError ile raporu kaybetmemek icin simdi, getattr ile.
        n_gpu = int(getattr(args, "n_gpu", 1) or 1)
        etkin = int(self.cfg.get("distill.batch_size")) * max(1, n_gpu)
        if n_gpu > 1:
            log.warning("%d GPU gorunuyor: DataParallel, etkin grup boyu %d. Tek kart icin "
                        "CUDA_VISIBLE_DEVICES=0.", n_gpu, etkin)
        t0 = time.perf_counter()
        out = trainer.train()
        kayip = float(out.training_loss)
        if not np.isfinite(kayip):
            raise RuntimeError(
                f"egitim kaybi {kayip} - fp16 tasmasi olabilir. `--no-fp16` ile yeniden kosun."
            )
        self.model = trainer.model
        self.train_info = {
            "n_gpu": n_gpu,
            "effective_batch_size": etkin,
            "gradient_checkpointing": checkpointing,
            "fp16_active": bool(kw["fp16"]),
            "training_loss": round(kayip, 5),
            "train_runtime_s": round(time.perf_counter() - t0, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
        }

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch  # noqa: PLC0415

        self.model.eval()
        enc = self._encode(texts)   # dolgu yok: gercek token uzunlugu
        ids = enc["input_ids"]
        batches = token_batches([len(x) for x in ids], max_tokens=PREDICT_MAX_BATCH_TOKENS,
                                max_rows=PREDICT_MAX_BATCH_ROWS)
        keys = list(enc.keys())
        probs = np.empty((len(texts), len(LABELS)), dtype=np.float32)
        with torch.inference_mode():
            for idx in batches:
                grup = self.tokenizer.pad(
                    {k: [enc[k][i] for i in idx] for k in keys}, return_tensors="pt"
                ).to(self.device)
                with torch.autocast(self.device, dtype=torch.float16,
                                    enabled=bool(self.fp16 and self.device == "cuda")):
                    logits = self.model(**grup).logits
                probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))

    @classmethod
    def load(cls, cfg: Config, model_key: str, path: Path, *, fp16: bool = True) -> HFStudent:
        obj = cls(cfg, model_key, fp16=fp16)
        obj._load(str(path))
        return obj


def make_student(cfg: Config, model_key: str, backend: str, *, fp16: bool = True):
    if backend not in BACKENDS:
        raise ValueError(f"bilinmeyen backend: {backend}. Gecerli: {BACKENDS}")
    if model_key not in MODEL_KEYS:
        raise ValueError(f"bilinmeyen model: {model_key}. Gecerli: {tuple(MODEL_KEYS)}")
    return StubStudent(cfg, model_key) if backend == "stub" else HFStudent(cfg, model_key, fp16=fp16)


def load_student(cfg: Config, model_key: str, backend: str, *, fp16: bool = True):
    path = model_dir(cfg, model_key)
    if not path.exists():
        raise FileNotFoundError(f"{path} yok - once `distill train` kosun")
    if backend == "stub":
        return StubStudent.load(cfg, model_key, path)
    return HFStudent.load(cfg, model_key, path, fp16=fp16)


# ------------------------------------------------------------ olcum + kapi
def _axes(ref: list[str], pred: list[str], *, n_boot: int, seed: int) -> dict:
    """C1 ve C1b eksenleri - `analysis.validation` ile AYNI olcum."""
    from ..analysis.validation import binary_axis_measurement  # noqa: PLC0415

    keys = [("all", p) for p in pred]
    return {
        CONDITION_OF[ad]: binary_axis_measurement(
            ref, pred, np.ones(len(ref)), keys, etiketler, n_boot=n_boot, seed=seed
        )
        for ad, etiketler in CONTAMINATION.items()
    }


def _five_class(ref: list[str], pred: list[str]) -> dict:
    from ..analysis.validation import compare  # noqa: PLC0415

    out = compare(pl.Series(ref), pl.Series(pred))
    return {k: out[k] for k in ("n_compared", "accuracy", "macro_f1", "per_class")}


def teacher_c1_on_human(cfg: Config) -> float | None:
    """Ogretmenin 500 satirdaki C1 F1'i - RAPORDAN, kodla uretilmis haliyle."""
    from ..analysis.validation import validation_report_path  # noqa: PLC0415

    path = validation_report_path(cfg, int(cfg.get("validation.n")))
    if not path.exists():
        return None
    try:
        return float(
            read_json(path)["measurements"]["contamination_axes"]["sample"]["C1"]["f1"]["value"]
        )
    except (KeyError, TypeError):
        return None


def fidelity_verdict(cfg: Config, holdout_axes: dict, human_axes: dict) -> dict:
    """Onceden kayitli uc olcut (DECISIONS 2026-09-14). Biri olculemezse INCOMPLETE."""
    fid = cfg.get("distill.fidelity")
    c1 = holdout_axes["C1"]["f1"]["value"]
    c1b = holdout_axes["C1b"]["f1"]["value"]
    ogrenci = human_axes["C1"]["f1"]["value"]
    ogretmen = teacher_c1_on_human(cfg)
    esik_dus = float(fid["max_drop_vs_teacher_on_human"])

    def durum(deger, kosul):
        return None if deger is None else bool(kosul)

    criteria = {
        "1_c1_axis_vs_teacher": {
            "f1": c1, "threshold": float(fid["c1_axis_f1_min"]),
            "passed": durum(c1, c1 is not None and c1 >= float(fid["c1_axis_f1_min"])),
        },
        "2_c1b_axis_vs_teacher": {
            "f1": c1b, "threshold": float(fid["c1b_axis_f1_min"]),
            "passed": durum(c1b, c1b is not None and c1b >= float(fid["c1b_axis_f1_min"])),
        },
        "3_c1_axis_vs_human_not_worse_than_teacher": {
            "student_f1": ogrenci, "teacher_f1": ogretmen, "max_drop": esik_dus,
            "passed": (
                None if ogrenci is None or ogretmen is None
                else bool(ogrenci >= ogretmen - esik_dus)
            ),
        },
    }
    kararlar = [c["passed"] for c in criteria.values()]
    verdict = (
        "INCOMPLETE" if any(k is None for k in kararlar)
        else "PASS" if all(kararlar) else "FAIL"
    )
    return {"criteria": criteria, "verdict": verdict}


def train(
    cfg: Config, *, model_key: str = "base", backend: str = "hf", fp16: bool = True,
    force: bool = False,
) -> dict:
    """Egitir, uc kumede olcer, kapiyi uygular, raporu yazar."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"bilinmeyen model: {model_key}. Gecerli: {tuple(MODEL_KEYS)}")
    dest = report_path(cfg, model_key)
    if should_skip(dest, force, log):
        return read_json(dest)
    if model_key == "fallback":
        # Onceden kayitli kural (DECISIONS 2026-09-14): yedek YALNIZCA ana model
        # kapidan FAIL ile donerse, bir kez. Iki modeli egitip iyisini secmek,
        # kapiyi ayni kumede olcerken model secmek olurdu.
        ana = report_path(cfg, "base")
        karar = read_json(ana).get("verdict") if ana.exists() else None
        if karar != "FAIL":
            raise RuntimeError(
                f"yedek model yalnizca ana model FAIL olursa denenir; ana modelin karari: {karar}"
            )
    if not bundle_path(cfg).exists():
        raise FileNotFoundError("bundle yok - once `distill prepare` kosun")

    bundle = pl.read_parquet(bundle_path(cfg))
    human = pl.read_parquet(human_path(cfg))
    egitim = bundle.filter(pl.col("split") == SPLIT_TRAIN)
    ayrilmis = bundle.filter(pl.col("split") == SPLIT_HOLDOUT)

    student = make_student(cfg, model_key, backend, fp16=fp16)
    log.info("egitim: %s satir | ayrilmis: %s | insan: %s | model: %s",
             f"{egitim.height:,}", f"{ayrilmis.height:,}", human.height, student.model_name)
    student.fit(egitim["text"].to_list(), egitim["label"].to_list())

    def tahmin(texts: list[str]) -> list[str]:
        return [LABELS[i] for i in student.predict_proba(texts).argmax(axis=1)]

    n_boot = int(cfg.get("validation.bootstrap_n"))
    seed = int(cfg.get("seed"))
    p_hold = tahmin(ayrilmis["text"].to_list())
    p_human = tahmin(human["text"].to_list())

    hold_axes = _axes(ayrilmis["label"].to_list(), p_hold, n_boot=n_boot, seed=seed)
    human_axes = _axes(human["human_label"].to_list(), p_human, n_boot=n_boot, seed=seed)
    k_mask = ayrilmis["in_kcore"].to_list()
    k_ref = [r for r, m in zip(ayrilmis["label"].to_list(), k_mask) if m]
    k_pred = [p for p, m in zip(p_hold, k_mask) if m]

    kapi = fidelity_verdict(cfg, hold_axes, human_axes)
    model_path = model_dir(cfg, model_key)
    student.save(model_path)

    report = {
        "meta": {
            "model": student.model_name,
            "model_key": model_key,
            "backend": student.name,
            "fp16": bool(fp16),
            "code_version": code_version(),
            "epochs": cfg.get("distill.epochs"),
            "max_length": cfg.get("distill.max_length"),
            "n_train": egitim.height,
            "n_holdout": ayrilmis.height,
            "n_human": human.height,
            "thresholds_fixed": "2026-09-14, configs/base.yaml -> distill.fidelity (egitimden once)",
            "model_selection": "yok - sabit epoch, ara kontrol noktasi secilmedi",
            "training": getattr(student, "train_info", {}),
        },
        "student_vs_teacher_holdout": {"axes": hold_axes, "five_class": _five_class(ayrilmis["label"].to_list(), p_hold)},
        "student_vs_human": {"axes": human_axes, "five_class": _five_class(human["human_label"].to_list(), p_human)},
        # Dagilim kaymasi: egitim clean korpustan, deney 5-core'da. Kapiya GIRMEZ.
        "student_vs_teacher_holdout_in_kcore": (
            {"n": len(k_ref), "axes": _axes(k_ref, k_pred, n_boot=n_boot, seed=seed)}
            if k_ref else {"n": 0, "skipped": True, "reason": "ayrilmis kumede 5-core satiri yok"}
        ),
        **kapi,
    }
    if student.name == "stub":
        report["verdict"] = "INCOMPLETE"
        report["stub"] = "STUB ogrenci - kapi karari gecersiz, deney bu modelle kosamaz"

    write_json(report, dest, log)
    log.info("SADAKAT KAPISI -> %s", report["verdict"])
    for name, c in kapi["criteria"].items():
        log.info("  %-44s %s", name, {k: v for k, v in c.items()})
    if report["verdict"] == "FAIL" and model_key == "base":
        log.warning("Kapi kacti. Onceden kayitli kural: `--model fallback` ile BIR KEZ deneyin.")
    return report


# -------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("prepare", "train"))
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default="all",
                        help="prepare icin: egitime girecek kategoriler (varsayilan: hepsi)")
    parser.add_argument("--model", default="base", choices=tuple(MODEL_KEYS))
    parser.add_argument("--backend", default="hf", choices=BACKENDS)
    parser.add_argument("--no-fp16", action="store_true", help="fp16 kaybi NaN verirse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from ..config import resolve_roles  # noqa: PLC0415

    cfg = Config.load(args.config)
    if args.step == "prepare":
        prepare(cfg, resolve_roles(cfg, args.category), force=args.force)
    else:
        train(cfg, model_key=args.model, backend=args.backend,
              fp16=not args.no_fp16, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
