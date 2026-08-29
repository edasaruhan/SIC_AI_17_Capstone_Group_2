"""RecBole deney kosucusu (Hafta 6-7).

Girdi : data/processed/recbole/<slug>/<kosul>.inter
Cikti : reports/results/experiment_<slug>_<kosul>_<model>_seed<N>.json

BU MODUL AYRI BIR ORTAMDA KOSAR. RecBole ve torch `.venv-recbole` altinda;
projenin geri kalani `.venv` altinda. Ayirmanin sebebi RecBole'un bagimlilik
agacinin (tensorboard, thop, protobuf...) ana ortami asagi cekme riski. Iki
taraf zaten DOSYA ile konusuyor - `.inter` dosyasi arayuz.

    .venv-recbole/Scripts/python.exe -m gift_contamination.recsys.run_experiment \\
        --category mid --condition C0 --model BPR --epochs 5

BOLMEYI RECBOLE YAPMIYOR - BIZ VERIYORUZ. `benchmark_filename` ile uc ayri
dosya (train/valid/test) veriliyor ve RecBole onlari oldugu gibi kullaniyor.
Kendi bolmesini yapsaydi C1'de satirlar eksildigi icin kullanicinin "son
alimi" degisir ve C0 ile C1 FARKLI test setleri uzerinde karsilastirilirdi -
o noktada olculen sey mudahale olmaktan cikar.

TEST ITEM'I `self` OLMALI (CLAUDE.md 5, kural 2). Kural burada YAPISAL olarak
uygulaniyor: test dosyasina yalnizca `label == self` satirlari yaziliyor.
Validasyon seti kisitlanmiyor - yazili kural yalnizca test icin ve erken
durdurma setini daraltmak ekip karari olur, tek tarafli verilmez.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from ..config import Config
from ..utils.logging import get_logger
from .atomic import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from .conditions import condition_path, condition_report_path

log = get_logger("recsys.run_experiment")

SELF = "self"
BENCHMARK_SUFFIXES = ("train", "valid", "test")

# Config'te metrikler RAPORUN terminolojisiyle yaziliyor (HR@10 literaturde
# boyle geciyor); RecBole ayni seye `Hit` diyor. Config'i RecBole'un ic adina
# gore degistirmek yanlis yon olurdu - esleme burada.
METRIC_ALIASES = {"HitRate": "Hit", "HR": "Hit"}


@contextmanager
def _torch_load_full():
    """RecBole'un checkpoint'ini okuyabilmek icin `weights_only=False`.

    torch 2.6'da `torch.load` varsayilani `weights_only=True` oldu; RecBole
    1.2.0 checkpoint'e model agirliklarinin yaninda kendi `Config` nesnesini
    de gomuyor ve dosya reddediliyor. Bilinen bir surum uyumsuzlugu.

    Burada guvenli olmasinin sebebi dosyanin KAYNAGININ BIZ olmamiz: saniyeler
    once `trainer.fit(saved=True)` ile kendi `saved/` klasorumuze yazildi.
    `weights_only=False`in uyardigi risk GUVENILMEYEN dosyalar icin gecerli.
    Yama daralticidir: yalnizca bu blok boyunca surer.
    """
    import torch  # noqa: PLC0415

    orijinal = torch.load

    def yamali(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orijinal(*args, **kwargs)

    torch.load = yamali
    try:
        yield
    finally:
        torch.load = orijinal


def experiment_path(cfg: Config, role: str, code: str, model: str, seed: int) -> Path:
    slug = cfg.category_slug(role)
    return cfg.path("results", f"experiment_{slug}_{code}_{model}_seed{seed}.json")


def bench_dir(cfg: Config, role: str, code: str) -> Path:
    return cfg.path("processed", "recbole", cfg.category_slug(role), code)


def write_benchmark_files(cfg: Config, role: str, code: str) -> tuple[Path, dict]:
    """Kosulun `.inter` dosyasini RecBole'un bekledigi uc parcaya boler.

    Bolme BIZIM `split` kolonumuzdan geliyor, RecBole yeniden hesaplamiyor.
    Test parcasina yalnizca `label == self` satirlari giriyor.
    """
    import polars as pl  # noqa: PLC0415

    src = condition_path(cfg, role, code)
    if not src.exists():
        raise FileNotFoundError(
            f"{src.name} yok. Once kosulu uretin:\n"
            "  python -m gift_contamination.recsys.conditions "
            f"--category {role} --condition {code}"
        )
    df = pl.read_csv(src, separator="\t")
    df = df.rename({c: c.split(":")[0] for c in df.columns})

    out = bench_dir(cfg, role, code)
    out.mkdir(parents=True, exist_ok=True)
    name = out.name

    parcalar = {
        "train": df.filter(pl.col("split") == SPLIT_TRAIN),
        "valid": df.filter(pl.col("split") == SPLIT_VALID),
        # KURAL: test item'i `self` olmali. Yapisal olarak uygulaniyor -
        # dosyada olmayan satir degerlendirmeye giremez.
        "test": df.filter(
            (pl.col("split") == SPLIT_TEST) & (pl.col("label") == SELF)
        ),
    }
    sayilar = {}
    for suffix, part in parcalar.items():
        dest = out / f"{name}.{suffix}.inter"
        part.select(
            pl.col("user_id").alias("user_id:token"),
            pl.col("item_id").alias("item_id:token"),
            pl.col("timestamp").cast(pl.Float64).alias("timestamp:float"),
        ).write_csv(dest, separator="\t")
        sayilar[suffix] = part.height

    n_test_all = df.filter(pl.col("split") == SPLIT_TEST).height
    meta = {
        **sayilar,
        "n_test_before_self_rule": n_test_all,
        "n_test_dropped_not_self": n_test_all - sayilar["test"],
    }
    log.info(
        "%s/%s: train=%s valid=%s test=%s (self kurali %s satir dusurdu)",
        role, code, f"{sayilar['train']:,}", f"{sayilar['valid']:,}",
        f"{sayilar['test']:,}", f"{meta['n_test_dropped_not_self']:,}",
    )
    return out, meta


def run(
    cfg: Config,
    role: str,
    code: str,
    model: str = "BPR",
    *,
    seed: int | None = None,
    epochs: int | None = None,
) -> dict:
    """Tek kosu: bir kategori, bir kosul, bir model, bir seed."""
    from recbole.config import Config as RecConfig  # noqa: PLC0415
    from recbole.data import create_dataset, data_preparation  # noqa: PLC0415
    from recbole.utils import get_model, get_trainer, init_seed  # noqa: PLC0415

    seed = int(seed if seed is not None else cfg.get("seed"))
    folder, meta = write_benchmark_files(cfg, role, code)

    topk = list(cfg.get("experiment.topk"))
    params = {
        "data_path": str(folder.parent),
        "dataset": folder.name,
        # BOLMEYI BIZ VERIYORUZ. RecBole yeniden hesaplamiyor.
        "benchmark_filename": list(BENCHMARK_SUFFIXES),
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "seed": seed,
        "reproducibility": True,
        "topk": topk,
        "metrics": [
            METRIC_ALIASES.get(m, m) for m in cfg.get("experiment.metrics")
        ],
        "valid_metric": f"NDCG@{topk[0]}",
        # Evren C0'da dondu; RecBole'un ayrica k-core uygulamasi YASAK.
        "user_inter_num_interval": "[0,inf)",
        "item_inter_num_interval": "[0,inf)",
        "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "mode": "full"},
        "show_progress": False,
    }
    if epochs is not None:
        params["epochs"] = int(epochs)

    conf = RecConfig(model=model, config_dict=params)
    init_seed(conf["seed"], conf["reproducibility"])

    dataset = create_dataset(conf)
    train_data, valid_data, test_data = data_preparation(conf, dataset)

    net = get_model(conf["model"])(conf, train_data._dataset).to(conf["device"])
    trainer = get_trainer(conf["MODEL_TYPE"], conf["model"])(conf, net)
    with _torch_load_full():
        # `fit` (skor, sonuc sozlugu) doner - ikisi ayri sey.
        best_score, best_valid = trainer.fit(
            train_data, valid_data, saved=True, show_progress=False
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=True, show_progress=False
        )

    kosul_raporu = json.loads(condition_report_path(cfg, role, code).read_text("utf-8"))
    report = {
        "meta": {
            "category": cfg.category_slug(role),
            "condition": code,
            "model": model,
            "seed": seed,
            "epochs": conf["epochs"],
            "device": str(conf["device"]),
            # Vekil etiketten uretilmis kosu RAPORLANAMAZ.
            "label_source": kosul_raporu.get("label_source"),
            "reportable": kosul_raporu.get("reportable", False),
        },
        "split": meta,
        "n_users": dataset.user_num,
        "n_items": dataset.item_num,
        "best_valid_score": float(best_score) if best_score is not None else None,
        "best_valid": {k: float(v) for k, v in dict(best_valid).items()} if best_valid else None,
        "test": {k: float(v) for k, v in test_result.items()},
    }
    if not report["meta"]["reportable"]:
        log.warning(
            "label_source=%s - DUMAN TESTI. Bu sayilar sonuc tablosuna GIREMEZ.",
            report["meta"]["label_source"],
        )

    dest = experiment_path(cfg, role, code, model, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("yazildi: %s", dest.name)
    for k, v in report["test"].items():
        log.info("  %-16s %.6f", k, v)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--category", default="mid")
    parser.add_argument("--condition", default="C0")
    parser.add_argument("--model", default="BPR")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Duman testi icin dusuk tutun (or. 5). Bos birakilirsa RecBole varsayilani.",
    )
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    run(
        cfg, args.category, args.condition, args.model,
        seed=args.seed, epochs=args.epochs,
    )
    return 0


if __name__ == "__main__":
    # Ayri ortamda kosuyoruz; paket yolu PYTHONPATH'te olmayabilir.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
