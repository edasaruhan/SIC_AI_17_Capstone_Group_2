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

GOLGE URUNLER (C3) ONERILEMEZ. Degerlendirmede (validasyon ve test) `::gift`
sonekli urunlerin skoru -inf yapilir. Test urunu her zaman gercek oldugu icin
isabet saklanmaz; ama maskelenmeseydi golge urun top-K'da bir yer kaplar ve C3'u
yapay olarak cezalandirirdi.

ALINMIS URUN MASKESI BUTUN KOSULLARDA C0'IN. RecBole genel modellerde (BPR)
kullanicinin egitimde gordugu urunleri oneriden cikarir - ama KOSULUN egitiminden.
C1/C4 bir satiri silince o urun maskeden de duser ve top-K'da yer kaplar; C3'te
gercek kimlik golgeye donunce ayni sey olur. Yani kosul yalnizca egitimi degil
DEGERLENDIRMEYI de degistirirdi. Maskeye C0 egitiminde olup kosulun egitiminde
olmayan (kullanici, urun) ciftleri eklenir; C0'in sonucu degismez. Sirali
modellerde RecBole hic maskelemiyor - her kosulda ayni, dokunulmuyor.
DECISIONS 2026-09-14.

KULLANICI BASI CIKTI. Esli bootstrap (Faz 4) ve pazarlama metrikleri (M1/M2)
kullanici basi sayi istiyor; RecBole yalnizca ortalama veriyor. Test
degerlendirmesi SIRASINDA RecBole'un kendi skor tensorunden ayni `topk` cagrisiyla
yakalanir ve kullanici basi ortalamanin RecBole'un toplamina esit oldugu KONTROL
edilir - esit degilse kosu hata verir.
    -> data/processed/recbole/<slug>/peruser/<kosul>_<model>_seed<N>.parquet (git'e girmez)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import Config
from ..utils.io import code_version
from ..utils.logging import get_logger
from .atomic import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID
from .conditions import SHADOW_SUFFIX, condition_path, condition_report_path

log = get_logger("recsys.run_experiment")

SELF = "self"
BENCHMARK_SUFFIXES = ("train", "valid", "test")

# Config'te metrikler RAPORUN terminolojisiyle yaziliyor (HR@10 literaturde
# boyle geciyor); RecBole ayni seye `Hit` diyor. Config'i RecBole'un ic adina
# gore degistirmek yanlis yon olurdu - esleme burada.
METRIC_ALIASES = {"HitRate": "Hit", "HR": "Hit"}
# RecBole ortalamayi bu basamaga yuvarliyor (varsayilan 4). Kullanici basi
# ortalamayla karsilastirma bu hassasiyette yapiliyor.
METRIC_DECIMALS = 6


# ----------------------------------------------------- torch'suz yardimcilar
def shadow_item_ids(tokens) -> np.ndarray:
    """RecBole ic kimlikleri: `::gift` sonekli urun token'lari. 0 = [PAD]."""
    arr = np.asarray(tokens, dtype=object)
    return np.array([i for i, t in enumerate(arr) if str(t).endswith(SHADOW_SUFFIX)], dtype=np.int64)


def per_user_topk_metrics(pos_idx: np.ndarray, pos_len: np.ndarray, ks: list[int]) -> dict:
    """RecBole 1.2.0 `Recall`, `NDCG`, `Hit` formullerinin kullanici basi hali.

    pos_idx: (kullanici, maxK) 0/1 - top-K'nin i. sirasi pozitif mi
    pos_len: (kullanici,) kullanicinin pozitif sayisi
    """
    pos_idx = np.asarray(pos_idx, dtype=np.float64)
    pos_len = np.asarray(pos_len, dtype=np.float64).reshape(-1)
    maxk = pos_idx.shape[1]
    birikim = np.cumsum(pos_idx, axis=1)
    sira = np.arange(1, maxk + 1, dtype=np.float64)
    dcg = np.cumsum(np.where(pos_idx > 0, 1.0 / np.log2(sira + 1), 0.0), axis=1)
    idcg_tam = np.cumsum(1.0 / np.log2(sira + 1))
    idcg_len = np.minimum(pos_len, maxk).astype(int)
    out: dict[str, np.ndarray] = {}
    for k in ks:
        idcg = np.where(idcg_len >= k, idcg_tam[k - 1], idcg_tam[np.maximum(idcg_len, 1) - 1])
        out[f"recall@{k}"] = birikim[:, k - 1] / pos_len
        out[f"ndcg@{k}"] = dcg[:, k - 1] / idcg
        out[f"hit@{k}"] = (birikim[:, k - 1] > 0).astype(np.float64)
    return out


def eval_mask_pairs(c0, cond):
    """C0'in EGITIMINDE olup kosulun egitiminde GERCEK kimligiyle olmayan (kullanici, urun).

    C1/C4'te silinen satirlar, C3'te golgeye donen satirlar. Degerlendirmede bu
    ciftlerin skoru -inf yapilir ki alinmis urun maskesi kosuldan bagimsiz olsun.
    """
    import polars as pl  # noqa: PLC0415

    def ciftler(df):
        return (df.filter(pl.col("split") == SPLIT_TRAIN)
                .select(pl.col("user_id").cast(pl.String), pl.col("item_id").cast(pl.String))
                .unique())

    return ciftler(c0).join(ciftler(cond), on=["user_id", "item_id"], how="anti").sort(
        "user_id", "item_id")


class UserItemMask:
    """Kullanici basi urun kumesi, CSR bicimde (ic kimlikler). Torch'suz."""

    def __init__(self, users: np.ndarray, items: np.ndarray, n_users: int) -> None:
        sira = np.lexsort((items, users))
        self.items = np.asarray(items, dtype=np.int64)[sira]
        adet = np.bincount(np.asarray(users, dtype=np.int64), minlength=n_users)
        self.indptr = np.concatenate([[0], np.cumsum(adet)]).astype(np.int64)
        self.n_pairs = len(self.items)

    def lookup(self, batch_users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(satir, urun): satir = batch icindeki konum."""
        u = np.asarray(batch_users, dtype=np.int64)
        bas, adet = self.indptr[u], self.indptr[u + 1] - self.indptr[u]
        toplam = int(adet.sum())
        satir = np.repeat(np.arange(len(u), dtype=np.int64), adet)
        ofset = np.arange(toplam, dtype=np.int64) - np.repeat(np.cumsum(adet) - adet, adet)
        return satir, self.items[np.repeat(bas, adet) + ofset]


def _model_loss_type(model: str) -> str | None:
    """Modelin RecBole varsayilan `loss_type`'i (kendi properties dosyasindan)."""
    import recbole  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    yol = Path(recbole.__file__).parent / "properties" / "model" / f"{model}.yaml"
    if not yol.exists():
        return None
    return (yaml.safe_load(yol.read_text(encoding="utf-8")) or {}).get("loss_type")


def peruser_path(cfg: Config, role: str, code: str, model: str, seed: int) -> Path:
    return cfg.path("processed", "recbole", cfg.category_slug(role), "peruser",
                    f"{code}_{model}_seed{seed}.parquet")


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


class _EvalHook:
    """RecBole'un degerlendirme toplayicisina iki is ekler (validasyon + test):

    1. golge urunlerin ve C0 maskesinin (`UserItemMask`) skorunu RecBole top-K'yi
       almadan ONCE -inf yapar;
    2. `active` iken kullanici basi top-K'yi ve pozitif konumlarini yakalar -
       RecBole'un kullandigi TENSORUN AYNISINDAN, ayni `torch.topk` cagrisiyla.
    """

    def __init__(self, trainer, dataset, shadow_idx: np.ndarray, max_k: int,
                 extra_mask: UserItemMask | None = None) -> None:
        import torch  # noqa: PLC0415

        self.active = False
        self.users: list[np.ndarray] = []
        self.topk: list[np.ndarray] = []
        self.pos_idx: list[np.ndarray] = []
        self.pos_len: list[np.ndarray] = []
        uid_field = dataset.uid_field
        golge = torch.as_tensor(shadow_idx, dtype=torch.long)
        orijinal = trainer.eval_collector.eval_batch_collect

        def topla(scores, interaction, positive_u, positive_i):
            if len(golge):
                scores[:, golge.to(scores.device)] = -np.inf
            if extra_mask is not None and extra_mask.n_pairs:
                satir, urun = extra_mask.lookup(interaction[uid_field].cpu().numpy())
                if len(satir):
                    scores[torch.as_tensor(satir, device=scores.device),
                           torch.as_tensor(urun, device=scores.device)] = -np.inf
            orijinal(scores, interaction, positive_u, positive_i)
            if not self.active:
                return
            _, topk_idx = torch.topk(scores, max_k, dim=-1)
            pos = torch.zeros_like(scores, dtype=torch.int)
            pos[positive_u, positive_i] = 1
            self.pos_len.append(pos.sum(dim=1).cpu().numpy())
            self.pos_idx.append(torch.gather(pos, dim=1, index=topk_idx).cpu().numpy())
            self.topk.append(topk_idx.cpu().numpy())
            self.users.append(interaction[uid_field].cpu().numpy())

        trainer.eval_collector.eval_batch_collect = topla


def experiment_path(cfg: Config, role: str, code: str, model: str, seed: int) -> Path:
    slug = cfg.category_slug(role)
    return cfg.path("results", f"experiment_{slug}_{code}_{model}_seed{seed}.json")


def checkpoint_dir(cfg: Config, role: str, code: str, model: str, seed: int) -> Path:
    """Kosu basina checkpoint klasoru (RecBole'un varsayilani gibi calisma dizinine gore).

    RecBole dosyayi `<model>-<Ay-Gun-Yil_SS-DD-ss>.pth` diye adlandiriyor - SANIYE
    cozunurluklu. Kaggle'da iki kartta ayni saniyede kurulan ayni model AYNI dosyaya
    yazar: biri otekinin agirligini yukler (ayni urun sayisinda sessizce), sonra
    dosyayi siler ve oteki coker. Klasor kosunun kimligini tasiyinca carpisma olmaz.
    """
    return Path("saved") / experiment_path(cfg, role, code, model, seed).stem


def bench_dir(cfg: Config, role: str, code: str, *, sequential: bool = False) -> Path:
    # Sirali modellerin dosya bicimi farkli (satir basina gecmis) - ayri klasor.
    ad = f"{code}_seq" if sequential else code
    return cfg.path("processed", "recbole", cfg.category_slug(role), ad)


def sequential_parts(df, *, max_len: int) -> tuple[dict, dict]:
    """Sirali modeller icin ONCEDEN GENISLETILMIS parcalar (RecBole benchmark sozlesmesi).

    RecBole `benchmark_filename` verilince sekanslari kendisi genisletmiyor; her
    satir `item_id_list` (onceki urunler, en fazla `max_len`, en eskisi basta)
    tasimali. RecBole'un kendi genisletmesiyle ayni kural:
      egitim : her egitim urunu, ONCEKI egitim urunleriyle (ilk urunun gecmisi yok -> satir yok)
      valid  : gecmis = kullanicinin butun egitim urunleri
      test   : gecmis = egitim + valid urunleri; yalnizca `label == self`
    `df` dosya sirasinda olmali (kullanici icinde kronolojik, esitlikte row_id).

    Valid satirinin gecmisi BOS olabilir (C1/C1b kullanicinin butun egitim
    satirlarini silebilir). SASRec bos gecmisi isleyemez; o satir yalnizca erken
    durdurma setinden duser ve sayilir. Test satirinin gecmisi valid urununu
    icerdigi icin HICBIR ZAMAN bos degil - test ciftleri kosullar arasinda ayni kalir.
    """
    import polars as pl  # noqa: PLC0415

    d = df.with_row_index("_ord")

    def gecmis(filtre) -> pl.DataFrame:
        return (d.filter(filtre).sort("_ord")
                .group_by("user_id", maintain_order=True)
                .agg(pl.col("item_id").cast(pl.String).alias("_h")))

    acik = (
        gecmis(pl.col("split") == SPLIT_TRAIN)
        .join(d.filter(pl.col("split") == SPLIT_TRAIN).sort("_ord")
              .group_by("user_id", maintain_order=True).agg(pl.col("timestamp").alias("_t")),
              on="user_id")
        .with_columns(pl.int_ranges(1, pl.col("_h").list.len()).alias("_k"))
        .explode("_k", empty_as_null=False)
    )
    if acik.height:
        egitim = acik.select(
            "user_id",
            pl.col("_h").list.slice(
                pl.max_horizontal(pl.col("_k") - max_len, pl.lit(0)),
                pl.min_horizontal(pl.col("_k"), pl.lit(max_len)),
            ).list.join(" ").alias("item_id_list"),
            pl.col("_h").list.get(pl.col("_k")).alias("item_id"),
            pl.col("_t").list.get(pl.col("_k")).alias("timestamp"),
        )
    else:
        # Bos cercevede polars dilimin tipini `null` cikarip `join`de patliyor.
        egitim = pl.DataFrame(schema={"user_id": d.schema["user_id"], "item_id_list": pl.String,
                                      "item_id": pl.String, "timestamp": d.schema["timestamp"]})

    def hedefli(filtre_hedef, filtre_gecmis) -> pl.DataFrame:
        return (
            d.filter(filtre_hedef).sort("_ord")
            .join(gecmis(filtre_gecmis), on="user_id", how="left")
            .select(
                "user_id",
                pl.col("_h").fill_null([]).list.tail(max_len).list.join(" ").alias("item_id_list"),
                pl.col("item_id").cast(pl.String),
                "timestamp",
            )
        )

    valid = hedefli(pl.col("split") == SPLIT_VALID, pl.col("split") == SPLIT_TRAIN)
    n_valid_bos = valid.filter(pl.col("item_id_list") == "").height
    valid = valid.filter(pl.col("item_id_list") != "")
    test = hedefli((pl.col("split") == SPLIT_TEST) & (pl.col("label") == SELF),
                   pl.col("split").is_in([SPLIT_TRAIN, SPLIT_VALID]))
    if test.filter(pl.col("item_id_list") == "").height:
        raise RuntimeError("bos gecmisli test satiri - valid satiri kaybolmus olabilir")
    return ({"train": egitim, "valid": valid, "test": test},
            {"n_valid_dropped_empty_history": n_valid_bos, "max_item_list_length": max_len})


def hash_test_pairs(test_part) -> str:
    """Degerlendirilen (kullanici, urun) ciftlerinin sira bagimsiz ozeti.

    Kapi 2, olcut 1: bu ozet butun kosullarda (ve seed'lerde) AYNI olmali. Kosu
    aninda, RecBole'a verilen test dosyasinin kendisinden hesaplaniyor.
    """
    import hashlib  # noqa: PLC0415

    import polars as pl  # noqa: PLC0415

    ciftler = (test_part.select(pl.col("user_id").cast(pl.String), pl.col("item_id").cast(pl.String))
               .sort("user_id", "item_id"))
    h = hashlib.sha256()
    for u, i in ciftler.iter_rows():
        h.update(f"{u}\t{i}\n".encode())
    return h.hexdigest()


def read_condition(cfg: Config, role: str, code: str):
    """Kosulun `.inter` dosyasi; kimlikler METIN (sayiya benzeyen ASIN bozulmasin)."""
    import polars as pl  # noqa: PLC0415

    src = condition_path(cfg, role, code)
    if not src.exists():
        raise FileNotFoundError(
            f"{src.name} yok. Once kosulu uretin:\n"
            "  python -m gift_contamination.recsys.conditions "
            f"--category {role} --condition {code}"
        )
    df = pl.read_csv(src, separator="\t", schema_overrides={"item_id:token": pl.String,
                                                            "user_id:token": pl.String})
    return df.rename({c: c.split(":")[0] for c in df.columns})


def c0_eval_mask(cfg: Config, role: str, code: str, dataset) -> tuple[UserItemMask | None, dict]:
    """Genel modeller icin C0 alinmis urun maskesinin EKSIK kalan kismi (ic kimliklerle)."""
    import polars as pl  # noqa: PLC0415

    if code == "C0":
        return None, {"eval_history_mask": "c0_train", "n_extra_mask_pairs": 0}
    kosul = read_condition(cfg, role, code)
    ciftler = eval_mask_pairs(read_condition(cfg, role, "C0"), kosul)
    # Maskelenen cift hicbir zaman bir POZITIF olamaz (kullanici+urun tekil).
    # Olursa isabet saklanir ve metrik sessizce duser - kosu durur.
    hedef = kosul.filter(pl.col("split") != SPLIT_TRAIN).select(
        pl.col("user_id").cast(pl.String), pl.col("item_id").cast(pl.String))
    cakisan = ciftler.join(hedef, on=["user_id", "item_id"], how="semi").height
    if cakisan:
        raise RuntimeError(f"{cakisan} maske cifti valid/test pozitifiyle cakisiyor")
    u_map = dataset.field2token_id[dataset.uid_field]
    i_map = dataset.field2token_id[dataset.iid_field]
    # Hicbir dosyada gecmeyen urunun skor sutunu yok - maskelenecek bir sey yok.
    ciftler = ciftler.filter(pl.col("item_id").is_in(list(i_map)) & pl.col("user_id").is_in(list(u_map)))
    users = np.array([u_map[u] for u in ciftler["user_id"].to_list()], dtype=np.int64)
    items = np.array([i_map[i] for i in ciftler["item_id"].to_list()], dtype=np.int64)
    return UserItemMask(users, items, dataset.user_num), {
        "eval_history_mask": "c0_train", "n_extra_mask_pairs": int(len(items))}


def write_benchmark_files(cfg: Config, role: str, code: str, *,
                          sequential: bool = False) -> tuple[Path, dict]:
    """Kosulun `.inter` dosyasini RecBole'un bekledigi uc parcaya boler.

    Bolme BIZIM `split` kolonumuzdan geliyor, RecBole yeniden hesaplamiyor.
    Test parcasina yalnizca `label == self` satirlari giriyor.
    """
    import polars as pl  # noqa: PLC0415

    df = read_condition(cfg, role, code)
    out = bench_dir(cfg, role, code, sequential=sequential)
    out.mkdir(parents=True, exist_ok=True)
    name = out.name

    ek: dict = {}
    if sequential:
        parcalar, ek = sequential_parts(df, max_len=int(cfg.get("experiment.max_item_list_length")))
        kolonlar = [
            pl.col("user_id").alias("user_id:token"),
            pl.col("item_id_list").alias("item_id_list:token_seq"),
            pl.col("item_id").alias("item_id:token"),
            pl.col("timestamp").cast(pl.Float64).alias("timestamp:float"),
        ]
    else:
        parcalar = {
            "train": df.filter(pl.col("split") == SPLIT_TRAIN),
            "valid": df.filter(pl.col("split") == SPLIT_VALID),
            # KURAL: test item'i `self` olmali. Yapisal olarak uygulaniyor -
            # dosyada olmayan satir degerlendirmeye giremez.
            "test": df.filter(
                (pl.col("split") == SPLIT_TEST) & (pl.col("label") == SELF)
            ),
        }
        kolonlar = [
            pl.col("user_id").alias("user_id:token"),
            pl.col("item_id").alias("item_id:token"),
            pl.col("timestamp").cast(pl.Float64).alias("timestamp:float"),
        ]
    sayilar = {}
    for suffix, part in parcalar.items():
        dest = out / f"{name}.{suffix}.inter"
        part.select(kolonlar).write_csv(dest, separator="\t")
        sayilar[suffix] = part.height

    n_test_all = df.filter(pl.col("split") == SPLIT_TEST).height
    meta = {
        **sayilar,
        "n_test_before_self_rule": n_test_all,
        "n_test_dropped_not_self": n_test_all - sayilar["test"],
        "test_pairs_sha256": hash_test_pairs(parcalar["test"]),
        **ek,
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
    force: bool = False,
) -> dict:
    """Tek kosu: bir kategori, bir kosul, bir model, bir seed."""
    import time  # noqa: PLC0415

    seed = int(seed if seed is not None else cfg.get("seed"))
    dest = experiment_path(cfg, role, code, model, seed)
    if dest.exists() and peruser_path(cfg, role, code, model, seed).exists() and not force:
        # Kaggle oturumu kesilince matris bastan kosmasin (CLAUDE.md 7: idempotent).
        log.info("atlaniyor (rapor ve kullanici basi dosya var, --force ile ezilir): %s", dest.name)
        return json.loads(dest.read_text(encoding="utf-8"))

    from recbole.config import Config as RecConfig  # noqa: PLC0415
    from recbole.data import create_dataset, data_preparation  # noqa: PLC0415
    from recbole.utils import ModelType, get_model, get_trainer, init_seed  # noqa: PLC0415

    sirali = get_model(model).type == ModelType.SEQUENTIAL
    folder, meta = write_benchmark_files(cfg, role, code, sequential=sirali)

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
        "metric_decimal_place": METRIC_DECIMALS,
        "show_progress": False,
        # RecBole `gpu_id`i CUDA_VISIBLE_DEVICES'a YAZIYOR (varsayilan '0'). Kaggle'da
        # ikinci karta verilen kosu boylece sessizce birinci karta dusuyordu.
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        # Paralel kosular ayni checkpoint dosyasini paylasmasin (bkz. `checkpoint_dir`).
        "checkpoint_dir": str(checkpoint_dir(cfg, role, code, model, seed)),
    }
    if epochs is not None:
        params["epochs"] = int(epochs)
    if sirali:
        # Satir basina gecmis dosyada hazir; ayni kimlik uzayina esleniyor.
        params["load_col"] = {"inter": ["user_id", "item_id_list", "item_id", "timestamp"]}
        params["alias_of_item_id"] = ["item_id_list"]
        params["MAX_ITEM_LIST_LENGTH"] = int(cfg.get("experiment.max_item_list_length"))
    if _model_loss_type(model) == "CE":
        # SASRec varsayilani CE: butun urunler uzerinde softmax, negatif ornekleme
        # YOK. RecBole'un genel varsayilani (uniform, 1 negatif) CE ile birlikte
        # verilince kurulumda hata veriyor - SASRec bu yuzden hic kosmamisti.
        params["train_neg_sample_args"] = None

    conf = RecConfig(model=model, config_dict=params)
    init_seed(conf["seed"], conf["reproducibility"])

    dataset = create_dataset(conf)
    # Degerlendirme grubu SKOR HUCRESIYLE sinirli (GPU bellegi). RecBole'un varsayilani
    # (4.096) genel modellerde urun sayisina bolunuyor: urun sayisi bunu asinca tam
    # siralama kullanici basina BIR grup kosuyordu. Sonucu degistirmez.
    hucre = int(cfg.get("experiment.eval_score_cells"))
    conf["eval_batch_size"] = max(1, hucre // dataset.item_num) if sirali else max(hucre, dataset.item_num)
    train_data, valid_data, test_data = data_preparation(conf, dataset)

    net = get_model(conf["model"])(conf, train_data._dataset).to(conf["device"])
    trainer = get_trainer(conf["MODEL_TYPE"], conf["model"])(conf, net)
    golge = shadow_item_ids(dataset.field2id_token[dataset.iid_field])
    if len(golge):
        log.info("%d golge urun degerlendirmede maskeleniyor", len(golge))
    if sirali:
        # RecBole sirali modellerde alinmis urunu hic maskelemiyor; her kosulda ayni.
        c0_maske, maske_meta = None, {"eval_history_mask": "none_sequential", "n_extra_mask_pairs": 0}
    else:
        c0_maske, maske_meta = c0_eval_mask(cfg, role, code, dataset)
        if c0_maske is not None:
            log.info("C0 alinmis urun maskesine %s cift eklendi", f"{c0_maske.n_pairs:,}")
    kanca = _EvalHook(trainer, dataset, golge, max(topk), c0_maske)
    with _torch_load_full():
        t0 = time.time()
        # `fit` (skor, sonuc sozlugu) doner - ikisi ayri sey.
        best_score, best_valid = trainer.fit(
            train_data, valid_data, saved=True, show_progress=False
        )
        t1 = time.time()
        kanca.active = True
        test_result = trainer.evaluate(
            test_data, load_best_model=True, show_progress=False
        )
        kanca.active = False
        sure = {"fit": round(t1 - t0, 1), "test": round(time.time() - t1, 1)}

    peruser, tutarlilik = _write_peruser(cfg, role, code, model, seed, dataset, kanca,
                                         topk, dict(test_result), golge, c0_maske)

    kosul_raporu = json.loads(condition_report_path(cfg, role, code).read_text("utf-8"))
    report = {
        "meta": {
            "category": cfg.category_slug(role),
            "condition": code,
            "model": model,
            "seed": seed,
            "epochs": conf["epochs"],
            "device": str(conf["device"]),
            "eval_batch_size": int(conf["eval_batch_size"]),
            # Zaman sondasi (Faz 4.1) bu sayilarla matrisi butceliyor.
            "runtime_seconds": sure,
            "code_version": code_version(),
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
        "n_shadow_items_masked": int(len(golge)),
        **maske_meta,
        # Mutlak yol DEGIL - commit edilen JSON isletim sistemi kullanici adini sizdirmasin.
        "peruser_file": peruser.name,
        "peruser_consistency": tutarlilik,
    }
    if not report["meta"]["reportable"]:
        log.warning(
            "label_source=%s - DUMAN TESTI. Bu sayilar sonuc tablosuna GIREMEZ.",
            report["meta"]["label_source"],
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("yazildi: %s", dest.name)
    # Checkpoint yeniden kullanilmiyor: M1/M2 kullanici basi top-K listelerinden
    # hesaplaniyor. Matris ~70 kosu x ~100-300 MB; Kaggle'in 20 GB diskini doldururdu.
    Path(trainer.saved_model_file).unlink(missing_ok=True)
    try:
        Path(trainer.saved_model_file).parent.rmdir()
    except OSError:
        pass  # bos degilse (ayni kosu yeniden basladiysa) dokunulmaz
    for k, v in report["test"].items():
        log.info("  %-16s %.6f", k, v)
    return report


def _write_peruser(cfg: Config, role: str, code: str, model: str, seed: int, dataset,
                   kanca: _EvalHook, topk: list[int], test_result: dict,
                   golge: np.ndarray, c0_maske: UserItemMask | None = None) -> tuple[Path, dict]:
    """Kullanici basi metrik + top-K listesi; ortalamasi RecBole'unkiyle ESIT olmali."""
    import polars as pl  # noqa: PLC0415

    if not kanca.users:
        raise RuntimeError("test degerlendirmesinde hic kullanici yakalanmadi")
    users = np.concatenate(kanca.users)
    topk_idx = np.concatenate(kanca.topk)
    metrikler = per_user_topk_metrics(np.concatenate(kanca.pos_idx),
                                      np.concatenate(kanca.pos_len), topk)
    if len(np.unique(users)) != len(users):
        raise RuntimeError("bir kullanici testte birden fazla kez degerlendirildi")
    if len(golge) and np.isin(topk_idx, golge).any():
        raise RuntimeError("golge urun top-K listesine girdi - maske calismiyor")
    if c0_maske is not None and c0_maske.n_pairs:
        satir, urun = c0_maske.lookup(users)
        if (topk_idx[satir] == urun[:, None]).any():
            raise RuntimeError("C0 maskesindeki urun top-K listesine girdi - maske calismiyor")

    # Kullanici basi ortalama == RecBole toplami. Degilse esli bootstrap RecBole'un
    # raporladigindan FARKLI bir sayinin guven araligini verirdi.
    tol = 10 ** (-METRIC_DECIMALS) * 0.5 + 1e-12
    farklar = {}
    for ad, deger in metrikler.items():
        if ad in test_result:
            farklar[ad] = abs(round(float(np.mean(deger)), METRIC_DECIMALS) - float(test_result[ad]))
    kotu = {k: v for k, v in farklar.items() if v > tol}
    if not farklar or kotu:
        raise RuntimeError(f"kullanici basi ortalama RecBole toplamiyla uyusmuyor: {kotu or 'ortak metrik yok'}")

    tokens = dataset.field2id_token[dataset.iid_field]
    df = pl.DataFrame({
        "user_id": [str(u) for u in dataset.id2token(dataset.uid_field, users)],
        **{ad: pl.Series(ad, v, dtype=pl.Float64) for ad, v in metrikler.items()},
        "topk_items": [[str(tokens[i]) for i in row] for row in topk_idx],
    })
    dest = peruser_path(cfg, role, code, model, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    log.info("yazildi: %s (%s kullanici)", dest.name, f"{df.height:,}")
    return dest, {"n_users": df.height, "metrics_checked": sorted(farklar),
                  "max_abs_diff": max(farklar.values()), "tolerance": tol}


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
    parser.add_argument("--force", action="store_true",
                        help="Rapor ve kullanici basi dosya varsa bile yeniden kos.")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    run(
        cfg, args.category, args.condition, args.model,
        seed=args.seed, epochs=args.epochs, force=args.force,
    )
    return 0


if __name__ == "__main__":
    # Ayri ortamda kosuyoruz; paket yolu PYTHONPATH'te olmayabilir.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
