"""Kaggle notebook'una YAPISTIRILACAK tek hucre - Hafta 6-7: RecBole deney matrisi.

`kaggle_distill.py` ile ayni desen (neden .py: diff'lenir, catismaz).

NE YAPAR:
  1. RecBole'u AYRI bir klasore kurar (numpy 1.26 + yerelde dogrulanmis pinler).
     Kaggle imajinin numpy 2 / pandas 3 yigini RecBole'u ilk satirda cokertiyor;
     torch imajdan kullaniliyor. Ayni tarif yerelde torch 2.14 + numpy 2.5 olan bir
     ortamda denendi: 24 duman kosusu `.venv-recbole` ile BIREBIR ayni sayilari verdi.
  2. Kosullari (C0 C1 C4 C1b C4b C3) atomic dosyadan URETIR - yerelde uretilenle ayni
     kod, ayni seed; yuklenecek dosya 12 degil 4.
  3. Kosulari iki T4'e dagitir: her kart bir sonraki isi kuyruktan alir.
     PLAN = "probe"  -> Toys x C0 x {SASRec, BPR} x ilk seed (ZAMAN SONDASI, Faz 4.1)
     PLAN = "matrix" -> config'teki matris, oncelik sirasiyla (asagida)
  Her kosu biter bitmez rapor + kullanici basi dosya out/'a kopyalanir; oturum
  kesilse bile biten kosular kaybolmaz.

ON KOSULLAR (Kaggle arayuzunde, elle):
  1. Settings > Accelerator = **GPU T4 x2**   ·   Settings > Internet = **On** (pip)
  2. Add Input > Datasets > **OZEL (private)** dataset. Yerelde, GERCEK etiketle:
        python -m gift_contamination.recsys.atomic --category high --labels distilled --force
        python -m gift_contamination.recsys.atomic --category low  --labels distilled --force
     Yuklenecekler (klasor yapisi korunursa en iyisi; duzlesirse split dosyasini
     `<slug>_split.json` diye yeniden adlandirin):
        data/processed/recbole/Toys_and_Games/Toys_and_Games.inter             + split.json
        data/processed/recbole/Grocery_and_Gourmet_Food/Grocery_and_Gourmet_Food.inter + split.json
     `.inter` review metni tasimaz ama kullanici kimligi tasir - yine OZEL.
  3. (istege bagli) Onceki oturumun out/ klasorunu ikinci bir dataset olarak ekleyin:
     icindeki experiment_*.json + peruser/*.parquet yerlestirilir ve o kosular ATLANIR.

KOSU BITINCE: /kaggle/working/out/ klasorunu indirin ve depoya yerlestirin:
     experiment_*.json, condition_*.json      -> reports/results/
     peruser/<slug>/*.parquet                 -> data/processed/recbole/<slug>/peruser/
     sonra yerelde: python -m gift_contamination.analysis.experiment_stats

ONCELIK (plan: kesme sirasi M3 -> BPR'nin ek seed'leri -> Grocery'de C3):
  1. ilk seed, butun kosullar (Grocery C3 haric)   2. SASRec ek seed'ler
  3. BPR ek seed'ler                               4. Grocery C3
"""

# ---------------------------------------------------------------- ayarlar
DATASET_PATH = "/kaggle/input/recsys-experiment"
PREVIOUS_OUT_PATH = None      # or. "/kaggle/input/recsys-experiment-out1" - biten kosular atlanir
REPO = "https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_2.git"
PLAN = "probe"                # "probe" (once bu!) · "matrix"
CATEGORIES = None             # None -> experiment.categories  · or. ["high"]
MODELS = None                 # None -> experiment.models      · or. ["SASRec"]
CONDITIONS = None             # None -> experiment.conditions
SEEDS = None                  # None -> seeds
MAX_HOURS = 11.0              # bu saatten sonra YENI kosu baslatilmaz (Kaggle 12 sa keser)
FORCE = False                 # True: biten kosu da yeniden kosar

# ------------------------------------------------------------------ kurulum
import json  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


def sh(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


T_BASLA = time.time()
WORK = Path("/kaggle/working")
CLONE = WORK / "SIC_AI_17_Capstone_Group_2"
REPO_DIR = CLONE / "repo"
OUT = WORK / "out"
PKGS = WORK / "recbole_pkgs"
OUT.mkdir(exist_ok=True)

# Depo HER KOSUDA guncellenir (kaggle_annotate.py'deki 2026-08-28 hatasi).
if REPO_DIR.exists():
    sh(f"git -C {CLONE} fetch --depth 1 origin main")
    sh(f"git -C {CLONE} reset --hard origin/main")
else:
    sh(f"git clone --depth 1 {REPO} {CLONE}")
sh(f"git -C {CLONE} --no-pager log -1 --format='kod surumu: %h %s'")

# Pinler `.venv-recbole`un `pip freeze`inden (yerelde butun testler ve duman kosulari
# bununla gecti). torch LISTEDE YOK: imajinki kullanilir. --no-deps: pip torch'u
# yeniden indirmeye kalkmasin.
PINLER = [
    "recbole==1.2.0", "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2", "pandas==2.3.3",
    "joblib==1.5.3", "threadpoolctl==3.6.0", "python-dateutil==2.9.0.post0", "pytz==2026.3.post1",
    "tzdata==2026.3", "six==1.17.0", "colorlog==4.7.2", "colorama==0.4.4",
    "thop==0.1.1.post2209072238", "texttable==1.7.0", "tabulate==0.10.0", "plotly==7.0.0",
    "narwhals==2.25.0", "kmeans-pytorch==0.3", "tensorboard==2.21.0",
    "tensorboard-data-server==0.7.2", "absl-py==2.5.0", "grpcio==1.83.1", "protobuf==7.36.0",
    "Markdown==3.10.3", "Werkzeug==3.1.8", "MarkupSafe==3.0.3", "polars==1.44.1",
    "polars-runtime-32==1.44.1", "python-dotenv==1.2.3", "PyYAML==6.0.3", "tqdm==4.70.0",
]


def _ortam(gpu: str) -> dict:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        # Kurulan klasor ONCE: imajin numpy 2'si golgelenir.
        "PYTHONPATH": os.pathsep.join([str(PKGS), str(REPO_DIR / "src")]),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _dogrula() -> bool:
    kod = ("import numpy, pandas, torch, recbole, polars; "
           "assert numpy.__version__.startswith('1.26'), numpy.__version__; "
           "print('numpy', numpy.__version__, '| torch', torch.__version__, '| recbole', recbole.__version__, "
           "'| cuda', torch.cuda.is_available())")
    r = subprocess.run([sys.executable, "-c", kod], env=_ortam("0"), capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-1500:])
    return r.returncode == 0


if not _dogrula():
    sh(f"{sys.executable} -m pip install -q --no-deps --target {PKGS} {' '.join(PINLER)}")
    if not _dogrula():
        raise SystemExit("RecBole ortami kurulamadi - yukaridaki hata ciktisina bakin.")

import yaml  # noqa: E402  (imajda var; yalnizca config okumak icin)

CFG = yaml.safe_load((REPO_DIR / "configs" / "base.yaml").read_text(encoding="utf-8"))
SLUG = {rol: ad.replace("raw_review_", "") for rol, ad in CFG["dataset"]["categories"].items()}
kategoriler = CATEGORIES or CFG["experiment"]["categories"]
modeller = MODELS or CFG["experiment"]["models"]
kosullar = CONDITIONS or CFG["experiment"]["conditions"]
seedler = [int(s) for s in (SEEDS or CFG["seeds"])]

# ------------------------------------------------------- girdileri yerlestir
src = Path(DATASET_PATH)
for rol in kategoriler:
    slug = SLUG[rol]
    hedef = REPO_DIR / "data" / "processed" / "recbole" / slug
    hedef.mkdir(parents=True, exist_ok=True)
    inter = sorted(src.rglob(f"{slug}.inter"))
    if not inter:
        raise SystemExit(f"{slug}.inter dataset'te yok: {src}. Yerelde `recsys.atomic --labels distilled` "
                         "kosup yukleyin; DATASET_PATH'i sag paneldeki TAM yolla degistirin.")
    split = [p for p in (inter[0].parent / "split.json", *src.rglob(f"{slug}_split.json")) if p.exists()]
    if not split:
        raise SystemExit(f"{slug} icin split.json yok (ayni klasorde ya da `{slug}_split.json` adiyla).")
    shutil.copy(inter[0], hedef / inter[0].name)
    shutil.copy(split[0], hedef / "split.json")
    etiket = json.loads((hedef / "split.json").read_text(encoding="utf-8")).get("label_source")
    print(f"{slug}: {inter[0].stat().st_size / 1e6:.0f} MB · etiket kaynagi {etiket}")
    if etiket != "distilled":
        print(f"!! UYARI: {slug} etiket kaynagi '{etiket}' - bu kosular DUMAN TESTIDIR, raporlanamaz.")

if PREVIOUS_OUT_PATH:
    onceki = Path(PREVIOUS_OUT_PATH)
    for f in onceki.rglob("experiment_*.json"):
        shutil.copy(f, REPO_DIR / "reports" / "results" / f.name)
        shutil.copy(f, OUT / f.name)
    for f in onceki.rglob("*.parquet"):
        slug = f.parent.name
        d = REPO_DIR / "data" / "processed" / "recbole" / slug / "peruser"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, d / f.name)
        (OUT / "peruser" / slug).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, OUT / "peruser" / slug / f.name)
    print("onceki oturumdan yerlestirildi:", len(list(onceki.rglob("experiment_*.json"))), "rapor")


# ----------------------------------------------------------- alt surec kosucu
def kos(args: list[str], gpu: str, etiket: str) -> int:
    """Alt sureci kosar; ciktisini hucreye (onekli) ve out/logs/<etiket>.log'a yazar."""
    (OUT / "logs").mkdir(exist_ok=True)
    with open(OUT / "logs" / f"{etiket}.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", *args], cwd=REPO_DIR, env=_ortam(gpu),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
        )
        for satir in proc.stdout:
            log.write(satir)
            # RecBole'un pandas uyarilari hucreyi bogmasin; log dosyasinda duruyor.
            if "Warning" not in satir and "inplace" not in satir and satir.strip():
                print(f"[{etiket}] {satir}", end="", flush=True)
        return proc.wait()


# --------------------------------------------------------- 1. kosullari uret
for rol in kategoriler:
    kod = kos(["gift_contamination.recsys.conditions", "--category", rol, "--condition", "all"],
              "", f"conditions_{rol}")
    if kod != 0:
        raise SystemExit(f"kosul uretimi {rol} icin {kod} koduyla bitti - out/logs/conditions_{rol}.log")
    for f in (REPO_DIR / "reports" / "results").glob(f"condition_{SLUG[rol]}_*.json"):
        shutil.copy(f, OUT / f.name)

# ------------------------------------------------------------ 2. is listesi
if PLAN == "probe":
    isler = [(kategoriler[0], "C0", m, seedler[0]) for m in modeller]
elif PLAN == "matrix":
    ilk = seedler[0]

    def oncelik(is_: tuple) -> tuple:
        rol, kosul, model, seed = is_
        if rol == "low" and kosul == "C3":
            kademe = 3
        elif seed == ilk:
            kademe = 0
        else:
            kademe = 1 if model == "SASRec" else 2
        return (kademe, seedler.index(seed), kosullar.index(kosul), kategoriler.index(rol),
                modeller.index(model))

    isler = sorted(((r, k, m, s) for r in kategoriler for k in kosullar for m in modeller
                    for s in seedler), key=oncelik)
else:
    raise SystemExit(f"PLAN bilinmiyor: {PLAN}")
print(f"\n{len(isler)} kosu:", ", ".join(f"{SLUG[r][:4]}/{k}/{m}/s{s}" for r, k, m, s in isler))

n_gpu = len([s for s in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
             .stdout.splitlines() if s.startswith("GPU")])
print("gorunen GPU:", n_gpu)
if n_gpu < 1:
    raise SystemExit("GPU yok - Settings > Accelerator = GPU T4 x2")

# ------------------------------------------------------------ 3. kosular
kuyruk: queue.Queue = queue.Queue()
for is_ in isler:
    kuyruk.put(is_)
ozet: list[dict] = []
hatalar: list[str] = []
kilit = threading.Lock()


def isci(gpu: int) -> None:
    while True:
        try:
            rol, kosul, model, seed = kuyruk.get_nowait()
        except queue.Empty:
            return
        if (time.time() - T_BASLA) / 3600 > MAX_HOURS:
            with kilit:
                hatalar.append(f"{SLUG[rol]}/{kosul}/{model}/seed{seed}: MAX_HOURS doldu, baslatilmadi")
            continue
        slug = SLUG[rol]
        etiket = f"{slug}_{kosul}_{model}_seed{seed}"
        args = ["gift_contamination.recsys.run_experiment", "--category", rol, "--condition", kosul,
                "--model", model, "--seed", str(seed)] + (["--force"] if FORCE else [])
        t0 = time.time()
        kod = kos(args, str(gpu), etiket)
        rapor = REPO_DIR / "reports" / "results" / f"experiment_{etiket}.json"
        pu = REPO_DIR / "data" / "processed" / "recbole" / slug / "peruser" / f"{kosul}_{model}_seed{seed}.parquet"
        with kilit:
            if kod != 0 or not rapor.exists() or not pu.exists():
                hatalar.append(f"{etiket}: kod {kod} - out/logs/{etiket}.log")
                continue
            shutil.copy(rapor, OUT / rapor.name)
            (OUT / "peruser" / slug).mkdir(parents=True, exist_ok=True)
            shutil.copy(pu, OUT / "peruser" / slug / pu.name)
            r = json.loads(rapor.read_text(encoding="utf-8"))
            ozet.append({"kosu": etiket, "gpu": gpu, "duvar_sn": round(time.time() - t0),
                         "fit_sn": r["meta"].get("runtime_seconds", {}).get("fit"),
                         "recall@10": r["test"].get("recall@10"), "etiket": r["meta"].get("label_source")})
            print(f"\n>>> BITTI {etiket} (GPU {gpu}): {ozet[-1]}\n", flush=True)


iplikler = [threading.Thread(target=isci, args=(g,)) for g in range(min(n_gpu, 2))]
for t in iplikler:
    t.start()
for t in iplikler:
    t.join()

# ------------------------------------------------------------------ ozet
print(f"\nTOPLAM {time.time() - T_BASLA:.0f} sn · biten {len(ozet)} / {len(isler)}")
for o in sorted(ozet, key=lambda o: o["kosu"]):
    print("  ", o)
if hatalar:
    print("\nBITMEYEN / HATALI:")
    for h in hatalar:
        print("  ", h)
if PLAN == "probe" and ozet:
    print("\nZAMAN SONDASI: yukaridaki fit_sn degerleriyle matrisin suresi hesaplanir "
          f"({len(kategoriler)} kategori x {len(kosullar)} kosul x {len(seedler)} seed, model basina).")
print("\nindirilecek:", sorted(p.name for p in OUT.iterdir()))
