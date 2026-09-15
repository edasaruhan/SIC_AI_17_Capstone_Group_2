"""Kaggle notebook'una YAPISTIRILACAK tek hucre - Hafta 5: damitma + tam korpus cikarimi.

`kaggle_annotate.py` ile ayni desen (neden .py: diff'lenir, catismaz).

NE YAPAR:
  1. `distill train`  - ModernBERT-base'i 46.655 LLM etiketiyle egitir (TEK T4),
                        sadakat kapisini olcer -> distill_report_<model>.json
  2. Kapi PASS ise `inference run` - Toys (GPU 0) ve Grocery (GPU 1) 5-core'unu
                        AYNI ANDA etiketler (4,6M satir) -> <kategori>_inferred.parquet
  Kapi PASS degilse 2. adim KOSMAZ. Esik degistirilmez (DECISIONS 2026-09-14).

ON KOSULLAR (Kaggle arayuzunde, elle):
  1. Settings > Accelerator = **GPU T4 x2**   (tek T4 de calisir; cikarim sirayla, 2x surer)
  2. Settings > Internet = **On**             (model indirme; gerekirse transformers guncellemesi)
  3. Add Input > Datasets > **OZEL (private)** bir dataset. Dosyalar REVIEW METNI
     tasiyor - herkese acik dataset YAPMAYIN. Yerelde uretilir:
        python -m gift_contamination.detection.distill   prepare
        python -m gift_contamination.detection.inference prepare --category high
        python -m gift_contamination.detection.inference prepare --category low
     Yuklenecekler:
        data/processed/distill/bundle.parquet                              (~6 MB)
        data/processed/distill/human.parquet                               (~80 KB)
        data/processed/distill/infer_Toys_and_Games.parquet                (~263 MB)
        data/processed/distill/infer_Grocery_and_Gourmet_Food.parquet      (~278 MB)
        data/annotations/Toys_and_Games_llm.parquet                        (istege bagli,
        data/annotations/Grocery_and_Gourmet_Food_llm.parquet               ortusme sagdamasi)
     Dataset'in adi onemsiz: DATASET_PATH = None iken dosyalar /kaggle/input altinda
     aranir (Kaggle'in baglama yolu surumden surume degisiyor; elle yazilan yol en sik
     hata kaynagiydi). Bulunan yollar hucrede yazdirilir.
     Ogretmenin C1 F1'i (kapinin 3. olcutu) depodaki reports/results/validation_500.json'dan
     okunur - dataset'e koymaniz gerekmez.

KOSU BITINCE: /kaggle/working/out/ klasorunu indirin ve depoya yerlestirin:
     distill_report_*.json, inference_*.json  -> reports/results/
     *_inferred.parquet                       -> data/annotations/
     *.log                                    -> saklamak isterseniz; depoya girmez

KESINTI OLURSA: interaktif oturum hala aciksa hucreyi AYNEN yeniden calistirin
(FORCE=False). Egitim raporu varsa egitim atlanir; cikarimda bitmis parcalar
(200.000 satir) yeniden hesaplanmaz. "Save & Run All" versiyonu ise her seferinde
SIFIRDAN baslar (/kaggle/working bos gelir).

KAPI PASS DEGILSE hucre HATA VERMEDEN biter ve cikarim kosmaz: hata veren bir
"Save & Run All" versiyonunda cikti dosyalarinin saklanacagi garanti degil; rapor
kaybolmasin.

KAPI FAIL VERIRSE: out/ klasorunu indirin; distill_report_base.json depoya commit
edilip push'lanir (FAIL once kayda girer). SONRA MODEL = "fallback" yapip BIR KEZ
daha calistirin. Yeni oturum depoyu klonlar ve ana modelin FAIL raporunu oradan
okur - kod, ana model FAIL degilse (rapor yoksa da) yedegi reddeder (onceden kayitli
kural). Yedek de gecemezse deney DURUR - sonucu ekiple paylasin.
"""

# ---------------------------------------------------------------- ayarlar
DATASET_PATH = None     # None: /kaggle/input altinda aranir · or. "/kaggle/input/recsys-distill"
REPO = "https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_2.git"
MODEL = "base"          # "base" = ModernBERT-base · "fallback" = DeBERTa-v3 (yalnizca base FAIL ise)
FP16 = True             # egitim kaybi NaN verirse False (fp32; ~2x yavas)
RUN_INFERENCE = True    # False: yalnizca egitim + kapi
FORCE = False           # True: diskteki rapor/cikti EZILIR (yarim kosuya devam icin False birakin)
COPY_MODEL = False      # True: egitilmis model (~600 MB) de out/'a kopyalanir

# ------------------------------------------------------------------ kurulum
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


def sh(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


WORK = Path("/kaggle/working")
CLONE = WORK / "SIC_AI_17_Capstone_Group_2"
REPO_DIR = CLONE / "repo"
OUT = WORK / "out"
OUT.mkdir(exist_ok=True)

# Depo HER KOSUDA guncellenir (kaggle_annotate.py'deki 2026-08-28 hatasi).
if REPO_DIR.exists():
    sh(f"git -C {CLONE} fetch --depth 1 origin main")
    sh(f"git -C {CLONE} reset --hard origin/main")
else:
    sh(f"git clone --depth 1 {REPO} {CLONE}")
sh(f"git -C {CLONE} --no-pager log -1 --format='kod surumu: %h %s'")


def _surum(modul: str) -> str | None:
    """Kurulu degilse None; kurulu ama surum alani yoksa '?'."""
    try:
        return getattr(__import__(modul), "__version__", "?")
    except Exception:  # noqa: BLE001 - kurulu degil
        return None


# ModernBERT transformers 4.48'de geldi. Kaggle imajindaki surum bilinmiyor.
# Guncelleme bu cekirdekte ESKI transformers'i bellekte birakir; bu yuzden asil
# is asagida ALT SURECLERDE kosuyor (taze yorumlayici, guncel paket).
eksik = []
tf = _surum("transformers")
if tf is None or tuple(int(p) for p in tf.split(".")[:2] if p.isdigit()) < (4, 48):
    eksik += ["'transformers>=4.48'", "'accelerate>=0.26'"]
# matplotlib: kapi olcumu `analysis.validation` uzerinden geliyor ve o modul
# figur kodunu da import ediyor (Kaggle imajinda kurulu; yoksa kurulur).
for modul, paket in [("accelerate", "accelerate"), ("datasets", "datasets"),
                     ("polars", "polars"), ("pydantic", "pydantic"),
                     ("yaml", "pyyaml"), ("dotenv", "python-dotenv"),
                     ("matplotlib", "matplotlib")]:
    if _surum(modul) is None and paket not in " ".join(eksik):
        eksik.append(paket)
if eksik:
    sh(f"{sys.executable} -m pip install -q -U {' '.join(eksik)}")
print("transformers:", tf, "->", "guncellendi" if eksik else "yeterli")

# ------------------------------------------------------- girdileri yerlestir
src = Path(DATASET_PATH or "/kaggle/input")
if not src.exists():
    print(f"!! {src} yok - /kaggle/input altinda araniyor")
    src = Path("/kaggle/input")
distill_dir = REPO_DIR / "data" / "processed" / "distill"
ann_dir = REPO_DIR / "data" / "annotations"
distill_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)

gerekli = ["bundle.parquet", "human.parquet"]
if RUN_INFERENCE:
    gerekli += ["infer_Toys_and_Games.parquet", "infer_Grocery_and_Gourmet_Food.parquet"]
for ad in gerekli:
    bulunan = sorted(src.rglob(ad))
    if not bulunan:
        raise SystemExit(
            f"{ad} bulunamadi: {src}\nDataset notebook'a eklendi mi (sag panel > Add Input)? "
            "Yerelde `prepare` adimlarini kosup dataset'e ekleyin."
        )
    if len(bulunan) > 1:
        print(f"!! {ad} birden fazla yerde; ilki kullaniliyor: {[str(p) for p in bulunan]}")
    print(f"{ad} <- {bulunan[0]}")
    shutil.copy(bulunan[0], distill_dir / ad)
for f in sorted(src.rglob("*_llm.parquet")):
    shutil.copy(f, ann_dir / f.name)
print("girdiler:", sorted(p.name for p in distill_dir.glob("*.parquet")),
      "| LLM etiketi:", sorted(p.name for p in ann_dir.glob("*_llm.parquet")))


# ----------------------------------------------------------- alt surec kosucu
def _ortam(gpu: str) -> dict:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": str(REPO_DIR / "src"),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TQDM_MININTERVAL": "30",          # ilerleme cubugu cikti hucresini bogmasin
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }


def baslat(args: list[str], gpu: str, etiket: str) -> tuple[subprocess.Popen, threading.Thread]:
    """Alt sureci baslatir; ciktisini hem hucreye (onekli) hem out/<etiket>.log'a yazar."""
    log = open(OUT / f"{etiket}.log", "a", encoding="utf-8")  # noqa: SIM115 - thread kapatir
    proc = subprocess.Popen(
        [sys.executable, "-m", *args], cwd=REPO_DIR, env=_ortam(gpu),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
    )

    def akit() -> None:
        for satir in proc.stdout:
            log.write(satir)
            log.flush()
            print(f"[{etiket}] {satir}", end="", flush=True)
        log.close()

    t = threading.Thread(target=akit, daemon=True)
    t.start()
    return proc, t


def bekle(isler: list[tuple[str, subprocess.Popen, threading.Thread]]) -> None:
    for etiket, proc, t in isler:
        kod = proc.wait()
        t.join()
        if kod != 0:
            raise SystemExit(f"{etiket} {kod} koduyla bitti - ayrinti out/{etiket}.log")


n_gpu = len([s for s in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
             .stdout.splitlines() if s.startswith("GPU")])
print("gorunen GPU:", n_gpu)
if n_gpu < 2:
    print("!! UYARI: tek GPU. Egitim ayni; cikarim sirayla kosar (~2x sure). "
          "Settings > Accelerator = GPU T4 x2 mi?")

# ------------------------------------------------------------------ 1. egitim
import json  # noqa: E402

t0 = time.time()
egitim = ["gift_contamination.detection.distill", "train", "--model", MODEL]
egitim += [] if FP16 else ["--no-fp16"]
egitim += ["--force"] if FORCE else []
# TEK KART: iki kart gorunurse Trainer DataParallel'e gecer ve etkin grup boyu
# config'tekinin iki katina cikar (yani config'te yazan hiperparametre kosmaz).
p, t = baslat(egitim, "0", f"distill_{MODEL}")
bekle([(f"distill_{MODEL}", p, t)])

rapor_yolu = REPO_DIR / "reports" / "results" / f"distill_report_{MODEL}.json"
rapor = json.loads(rapor_yolu.read_text(encoding="utf-8"))
shutil.copy(rapor_yolu, OUT / rapor_yolu.name)
print(f"\negitim + olcum: {time.time() - t0:.0f} sn")
print("egitim bilgisi:", json.dumps(rapor["meta"].get("training", {}), ensure_ascii=False))
print(f"\nSADAKAT KAPISI ({MODEL}) -> {rapor['verdict']}")
for ad, c in rapor["criteria"].items():
    print(f"  {ad:44s} {c}")

kapi_gecti = rapor["verdict"] == "PASS"
if not kapi_gecti:
    # Hata VERMEDEN biter: "Save & Run All" versiyonunda out/ (rapor dahil) saklansin.
    if rapor["verdict"] == "FAIL" and MODEL == "base":
        print("\n!! KAPI KACTI - cikarim KOSMAYACAK. out/ klasorunu indirin; distill_report_base.json "
              "depoya commit edildikten SONRA MODEL = 'fallback' ile BIR KEZ daha calistirin.")
    else:
        print(f"\n!! KAPI {rapor['verdict']} - cikarim KOSMAYACAK. out/ klasorunu indirip sonucu "
              "ekiple paylasin. Esik degistirilmez.")

# ---------------------------------------------------------------- 2. cikarim
if RUN_INFERENCE and kapi_gecti:
    t1 = time.time()
    kosu = ["gift_contamination.detection.inference", "run", "--model", MODEL]
    kosu += [] if FP16 else ["--no-fp16"]
    kosu += ["--force"] if FORCE else []
    roller = ["high", "low"]
    if n_gpu >= 2:
        isler = []
        for gpu, rol in enumerate(roller):
            p, t = baslat([*kosu, "--category", rol], str(gpu), f"inference_{rol}")
            isler.append((f"inference_{rol}", p, t))
        bekle(isler)
    else:
        for rol in roller:
            p, t = baslat([*kosu, "--category", rol], "0", f"inference_{rol}")
            bekle([(f"inference_{rol}", p, t)])
    print(f"\ncikarim: {time.time() - t1:.0f} sn")

    for slug in ("Toys_and_Games", "Grocery_and_Gourmet_Food"):
        r = REPO_DIR / "reports" / "results" / f"inference_{slug}.json"
        d = ann_dir / f"{slug}_inferred.parquet"
        shutil.copy(r, OUT / r.name)
        shutil.copy(d, OUT / d.name)
        ozet = json.loads(r.read_text(encoding="utf-8"))
        print(f"\n{slug}: {ozet['n_rows']:,} satir · {ozet['throughput']}")
        print("  kontaminasyon payi %:", ozet["contamination_share_pct"])
        print("  mevsimsellik        :", {k: v.get("ratio") for k, v in ozet["seasonality"].items()
                                          if isinstance(v, dict)})
        print("  LLM'le ortusme      :", ozet["overlap_with_llm"])

if COPY_MODEL:
    model_kok = REPO_DIR / "models" / "distill"
    for d in model_kok.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            shutil.copytree(d, OUT / "models" / d.name, dirs_exist_ok=True)

print(f"\nindirilecek dosyalar: {sorted(p.name for p in OUT.iterdir())}")
