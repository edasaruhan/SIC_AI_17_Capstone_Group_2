"""Kaggle notebook'una YAPISTIRILACAK tek hucre.

Neden .py ve notebook degil: notebook JSON'u diff'lenemez, cikti hucreleri
depoyu sisirir ve iki kisi ayni anda duzenleyince catisir. Bu dosya duz Python;
Kaggle'da tek bir hucreye yapistirilir.

ON KOSULLAR (Kaggle arayuzunde, elle):
  1. Notebook > Settings > Accelerator = **GPU T4 x2**
     Iki kart da kullanilir: `detection.gpus: auto` -> iki VERI-PARALEL surec,
     her biri satirlarin yarisini alir. Sure ~yariya, kota da ~yariya iner
     (Kaggle kotayi OTURUM saati olarak sayiyor, GPU basina degil).
     Tek T4 de calisir, sadece iki kat surer.
  2. Notebook > Settings > Internet = **On**  (vLLM kurulumu + HF model indirme)
  3. Add Input > Datasets > kendi ozel dataset'iniz:
     `data/interim/*_annotation_sample.parquet` dosyalari (4 dosya, ~6 MB)
     Ekledikten sonra sag panelde gorunen TAM yolu DATASET_PATH'e yazin.

KOSU BITINCE: /kaggle/working/out/ altindaki parquet + json dosyalarini indirin
ve depodaki data/annotations/ + reports/results/ altina koyun.

Kesinti olursa: ayni hucreyi tekrar calistirin (FORCE=False). Bitmis satirlar
atlanir - `_shards/` klasoru /kaggle/working altinda durdugu surece. Kaggle
oturumu tamamen sifirlanirsa parcalar da gider; o yuzden uzun kosularda ara ara
`out/` klasorunu indirin.

DUMAN TESTINDEN TAM KOSUYA GECERKEN: LIMIT=None **ve** FORCE=True. Aksi halde
diskte kalan 200 satirlik parquet yuzunden kod kosuyu reddeder (dogru davranis;
sessizce atlamaktansa hata vermeyi tercih ediyoruz).

IKI KOSU MODU VAR:
  TRIAL=None  -> kategori kosusu (11.800 satir, ~57 dk). Kapi 1'in 1-3. olcutleri.
  TRIAL=200   -> deneme kosusu (200 satir, ~1 dk). Kapi 1'in 4. olcutu: elle
                 etiketlenmis satirlarla uyum. Bu satirlar annotation orneginin
                 ICINDE DEGIL, o yuzden ayri bir kosu gerekiyor.
"""

# ---------------------------------------------------------------- ayarlar
CATEGORY = "high"          # high=Toys_and_Games · pilot=All_Beauty · mid · low
# Kaggle'da "Add Input > Datasets" ile ekledikten sonra sag panelde gorunen TAM yol.
DATASET_PATH = "/kaggle/input/recsys-interim"
REPO = "https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_2.git"
LIMIT = None               # duman testi icin ornegin 200; tam kosu icin None
# Diskteki cikti EZILSIN mi? Duman testinden (LIMIT=200) sonra tam kosuya
# gecerken TRUE yapin: 200 satirlik parquet /kaggle/working'de duruyor ve
# kod onu dogru sekilde reddediyor. Yarim kalmis bir kosuya DEVAM etmek
# istiyorsaniz False birakin - parcalar korunur ve bitmis satirlar atlanir.
FORCE = False
# Kapi 1'in DORDUNCU olcutu: elle etiketlenmis 200 satirlik deneme seti.
# Sayi verilirse KATEGORI KOSUSU YERINE o kosar (~200 satir, ~1 dakika uretim).
# Onkosul: kaynak dosya yerelde uretilip dataset'e eklenmis olmali -
#   python -m gift_contamination.data.sampling --trial-source 200
# Bu satirlar annotation orneginin ICINDE DEGIL; ayrica etiketlenmeleri sart.
TRIAL = None               # kategori kosusu icin None; deneme kosusu icin 200

# ------------------------------------------------------------------ kurulum
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def sh(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


WORK = Path("/kaggle/working")
REPO_DIR = WORK / "SIC_AI_17_Capstone_Group_2" / "repo"

# DEPO HER KOSUDA GUNCELLENIR. Onceki hali `if not REPO_DIR.exists(): clone`
# idi ve /kaggle/working oturum boyunca kaldigi icin depo BIR KEZ klonlanip
# bir daha hic guncellenmiyordu: yerelde duzeltilen bir hata Kaggle'a asla
# ulasmiyordu ve kosu sessizce eski kodla devam ediyordu. Fark ancak cikti
# formatindan anlasilabiliyordu.
CLONE = WORK / "SIC_AI_17_Capstone_Group_2"
if REPO_DIR.exists():
    sh(f"git -C {CLONE} fetch --depth 1 origin main")
    sh(f"git -C {CLONE} reset --hard origin/main")
else:
    sh(f"git clone --depth 1 {REPO} {CLONE}")
# Hangi surumun kostugu KAYDA GECSIN; cikti bir kod surumune baglanabilmeli.
sh(f"git -C {CLONE} --no-pager log -1 --format='kod surumu: %h %s'")

# vLLM Kaggle imajinda kurulu DEGIL. Surum araligi bilerek genis: structured
# output API'si 0.10'da yeniden adlandirildi ve `llm_annotate` ikisini de
# destekliyor, o yuzden burada bir surume kilitlenmiyoruz.
try:
    import vllm  # noqa: F401, PLC0415

    print("vLLM zaten kurulu, pip atlaniyor")
except ImportError:
    sh(f"{sys.executable} -m pip install -q 'vllm>=0.7' polars pydantic pyyaml "
       "python-dotenv")

sys.path.insert(0, str(REPO_DIR / "src"))
os.chdir(REPO_DIR)

# ------------------------------------------------- ornekleme dosyalarini yerlestir
# Config yollari depo koku ile goreli; dataset'ten kopyalamak, Kaggle'a ozel bir
# config turevi tutmaktan basit ve az hataya acik (6 MB).
src_dir = Path(DATASET_PATH)
dest_dir = REPO_DIR / "data" / "interim"
dest_dir.mkdir(parents=True, exist_ok=True)

found = sorted(src_dir.rglob("*_annotation_sample.parquet"))
# Deneme kaynagi da ayni klasore gidiyor: `trial_source_path` onu
# data/interim altinda ariyor.
found += sorted(src_dir.rglob("prompt_trial_*_source.parquet"))
if not found:
    raise SystemExit(
        f"Ornekleme dosyasi bulunamadi: {src_dir}\n"
        "Add Input > Datasets ile ozel dataset'i ekleyin ve DATASET_PATH'i "
        "sag paneldeki TAM yol ile degistirin."
    )
for f in found:
    shutil.copy(f, dest_dir / f.name)
print(f"{len(found)} girdi dosyasi kopyalandi -> {dest_dir}")
if TRIAL and not any(p.name.startswith("prompt_trial_") for p in found):
    raise SystemExit(
        f"""TRIAL={TRIAL} istendi ama dataset'te prompt_trial_{TRIAL}_source.parquet yok.
Yerelde uretip dataset'e ekleyin (dosya ~40 KB):
  python -m gift_contamination.data.sampling --trial-source {TRIAL}"""
    )

# ------------------------------------------------------------------- kosu
from gift_contamination.config import Config  # noqa: E402
from gift_contamination.detection.llm_annotate import (  # noqa: E402
    annotate,
    annotate_trial,
    annotation_path,
    annotation_stats_path,
    trial_stats_path,
)

import torch  # noqa: E402

cfg = Config.load("configs/base.yaml")
print("model  :", cfg.get("detection.primary_model"))
print("prompt :", cfg.get("detection.prompt_path"))
print("prefix caching:", cfg.get("detection.enable_prefix_caching"))
print("gorunen GPU   :", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if torch.cuda.device_count() < 2:
    print("!! UYARI: tek GPU gorunuyor. Settings > Accelerator = GPU T4 x2 mi?")

print("force  :", FORCE, "| limit:", LIMIT, "| trial:", TRIAL)

# Deneme kosusu TEK SURECTE koşar: 200 satir icin surec cogaltmanin kazanci
# kurulum maliyetinin altinda kalir (kurulum ~200 sn, uretim ~55 sn).
if TRIAL:
    out = annotate_trial(cfg, TRIAL, backend_name="vllm", force=FORCE)
    stats_file = trial_stats_path(cfg, TRIAL)
else:
    out = annotate(cfg, CATEGORY, backend_name="vllm", limit=LIMIT, force=FORCE)
    stats_file = annotation_stats_path(cfg, CATEGORY)

# ------------------------------------------------------------------- cikti
import json  # noqa: E402

report = json.loads(stats_file.read_text(encoding="utf-8"))
print(json.dumps(report, ensure_ascii=False, indent=2))

# Gerceklesen throughput: dokumanlardaki tahminin yerine bu sayi gecer.
# 73 dakikalik bir kosunun SONUNDA KeyError ile patlamak kabul edilemez, o
# yuzden alanlar `.get()` ile okunuyor.
tp = report.get("throughput", {})
meta = report.get("meta", {})
gpus = f"{meta.get('n_workers')} GPU, {tp.get('rows_per_s_per_gpu')} satir/sn/GPU" if not TRIAL else "tek surec"
print(f"\nuretim     : {tp.get('rows_per_s_generating')} satir/sn ({gpus})"
      f" · {tp.get('output_tokens_per_s')} cikti-token/sn")
print(f"kurulum    : {tp.get('startup_s')} sn (tek seferlik, satir sayisindan bagimsiz)")
print(f"duvar saati: {tp.get('elapsed_s')} sn")
print(f"ortak onek : {tp.get('shared_prefix_chars')} karakter "
      f"({tp.get('system_prompt_tokens')} token)")
if (tp.get("shared_prefix_chars") or 0) < 1000:
    print("!! UYARI: ortak onek cok kisa, prefix caching ise yaramiyor olabilir")
if meta.get("is_partial"):
    print(f"!! Bu KISMI bir kosu (limit={meta.get('limit')}). Tam kosu icin "
          "LIMIT=None ve FORCE=True yapin.")
if "rows_per_s_generating" not in tp:
    print("!! Rapor ESKI formatta - depo guncellenmemis olabilir, "
       "yukaridaki 'kod surumu' satirini kontrol edin.")

dl = WORK / "out"
dl.mkdir(exist_ok=True)
shutil.copy(out, dl / out.name)
shutil.copy(stats_file, dl / stats_file.name)
print(f"\nindirilecek dosyalar: {sorted(p.name for p in dl.iterdir())}")
