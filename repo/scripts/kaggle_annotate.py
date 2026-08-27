"""Kaggle notebook'una YAPISTIRILACAK tek hucre.

Neden .py ve notebook degil: notebook JSON'u diff'lenemez, cikti hucreleri
depoyu sisirir ve iki kisi ayni anda duzenleyince catisir. Bu dosya duz Python;
Kaggle'da tek bir hucreye yapistirilir.

ON KOSULLAR (Kaggle arayuzunde, elle):
  1. Notebook > Settings > Accelerator = **GPU T4 x2**  (tek T4 de yeter)
  2. Notebook > Settings > Internet = **On**  (vLLM kurulumu + HF model indirme)
  3. Add Input > Datasets > kendi ozel dataset'iniz:
     `data/interim/*_annotation_sample.parquet` dosyalari (4 dosya, ~6 MB)
     Dataset adini asagida SAMPLE_DATASET degiskenine yazin.

KOSU BITINCE: /kaggle/working/out/ altindaki parquet + json dosyalarini indirin
ve depodaki data/annotations/ + reports/results/ altina koyun.

Kesinti olursa: ayni hucreyi tekrar calistirin. Bitmis satirlar atlanir -
`_shards/` klasoru /kaggle/working altinda durdugu surece. Kaggle oturumu
tamamen sifirlanirsa parcalar da gider; o yuzden uzun kosularda ara ara
`out/` klasorunu indirin.
"""

# ---------------------------------------------------------------- ayarlar
CATEGORY = "high"          # high=Toys_and_Games · pilot=All_Beauty · mid · low
SAMPLE_DATASET = "gift-contamination-samples"   # Kaggle dataset klasor adi
REPO = "https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_2.git"
LIMIT = None               # duman testi icin ornegin 200; tam kosu icin None

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

if not REPO_DIR.exists():
    sh(f"git clone --depth 1 {REPO} {WORK / 'SIC_AI_17_Capstone_Group_2'}")

# vLLM Kaggle imajinda kurulu DEGIL. Surum araligi bilerek genis: structured
# output API'si 0.10'da yeniden adlandirildi ve `llm_annotate` ikisini de
# destekliyor, o yuzden burada bir surume kilitlenmiyoruz.
sh(f"{sys.executable} -m pip install -q 'vllm>=0.7' polars pydantic pyyaml python-dotenv")

sys.path.insert(0, str(REPO_DIR / "src"))
os.chdir(REPO_DIR)

# ------------------------------------------------- ornekleme dosyalarini yerlestir
# Config yollari depo koku ile goreli; dataset'ten kopyalamak, Kaggle'a ozel bir
# config turevi tutmaktan basit ve az hataya acik (6 MB).
src_dir = Path("/kaggle/input") / SAMPLE_DATASET
dest_dir = REPO_DIR / "data" / "interim"
dest_dir.mkdir(parents=True, exist_ok=True)

found = sorted(src_dir.rglob("*_annotation_sample.parquet"))
if not found:
    raise SystemExit(
        f"Ornekleme dosyasi bulunamadi: {src_dir}\n"
        "Add Input > Datasets ile ozel dataset'i ekleyin ve SAMPLE_DATASET'i duzeltin."
    )
for f in found:
    shutil.copy(f, dest_dir / f.name)
print(f"{len(found)} ornekleme dosyasi kopyalandi -> {dest_dir}")

# ------------------------------------------------------------------- kosu
from gift_contamination.config import Config  # noqa: E402
from gift_contamination.detection.llm_annotate import (  # noqa: E402
    annotate,
    annotation_path,
    annotation_stats_path,
)

cfg = Config.load("configs/base.yaml")
print("model  :", cfg.get("detection.primary_model"))
print("prompt :", cfg.get("detection.prompt_path"))
print("prefix caching:", cfg.get("detection.enable_prefix_caching"))

out = annotate(cfg, CATEGORY, backend_name="vllm", limit=LIMIT)

# ------------------------------------------------------------------- cikti
import json  # noqa: E402

report = json.loads(annotation_stats_path(cfg, CATEGORY).read_text(encoding="utf-8"))
print(json.dumps(report, ensure_ascii=False, indent=2))

# Gerceklesen throughput: dokumanlardaki tahminin yerine bu sayi gecer.
tp = report["throughput"]
print(f"\n{tp['rows_per_s']} satir/sn · {tp['output_tokens_per_s']} cikti-token/sn")
print(f"ortak onek: {tp['shared_prefix_chars']} karakter "
      f"({tp['system_prompt_tokens']} token)")
if tp["shared_prefix_chars"] < 1000:
    print("!! UYARI: ortak onek cok kisa, prefix caching ise yaramiyor olabilir")

dl = WORK / "out"
dl.mkdir(exist_ok=True)
shutil.copy(out, dl / out.name)
shutil.copy(annotation_stats_path(cfg, CATEGORY), dl / annotation_stats_path(cfg, CATEGORY).name)
print(f"\nindirilecek dosyalar: {sorted(p.name for p in dl.iterdir())}")
