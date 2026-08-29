# Hediye Kontaminasyonu — AI in Marketing Capstone

Öneri sistemleri "satın alma = tercih" varsayar. Hediye alımlarında bu varsayım yanlıştır ve
kullanıcı profilini kirletir. Bu proje kontaminasyonu Amazon review metninden LLM ile tespit
ediyor, yaygınlığını ölçüyor ve öneri modelini bu etkileşimlerle/onlarsız eğitip karşılaştırıyor.

**Araştırma pipeline'ıdır, ürün değildir.** Deploy edilmez, API'si yoktur.

## Dokümanlar
| Dosya | İçerik |
|---|---|
| **`docs/GENEL_BAKIS.md`** | **Buradan başlayın** — problem, araştırma soruları, pipeline diyagramı, nerede olduğumuz |
| `CLAUDE.md` | Claude Code için ana context ve bağlayıcı kurallar |
| `docs/PROJECT_SPEC.md` | Tam proje dokümanı — problem, gap analizi, iş bölümü |
| `docs/ROADMAP.md` | Teknik roadmap — aşama detayları, hafta planı |
| `docs/DECISIONS.md` | Spec'ten sapmalar ve gerekçeleri (tarihli) |
| `docs/ETIKETLEME_REHBERI.md` | **Etiketleyene verilecek rehber.** Kendi başına yeterli: projeyi bilmeyen biri sıfırdan okuyup 500 satırı etiketleyebilir. Hafta 4'te üç kişi de bunu okur. |

## Kurulum

RecBole ile vLLM aynı ortamda çakışır — **ayrı venv'ler gerekir**. Ama Data Research /
EDA aşaması ikisini de kullanmıyor; çekirdek ortam yeter.

```bash
cd repo
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                    # python -m gift_contamination.* icin

cp .env.example .env    # HF_TOKEN ve WANDB_API_KEY (ikisi de opsiyonel)
```

**LLM annotation (Hafta 3–5) yerelde koşmuyor.** Kaggle'da iki T4 üzerinde
koşuyor ve ortamı `scripts/kaggle_annotate.py` kendi kuruyor (vLLM dahil). Elinde
uygun bir GPU olan biri yerelde denemek isterse `requirements-llm.txt` var, ama
projenin ürettiği etiketlerin hepsi Kaggle'dan geldi.

**RecBole deneyi (Hafta 6–7) ayrı bir venv istiyor** — `.venv-recbole`. Ayırmak
tercih değil zorunluluk: RecBole 1.2.0 `np.float_` kullanıyor, numpy 2.0'da o ad
kaldırıldı; ana ortam numpy 2.5 üzerinde. Pin'lerin her biri ölçülerek bulundu,
gerekçeleri dosyanın içinde yazıyor.

```bash
python -m venv .venv-recbole
.venv-recbole/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-recbole/Scripts/python.exe -m pip install -r requirements-recbole.txt
```

İki ortam birbirine **dosyayla** bağlı: `data/processed/recbole/…/*.inter`. Ortak
kod import edilmiyor, o yüzden çakışma da yok.

> **Disk uyarısı.** Dört kategori ham hâlde ~16.3 GB. Proje bir bulut-senkron klasörü
> (OneDrive/Dropbox) altındaysa `data/raw`'ı senkron dışına alın — Windows'ta bir junction
> yeterlidir ve config'i değiştirmez:
> `mklink /J data\raw D:\amazon-reviews-2023`

## Hızlı başlangıç

Her komut idempotent: çıktı varsa `--force` olmadan yeniden hesaplamaz.
`--category` değerleri: `pilot` · `high` · `mid` · `low` · `all`.

```bash
# Hafta 1-2 — veri, vekil, EDA, örneklem
python -m gift_contamination.data.download          --category pilot
python -m gift_contamination.data.preprocess        --category pilot
python -m gift_contamination.analysis.keyword_scan  --category pilot
python -m gift_contamination.analysis.eda
python -m gift_contamination.analysis.deep_eda
# Örneklem boyutu config'ten gelir (`sampling.n_annotate`) — CLI bayrağı yok
python -m gift_contamination.data.sampling          --category all

# Hafta 3-5 — etiketleme Kaggle'da koşar (scripts/kaggle_annotate.py),
# çıktısı data/annotations/ altına indirilir. Sonra:
python -m gift_contamination.analysis.gate1         --category all
python -m gift_contamination.analysis.prevalence    --category all

# Hafta 4 — insan doğrulaması (etiketleyen kişi LLM'in cevabını GÖRMEZ)
python -m gift_contamination.data.sampling   --validation --category all
python -m gift_contamination.data.labelsheet --validation --export --annotators 3
#   ... üç kişi kendi xlsx'ini bağımsız doldurur ...
python -m gift_contamination.data.labelsheet --validation --ingest --annotators 3
python -m gift_contamination.analysis.validation

# Hafta 6 — deney iskelesi (bugün SÖZCÜKSEL VEKİL etiketiyle; çıktı
# `reportable: false` damgalı). run_experiment .venv-recbole altında koşar.
python -m gift_contamination.recsys.atomic     --category mid
python -m gift_contamination.recsys.conditions --category mid --condition all
PYTHONPATH=src .venv-recbole/Scripts/python.exe \
  -m gift_contamination.recsys.run_experiment --category mid --condition C0 --model BPR
```

Örneklem **üç ayrı çerçeve** üretir ve bu ayrım korunmak zorundadır:
`sample_frame == 'main'` orantılı katmanlı örnektir ve yaygınlık oranı **yalnızca**
onun üzerinden hesaplanır. `boost` (anahtar kelimeyle işaretlenmiş havuzdan ek
pozitifler) ve `boost_received` ("hediye aldım" satırları, zor negatifler) orana
**katılmaz**. Pilot kategoride karıştırmanın bedeli ölçüldü: aynı örneklem içinde
`main` çerçevesi %2,02 verirken havuzlanmış hâli %14,42 görünüyor — **7,1 kat**
şişme. (`main`'in kendisi korpusun gerçek oranı %2,13'e yakınsıyor; doğru
karşılaştırma bu ikisi değil, örneklem içindeki `main` ile havuzlanmış hâlidir.)

Elle doğrulama örneklemi (üretilen CSV birebir review metni taşır, git'e **girmez**):

```bash
python -m gift_contamination.analysis.precision_check sample --category all
# label sutununu doldur: gift_given | self | household | unclear
python -m gift_contamination.analysis.precision_check score
```

Testler:

```bash
pytest tests -q
```

> `datasets.load_dataset(..., trust_remote_code=True)` **kullanılmıyor** — `datasets` 4.x
> loading-script desteğini kaldırdı. Dosyalar `hf_hub_download` ile doğrudan çekiliyor.
> Bkz. `docs/DECISIONS.md`.

## Çıktılar

Veri dosyaları git'e **girmez**; `reports/` altındaki toplulaştırılmış sonuçlar girer.

| Yol | İçerik |
|---|---|
| `data/interim/<kategori>_clean.parquet` | verified + min_words + dedup — annotation ve EDA korpusu |
| `data/interim/<kategori>_kcore.parquet` | + iteratif 5-core — recsys deneyinin korpusu |
| `data/interim/<kategori>_annotation_sample.parquet` | üç çerçeveli örneklem (main / boost / boost_received) |
| `data/annotations/<kategori>_llm.parquet` | LLM etiketleri — `evidence_span` birebir metin taşır, **git'e girmez** |
| `data/annotations/human/` | elle etiketleme sayfaları — birebir metin, **git'e girmez** |
| `data/processed/recbole/<kategori>/` | RecBole `.inter` dosyaları ve koşullar (C0–C4) |
| `reports/results/` | huni sayaçları, keyword oranları, EDA tabloları, Kapı 1, yaygınlık, deney raporları |
| `reports/figures/` | F1–F19 (F18 = Hafta 4 doğrulaması, henüz üretilmedi) |
| `../data-research/data-research.md` | Data Research teslimi |

## Veri
`McAuley-Lab/Amazon-Reviews-2023` (HuggingFace, açık). Kategori bazlı indirilir.
Veri dosyaları git'e **girmez**.

## Ekip
| Kulvar | Sorumluluk | Kişi |
|---|---|---|
| A — Veri & Tespit | preprocess, örnekleme, LLM annotation, distillation | TBD |
| B — Doğrulama & Analiz | insan etiketleme, kappa/F1, mevsimsellik, EDA | TBD |
| C — Recsys & Deney | RecBole, C0–C4, istatistik, pazarlama metrikleri | TBD |

## Lisans / Etik
Amazon Reviews 2023 akademik kullanım içindir. Review metinlerinde kişisel bilgi geçebilir;
yayınlanan hiçbir çıktıda ham review metni birebir paylaşılmaz.
