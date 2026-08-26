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
| `docs/ETIKETLEME_REHBERI.md` | Elle etiketleme kuralları — Hafta 4'te 3 annotator da bunu okur |

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

LLM ve recsys aşamaları geldiğinde (Hafta 3 ve 6) ayrı ortamlar kurulur:

```bash
python -m venv venv-llm    && venv-llm/Scripts/activate    && pip install -r requirements-llm.txt
python -m venv venv-recsys && venv-recsys/Scripts/activate && pip install -r requirements-recsys.txt
```

> **Disk uyarısı.** Dört kategori ham hâlde ~16.3 GB. Proje bir bulut-senkron klasörü
> (OneDrive/Dropbox) altındaysa `data/raw`'ı senkron dışına alın — Windows'ta bir junction
> yeterlidir ve config'i değiştirmez:
> `mklink /J data
aw D:mazon-reviews-2023`

## Hızlı başlangıç

Her komut idempotent: çıktı varsa `--force` olmadan yeniden hesaplamaz.
`--category` değerleri: `pilot` · `high` · `mid` · `low` · `all`.

```bash
python -m gift_contamination.data.download          --category pilot
python -m gift_contamination.data.preprocess        --category pilot
python -m gift_contamination.analysis.keyword_scan  --category pilot
python -m gift_contamination.analysis.eda
python -m gift_contamination.data.sampling          --category all
```

Örneklem **üç ayrı çerçeve** üretir ve bu ayrım korunmak zorundadır:
`sample_frame == 'main'` orantılı katmanlı örnektir ve yaygınlık oranı **yalnızca**
onun üzerinden hesaplanır. `boost` (anahtar kelimeyle işaretlenmiş havuzdan ek
pozitifler) ve `boost_received` ("hediye aldım" satırları, zor negatifler) orana
**katılmaz**. Pilot kategoride karıştırmanın bedeli ölçüldü: gerçek oran %2.13 iken
havuzlanmış oran %14.42 görünüyor — yedi kat şişme.

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

| Yol | İçerik |
|---|---|
| `data/interim/<kategori>_clean.parquet` | verified + min_words + dedup — annotation ve EDA korpusu |
| `data/interim/<kategori>_kcore.parquet` | + iteratif 5-core — recsys deneyinin korpusu |
| `reports/results/` | huni sayaçları, keyword oranları, EDA tabloları |
| `reports/figures/` | F1–F8 figürleri |
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
