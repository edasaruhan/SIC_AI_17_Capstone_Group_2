# Hediye Kontaminasyonu — SIC AI Capstone, Grup 2

Öneri sistemleri **"satın alma = tercih"** varsayar. Hediye alımlarında bu varsayım
yanlıştır: ürünü alan kişi onu kendisi için seçmemiştir, ama model bunu tercih sanar ve
sonraki önerileri kaydırır. Bu proje kontaminasyonu Amazon review metninden LLM ile tespit
ediyor, yaygınlığını ölçüyor, ve öneri modelini bu etkileşimlerle **ve onlarsız** eğitip
karşılaştırıyor.

**Araştırma pipeline'ıdır, ürün değildir** — deploy edilmiş bir servisi yoktur.

## Sonuç, tek tabloda

| Soru | Cevap |
|---|---|
| **RQ1** — hediye ne kadar yaygın? | Toys'ta review'ların **%19,0**'u [15,8–22,3] (insan kalibrasyonlu). Grocery / All Beauty / Video Games %5–8. Dört kategoride de Aralık–Ocak tepesi (1,58–2,08×). |
| **RQ2** — hediyeyi çıkarmak kendi sonraki alımın tahminini iyileştirir mi? | Hediye satırları **aynı sayıda rastgele satırdan belirgin daha az işe yarıyor** (C1 > C4): Toys'ta SASRec +%16, BPR +%40 Recall@10. Grocery'de çok küçük ya da saptanamıyor → **doz–yanıt tutuyor**. |
| **RQ3** — silmek mi, işaretlemek mi? | Bu çalışmadaki işaretleme (gölge token) her iki seçenekten de kötü ya da onlardan ayırt edilemiyor. |
| **RQ4** — liste ne kadar süre kirli kalıyor? | SASRec × Toys'ta fazla pay **2,0 puan**la başlıyor ve **~1 kendi alımında** yarıya iniyor (≈2–4 hafta). |

72 koşu (2 kategori × 2 model × 6 koşul × 3 seed), Kapı 2 dört hücrede de PASS, 463 test.
**İki doğrulama (kuralları sonuçlardan önce yazıldı, [`DECISIONS.md`](repo/docs/DECISIONS.md)
2026-09-21/22):**
- Seed 42'nin 24 hücresi bugünkü kodla Kaggle'da yeniden koşuldu: **24/24 bit bit aynı**,
  21–77 epoch, 300 tavanına değen yok.
- İkinci ve üçüncü etiketleyicinin sayfaları yapay zekâ / betik yardımıyla doldurulmuştu.
  İnsan etiketi sayılmadılar, bu yüzden etiketleyiciler arası uyum (κ) **ölçülmedi**; insan
  referansı hâlâ tek kişi.
**Capstone raporunun tamamı: [`final-report/final-report.md`](final-report/final-report.md)** ·
ayrıntı, güven aralıkları ve sınırlılıklar: [`repo/docs/SONUCLAR.md`](repo/docs/SONUCLAR.md) ·
her başlığı tek komutla basmak için: `python repo/scripts/demo.py`

## Bu depoda ne var

| Klasör | İçerik |
|---|---|
| **[`repo/`](repo/)** | **Kod, testler, config, sonuç JSON'ları ve figürler.** Buradan başlayın: [`repo/README.md`](repo/README.md) → kurulum, [`repo/docs/GENEL_BAKIS.md`](repo/docs/GENEL_BAKIS.md) → projenin tamamı |
| **[`final-report/`](final-report/)** | **Capstone final raporu** (Markdown, İngilizce) — problemden sonuçlara, her sayının kaynak JSON'u adıyla |
| **[`presentation/`](presentation/)** | **Sunum** (`.pptx` + PDF) ve tek sayfalık özet (`one_pager.pdf`); ikisini de üreten betikler |
| `concept-note/` | Ders teslimi — problem, hipotez, başarı ölçütleri |
| `literature-review/` | Ders teslimi — beş temada literatür |
| `data-research/` | Ders teslimi — veri kaynağı, EDA, F1–F16 figürleri (+ `.docx`) |
| `technology-review/` | Ders teslimi — model/kütüphane seçimleri ve gerekçeleri |
| `implementation-plan/` | Ders teslimi — haftalık plan, kapılar, iş bölümü |
| `model-refinement/` | Ders teslimi — model iyileştirme ve test (`.docx`) |
| `deployment/` | Ders teslimi — dağıtım değerlendirmesi (`.docx`) |
| `idea_proposal_grup_2.docx` | İlk fikir önerisi |
| [`LICENSE`](LICENSE) | Kod için MIT. Veri, model ağırlıkları ve ders teslimleri kapsam dışıdır — dosyanın kendisi bunu ayrı ayrı sayıyor. |

> Teslim belgelerindeki bazı erken ölçütler sonradan ölçümle değişti ya da tutturulamadı.
> Hepsi tarihli **errata** notlarıyla belgelerin kendi içinde işaretli; tam gerekçeler
> [`repo/docs/DECISIONS.md`](repo/docs/DECISIONS.md) içinde.

## Hızlı başlangıç

```bash
cd repo
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest tests -q                           # 463 test
```

Veri dosyaları (`data/`), model ağırlıkları (`models/`) ve kullanıcı başı deney çıktıları
git'e **girmez**. Ham review metni hiçbir commit'li çıktıda birebir paylaşılmaz.

## Veri ve etik

Kaynak: [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(açık, akademik kullanım). Review metinlerinde kişisel bilgi geçebilir; yayımlanan hiçbir
çıktıda ham metin, kullanıcı kimliği ya da mutlak dosya yolu yer almaz.

## Ekip

| Kulvar | Sorumluluk | Kişi |
|---|---|---|
| A — Veri & Tespit | preprocess, örnekleme, LLM annotation, damıtma | *(doldurulacak)* |
| B — Doğrulama & Analiz | insan etiketleme, F1/kalibrasyon, mevsimsellik, EDA | *(doldurulacak)* |
| C — Recsys & Deney | RecBole, koşullar, istatistik, pazarlama metrikleri | *(doldurulacak)* |
