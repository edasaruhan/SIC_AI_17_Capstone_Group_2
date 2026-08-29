# CLAUDE.md — Hediye Kontaminasyonu Projesi

> Claude Code bu dosyayı her oturumda otomatik okur. Buradaki kurallar bağlayıcıdır.
> Detaylı gerekçeler için `docs/PROJECT_SPEC.md` ve `docs/ROADMAP.md`.

---

## 1. Proje Nedir

Öneri sistemleri **implicit feedback** varsayımıyla çalışır: satın alma = tercih sinyali.
Bu varsayım **hediye alımlarında yanlıştır** — kullanıcı ürünü kendisi için almamıştır.
Model bunu tercih sanır ve sonraki önerileri kaydırır.

**Yaptığımız:** Amazon review metninden hediye alımlarını LLM ile tespit ediyoruz, yaygınlığını
ölçüyoruz, sonra öneri modelini bu etkileşimlerle ve onlarsız eğitip karşılaştırıyoruz.

**Bu bir araştırma pipeline'ı, bir ürün değil.** Deploy edilmiyor, API'si yok, DB'si yok.
Çıktı: reprodüksiyonu mümkün deney sonuçları + grafikler + capstone raporu.

### Araştırma soruları
- **RQ1** — Hediye alımlarının oranı nedir; kategoriye ve mevsime göre nasıl değişir?
- **RQ2** — Hediye etkileşimlerini çıkarmak, kullanıcının **kendi** sonraki alımını tahmin etmeyi
  iyileştirir mi? Etki, kategorinin hediye yoğunluğuyla orantılı mı?
- **RQ3** — Silmek mi (C1/C2), modele sinyal olarak vermek mi (C3) daha iyi?
- **RQ4** — Bir hediye alımından sonra öneri listesi ne kadar süre kirli kalıyor?

> RQ2 bilinçli olarak "ne kadar iyileşir" değil "duyarlı mı" diye soruluyor.
> **Null result geçerli bir sonuçtur.** Kod ve raporlama buna göre tasarlanmalı —
> "iyileşme çıkmadı" diye parametre kurcalamayın.

---

## 2. Mimari

```
Amazon Reviews 2023 (HF, kategori bazlı)
   │
   ├─ [A] preprocess + 5-core + kronolojik sekans      → data/interim/
   ├─ [B] katmanlı örnekleme (40-60K)                  → data/interim/
   ├─ [C] LLM annotation (Qwen3-4B-Instruct, vLLM)      → data/annotations/
   ├─ [D] insan doğrulama (500, 3 annotator)           → data/annotations/human/
   ├─ [E] ModernBERT distillation                      → models/
   ├─ [F] tam korpus inference                         → data/processed/
   │
   ├─ [G] betimsel analiz + doğrulama (V1–V5)          → reports/figures/
   └─ [H] RecBole deneyi (C0–C4) + pazarlama metrikleri→ reports/results/
```

### Paket yapısı
```
src/gift_contamination/
  config.py              # YAML config yükleme, path çözümleme. Tek doğruluk kaynağı.
  data/
    download.py          # HF'den review + metadata indirme
    metadata.py          # ürün metadata jsonl -> parquet, parent_asin join'i
    preprocess.py        # filtreleme, 5-core, sekans kurma
    sampling.py          # katmanlı örnekleme (main + boost çerçeveleri)
                         #   + deneme setinin LLM kaynağı (--trial-source)
                         #   + Hafta 4 doğrulama seti (--validation)
    labelsheet.py        # elle etiketleme sayfası: CSV <-> xlsx gidiş dönüşü
                         #   körleme: vekil VE LLM'in cevabı sayfaya girmez
                         #   --annotators N ile üç ayrı sayfa (Hafta 4)
  detection/
    schema.py            # Pydantic modelleri (etiket şeması)
    prompting.py         # prompt yükleme/render
    llm_annotate.py      # vLLM batch annotation (veri paralel) + deneme koşusu
    distill.py           # ⛔ YAZILMADI - ModernBERT fine-tune   (Hafta 5)
    inference.py         # ⛔ YAZILMADI - tam korpus inference   (Hafta 5)
  analysis/
    keyword_scan.py      # sözcüksel vekil (ölçü çubuğu, detektör değil)
    precision_check.py   # vekilin elle doğrulanması (T4)
    eda.py               # betimsel analiz, T1-T4 / F1-F8
    deep_eda.py          # varsayım sınama, T5-T14 / F9-F16
    gate1.py             # Kapı 1: dört ölçüt + LLM/vekil aylık eğri (F17)
    viz.py               # ortak grafik stili
    validation.py        # Fleiss kappa (kapı) + sınıf bazlı F1 + vekilin ilk
                         #   bağımsız ölçümü. Hafta 4.
    prevalence.py        # RQ1: kategori bazlı yaygınlık + Wilson GA (F19).
                         #   YALNIZCA `main` çerçevesi; C1 ve C1b tanımlarının
                         #   İKİSİNİ de yazar, hiçbirini seçmez.
  recsys/
    atomic.py            # RecBole .inter üretimi + zaman bazlı leave-one-out.
                         #   Bölme ve evren C0'da DONAR (kural 3'ün sonucu).
                         #   BUGÜN yalnızca sözcüksel vekille koşuyor - `labels`
                         #   parametresi var ama dolduran CLI yolu yok (§7)
    conditions.py        # C0–C4 + C1b (gift+household, sağlamlık kontrolü)
    run_experiment.py    # deney koşucusu — .venv-recbole altında koşar
    marketing_metrics.py # ⛔ YAZILMADI - M1, M2, M3              (Hafta 8)
  utils/
    io.py, logging.py
```

---

## 3. Veri

**Kaynak:** `McAuley-Lab/Amazon-Reviews-2023` (HuggingFace). Açık, ücretsiz.
571.54M review, Mayıs 1996 – Eylül 2023, 33 kategori. Kategori bazlı indirilir.

> ⚠️ **2026-08-25 güncellemesi.** Aşağıdaki tablo hâlâ geçerli ama iki noktada
> ölçümle güncellendi — ayrıntı `docs/DECISIONS.md`:
> `All_Beauty` 5-core sonrası **sıfır etkileşim** bırakıyor, deney pilotu olamaz
> (detektör pilotu olarak kalıyor). RQ2'nin **birincil kategorisi Toys_and_Games**,
> çünkü kontaminasyon yalnızca orada 5-core'dan sağ çıkıyor.

**Kategoriler (üçlü tasarım — değiştirmeyin):**

| Rol | Config anahtarı | HF config adı | Beklenti |
|---|---|---|---|
| Pilot | `pilot` | `raw_review_All_Beauty` | Sadece pipeline testi |
| Yüksek | `high` | `raw_review_Toys_and_Games` | Yüksek hediye oranı |
| Orta | `mid` | `raw_review_Video_Games` | Orta |
| Düşük (kontrol) | `low` | `raw_review_Grocery_and_Gourmet_Food` | Düşük — **kontrol grubu** |

Kullanılan review alanları: `text`, `title`, `rating`, `timestamp`, `user_id`,
`parent_asin`, `verified_purchase`. (`images` hiç okunmaz; `asin` taşınır ama join'de
kullanılmaz.)

**Ürün metadata'sı da indiriliyor** (`raw_meta_*`, 4 kategori ~4.35 GB). Tutulan
alanlar: `parent_asin`, `title`, `main_category`, `store`, `price`, `average_rating`,
`rating_number`, `categories`. Örnekleme `product_title` ve `product_category` olarak
join'lenir ve prompt'a girer — LLM v2'ye kadar *"she loved it"* cümlesindeki "it"in ne
olduğunu bilmiyordu. Join anahtarı **`parent_asin`** — `asin` DEĞİL.

**Depolama:** Ara çıktılar `.parquet`. Ham `.jsonl.gz` `data/raw/` altında, git'e girmez.

---

## 4. Etiket Şeması

`configs/annotation_schema.json` içinde tanımlı, `detection/schema.py` içinde Pydantic karşılığı.

> **v3 — 2026-08-27.** `received` beşinci sınıf oldu; `recipient`'tan `null`
> kaldırıldı. v2'de (2026-08-26) enum'lar deep EDA (T6/T6b) ölçümüyle revize
> edilmişti. Gerekçe `docs/DECISIONS.md`.

```json
{
  "purchase_type": "self | gift_given | household | received | unclear",
  "confidence":    "high | medium | low",
  "recipient":     "child | grandchild | partner | parent | sibling | extended_family | friend | other | unknown",
  "occasion":      "birthday | christmas | wedding | graduation | anniversary | baby_shower | valentines_day | mothers_day | fathers_day | other | none | unknown",
  "evidence_span": "<review içinden birebir alıntı>"
}
```

- `household` = ev halkı için alınmış (bebeğe bez). Hediye değil ama alıcının kendi tercihi de değil.
- **`received` = yorumcu hediyeyi ALDI, vermedi.** v2'de bu `self`'e katlanıyordu.
  Ama `household`'ın var olma gerekçesi *"hediye değil ama kendi tercihi de değil"* —
  hediye alan da tam olarak bu durumda; aynı mantık iki vakaya farklı uygulanıyordu.
  Daha önemlisi: bir kez `self` yazıldıktan sonra bilgi geri gelmez ve C1/C2/C3
  sonradan karar veremez. Annotation anında bilgi yok etmiyoruz.
- **`recipient` her zaman string, `null` yok.** Alıcı yoksa veya belli değilse
  `unknown`. Eskiden ikisi de vardı ve hangisinin ne zaman kullanılacağı yazmıyordu.
  `received` kaydında `recipient` **vereni** gösterir.
- **`occasion`'da `none` ile `unknown` farklı:** `none` = hediye değil, vesile kavramı
  geçersiz. `unknown` = hediye ama vesile metinde yok — beklenen çoğunluk cevabı.
- `confidence` LLM'in öz-beyanıdır. Kullanımı: distillation eğitim setini filtrelemek
  ve hata analizini önceliklendirmek. Kalibrasyonu zayıf olduğu için **kapı değildir**.
- **`grandchild` ayrı bir sınıf ve şemadaki en kritik ayrım.** Torun ayrı hanede yaşar →
  torununa alan kişi **hediye** alır. Kendi çocuğuna alan `household` olabilir. Eski şemada
  ikisi de `child`'a düşüyordu, yani şemanın en zor sınırı için en bilgilendirici ipucu
  kayboluyordu. Toys_and_Games'te adı geçen alıcıların **%26.2'si torun** — bu kozmetik
  bir eksik değil, birincil deney kategorisinin dörtte biri.
- `colleague` kaldırıldı (veride görünmüyor), `friend`'e katıldı. `spouse` → `partner`
  (sevgili/nişanlı da kapsıyor; Video Games'te alıcıların %5.5'i sevgili).
- **`occasion` çoğunlukla `unknown` çıkacak ve bu beklenen davranış.** Vesile metinden
  yalnızca %20–29 oranında çıkarılabiliyor. Modeli vesile uydurmaya zorlamayın.
- `evidence_span` **zorunlu** — halüsinasyonu kısar, insan doğrulamasını hızlandırır.
  Modelden gelen span review metninde birebir geçmiyorsa kayıt `unclear`'a düşürülür ve loglanır.

---

## 5. Deneysel Tasarım — Sapma Yasak

Sequential recommendation, leave-one-out.

### Beş koşul
| Kod | Ne yapar |
|---|---|
| C0 | Baseline — tüm etkileşimler |
| C1 | `gift_given` etkileşimleri eğitimden çıkarılır |
| C2 | Hediye etkileşimleri loss'ta w ∈ {0.25, 0.5, 0.75} ile ağırlıklandırılır |
| C3 | Hediye bayrağı feature/token olarak eklenir (silinmez) |
| C4 | **PLASEBO** — hediye sayısı kadar *rastgele* etkileşim çıkarılır |

### Bu dört kural ihlal edilirse deney geçersizdir

1. **C4 opsiyonel değildir.** C1 kazanıyorsa C4'ten de kazanmak zorunda. Yoksa bulgu gürültüdür.
2. **Test item'ı hediye olamaz.** Değerlendirme yalnızca son etkileşimi `self` olan kullanıcılarda.
3. **5-core filtreleme bir kez, C0 üzerinde uygulanır.** Kullanıcı/item evreni tüm koşullarda
   AYNI kalır. Koşul başına yeniden 5-core uygularsanız koşullar kıyaslanamaz hale gelir.
4. **Zaman bazlı split.** Rastgele split sızıntı üretir.

### Modeller (RecBole)
`Pop`, `ItemKNN`, `BPR`, `SASRec` (birincil), `GRU4Rec`.
Öncelik sırası: SASRec + BPR önce, diğerleri zaman kalırsa.

### Metrikler
- Teknik: Recall@10, NDCG@10, HR@10 + bootstrap CI + 3–5 seed
- Pazarlama: **M1** retargeting israf oranı, **M2** kontaminasyon yarı ömrü, **M3** segment analizi

---

## 6. Teknoloji Kararları (verilmiş — tartışmayın, uygulayın)

| Katman | Seçim | Not |
|---|---|---|
| Büyük veri işleme | **polars** | pandas 10M+ satırda kullanılmayacak |
| LLM servis | **vLLM 0.28** (`--dtype float16`, **`enable_prefix_caching=True`**, `max_model_len=4096`, iki T4 veri-paralel) | offline batch, Kaggle Linux notebook'ta. Yerelde sadece smoke test için llama.cpp/GGUF — Windows'ta vLLM yok, 4 GB'a model sığmıyor. **Prefix caching pazarlık konusu değil** — aşağıya bakın |
| Annotator LLM | **Qwen/Qwen3-4B-Instruct-2507** birincil, **google/gemma-4-E4B** ikincil (5K alt örneklem) | Seçim VRAM'e değil **GPU kuşağına** bağlı: elimizdeki her GPU pre-Ampere (Turing sm75), bfloat16 ve FlashAttention yok. Qwen3.5 ailesi hibrit GDN + VL, Turing'de pratikte koşmuyor. Gerekçe: technology-review §4.2 |
| Structured output | vLLM guided decoding (JSON schema) | serbest metin parse edilmeyecek |
| Distillation | **ModernBERT-base** | `AutoModelForSequenceClassification` |
| Recsys | **RecBole 1.2.0** — **ayrı venv** (`.venv-recbole`, `requirements-recbole.txt`) | kendi implementasyonumuzu yazmıyoruz. RecBole numpy 1.x dönemine ait: ana ortamın numpy 2.5 / pandas 3.0 yığınında ilk satırda çöküyor (`np.float_`). Ana ortama kurmak numpy/scipy/sklearn'i geri çekip polars/statsmodels tarafını kırardı. İki taraf `.inter` dosyasıyla konuşuyor, ayırmanın bedeli yok. Gerekçe: `docs/DECISIONS.md` 2026-08-29 |
| Deney takibi | **Weights & Biases** | her run config + seed + metrik loglar |
| Config | **YAML** (`configs/`) | kodda hardcode path/parametre yok |
| Test | **pytest** | |

**Prefix caching neden zorunlu.** Ölçüldü (2026-08-27): SYSTEM prompt'u `v3`'te
**~2.450 token**, review medyanı **~30 token**. Yani her satırın prefill'inin **%99'u
aynı**. Caching kapalıysa 47.200 satırda ~**116M gereksiz prefill token** üretilir ve
Kaggle kotası boşa gider.

> ✅ **Uygulandı 2026-08-28.** Anahtarlar `configs/base.yaml` → `detection:` altında ve
> `detection/llm_annotate.py` tarafından okunuyor. Ölü anahtar bırakılmadı.

- `LLM(..., enable_prefix_caching=True, max_model_len=4096)` — 4096 = 2.277 (system,
  ölçüldü) + review (≤6.000 karakter ≈ 1.500 token) + 220 çıktı. Fazlası KV cache'i
  şişirir ve eşzamanlı istek sayısını düşürür.
- Chat template'i **vLLM'e değil tokenizer'a** uygulatıyoruz (`llm.generate(list[str])`).
  Sebebi: caching'in çalışması için paylaşılan önek her satırda **bayt bayt aynı** olmalı
  ve `llm.chat()` sürümler arası değişken davranıyor. Gerçekleşen önek uzunluğu
  koşu raporuna yazılıyor (`shared_prefix_chars`) — varsayılmıyor. Kaggle koşusunda
  ölçüldü: **9.859 karakter = 2.277 token**.
- Koşu **gerçekleşen throughput'u loglar** ve **kurulumu üretimden ayırır**. Ayırmak
  şart: 200 satırlık duman testinde motor kurulumu (model indirme + `torch.compile` +
  CUDA graph capture) 252 sn'nin ~181'ini yiyip toplam hızı 0,79 satır/sn gösteriyordu,
  oysa üretim 2,8 satır/sn gidiyordu. Küçükten büyüğe uzatma **yalnızca**
  `rows_per_s_generating` ile yapılır.

- Review metni `str.format` ile değil düz `replace` ile yerleştirilir: gerçek bir review'da
  `{LOOSE}` geçiyor ve `format` bunu yer tutucu sanıp koşunun ortasında patlıyordu.

> ✅ **Ölçüldü 2026-08-28** (Kaggle 2× T4, Toys). Buradaki sayılar artık tahmin değil.
> Sağdaki sütun **11.800 satırlık tam koşu**; soldaki 200 satırlık duman testi:
>
> | | 200 satır | **11.800 satır (tam)** |
> |---|---|---|
> | SYSTEM prompt'u | 2.277 token | **2.277 token** (9.859 karakter) — tahmin 2.450'ydi |
> | Üretim | 2,8 satır/sn | **3,64 satır/sn** · 1,82 satır/sn/GPU |
> | Çıktı | 133 token/sn | **159 token/sn** — tahmin 1.000–2.000'di, **6–13× iyimserdi** |
> | Motor kurulumu | ~181 sn | **196 sn**, satır sayısından bağımsız tek seferlik |
> | parse hatası | %0 | **%0,03** (4 satır) · span düşürme **%2,77** |
>
> Küçük koşu **%30 karamsar** çıktı: prefix cache ve batch doluluğu uzun koşuda
> oturuyor. Kurulum ayrı raporlanmasa 200 satırlık test 0,79 satır/sn gösterecekti —
> gerçeğin 4,6 katı yanlış.
>
> **Gerçekleşen:** Toys 11.800 satır = **57 dk** (tahmin 73'tü). Dört kategori
> (47.200) ≈ **3,8 saat**, Kaggle'ın 30 sa/hafta kotasının %13'ü. Darboğaz T4'te
> decode: FlashAttention yok (compute capability 7.5 → `TRITON_ATTN`), KV cache
> 4,45 GiB → aynı anda ~8 istek.


**Kaggle iki T4 veriyor — GPU isteyen her adım ikisini de kullanmalı.**
Kural: **veri paralel, tensor paralel değil.** Model tek karta sığdığı sürece
(Qwen3-4B fp16 ~8 GB / 16 GB) modeli bölmenin tek yaptığı şey PCIe üzerinden
all-reduce maliyeti eklemek — Kaggle T4'lerinde NVLink yok. İki bağımsız süreç
birbiriyle hiç konuşmaz, her birinin kendi **tam** prefix cache'i olur, hızlanma
~2×. Kaggle kotası **oturum saati** olarak sayıldığı için bu kotayı da yarıya
indirir.

Desen `detection/llm_annotate.py`'de kurulu; sonraki modüller onu tekrar kullansın:

- İş, `row_id`'ye göre sıralanıp `[worker::n_workers]` ile bölünür. Bölme
  **yeniden başlatmada da aynı** olmak zorunda — değişirse bir işçi diğerinin
  yarım bıraktığı satırları asla görmez ve koşu hiç bitmez.
- Her işçi kendi parçalarını yazar (`part_w{worker}_*.parquet`), ebeveyn
  birleştirir ve **girdi = çıktı** satır kontrolünü yapar. Bir işçi çökerse
  eksik çıktı yazılmaz, hata verilir.
- Throughput **duvar saatinden** raporlanır: `rows_per_s` toplam hız,
  `rows_per_s_per_gpu` ayrıca. İkisini karıştırmak 2× hızlanmayı görünmez yapar.
- Testle kilitli: iki işçinin çıktısı tek işçininkiyle **birebir aynı**
  (`test_two_workers_produce_the_same_output_as_one`). Paralellik sonucu
  değiştirirse bu bir optimizasyon değil, sessiz bir veri hatasıdır.

`tensor_parallel_size` config'te duruyor ama **1**; yalnızca model tek karta
sığmazsa (14B+) gerekir ve o zaman `gpus` düşürülür — kod
`gpus × tensor_parallel_size ≤ mevcut kart` kontrolünü yapıyor.

`distill.py` (ModernBERT eğitimi) için karşılığı DDP; `inference.py` (4,97M
satır) için yine veri paralel, aynı parça deseni.

**Inference kapsamı: önce `kcore`, sonra `clean`.** RQ2–RQ4 yalnızca k-core korpusuna
etiket istiyor (**4,97M satır**, ~3–5 saat yerel GPU). `clean` (27M, ~15–25 saat) RQ1'in
betimsel eğrilerini keskinleştiriyor ama zorunlu değil — RQ1 `main` çerçevesinden güven
aralığıyla zaten cevaplanabiliyor. Ters sırada deney 20 saat boşuna bekler.

**Bağımlılık uyarısı — ölçüldü, varsayım değil.** RecBole 1.2.0 `np.float_`
kullanıyor; numpy 2.0 o adı kaldırdı ve ana ortam numpy 2.5 üzerinde. Çözmeye
çalışmayın, **ayırın**: RecBole `.venv-recbole` altına `requirements-recbole.txt`
ile kurulur (pin gerekçeleri dosyanın içinde). İki ortam yalnızca `.inter`
dosyaları üzerinden konuşur, ortak kod import etmez.

LLM tarafı yerelde koşmuyor: Kaggle'da, `scripts/kaggle_annotate.py` kendi
ortamını kurarak. `requirements-llm.txt` yalnızca yerel GPU'su olan biri denemek
isterse duruyor.

---

## 7. Komut Arayüzü

Her adım tek başına yeniden çalıştırılabilir; çıktı varsa `--force` olmadan
yeniden hesaplanmaz.

> **Aşağıdaki komutların hepsi BUGÜN çalışır.** Henüz yazılmamış olanlar en
> altta, ayrı bir blokta ve öyle işaretli. Karışık liste tutmak, hangi adımın
> gerçekten koşabildiğini belirsiz bırakıyordu (denetim, 2026-08-29).

```bash
python -m gift_contamination.data.download      --config configs/base.yaml --category pilot
python -m gift_contamination.data.preprocess    --config configs/base.yaml --category pilot
python -m gift_contamination.analysis.keyword_scan --config configs/base.yaml --category pilot
# Örneklem boyutu `sampling.n_annotate` ile config'ten gelir - CLI bayrağı YOK.
python -m gift_contamination.data.sampling      --config configs/base.yaml --category all
python -m gift_contamination.analysis.eda       --config configs/base.yaml
python -m gift_contamination.analysis.deep_eda  --config configs/base.yaml
python -m gift_contamination.detection.llm_annotate --config configs/base.yaml --category high
# Kapı 1'in 4. ölçütü: deneme satırları örneğin İÇİNDE DEĞİL, ayrıca etiketlenir
python -m gift_contamination.data.sampling          --config configs/base.yaml --trial-source 200
python -m gift_contamination.detection.llm_annotate --config configs/base.yaml --trial 200
python -m gift_contamination.analysis.gate1         --config configs/base.yaml --category high
# RQ1 tablosu - dört kategori etiketlendikten SONRA
python -m gift_contamination.analysis.prevalence    --config configs/base.yaml --category all
# Hafta 4 - insan doğrulaması. Etiketleyen kişi LLM'in cevabını GÖRMEZ.
python -m gift_contamination.data.sampling   --config configs/base.yaml --validation
python -m gift_contamination.data.labelsheet --validation --export --annotators 3
#   ... üç kişi kendi xlsx'ini doldurur ...
python -m gift_contamination.data.labelsheet --validation --ingest --annotators 3
python -m gift_contamination.analysis.validation  --config configs/base.yaml
# Hafta 6 — deney. Atomic dosya ve koşullar RecBole GEREKTİRMEZ; yalnızca
# run_experiment gerektirir. Etiket verilmezse sözcüksel vekil kullanılır ve
# çıktı `reportable: false` damgası taşır — sonuç tablosuna giremez.
python -m gift_contamination.recsys.atomic     --config configs/base.yaml --category mid
python -m gift_contamination.recsys.conditions --config configs/base.yaml --category mid --condition all
python -m gift_contamination.recsys.run_experiment --config configs/base.yaml \
       --category high --condition C0 --model SASRec --seed 42
```

**Henüz YAZILMADI — hedef sözleşme.** Aşağıdakiler çalışmaz; modülleri yok.
Bir komutu buradan yukarıdaki bloğa taşımak, o modülün testleriyle birlikte
geldiği anlamına gelir.

```bash
# Hafta 5 — damıtma ve tam korpus çıkarımı
python -m gift_contamination.detection.distill   --config configs/base.yaml
python -m gift_contamination.detection.inference --config configs/base.yaml --category high
# Hafta 8 — pazarlama metrikleri (M1/M2/M3)
python -m gift_contamination.recsys.marketing_metrics --config configs/base.yaml
```

> **`recsys.atomic` bugün YALNIZCA sözcüksel vekille koşuyor.** `build_atomic`
> bir `labels` parametresi taşıyor ama onu dolduran bir CLI yolu yok ve hiçbir
> çağıran vermiyor — çıktı bu yüzden `label_source: proxy` damgalı ve
> `reportable: false`. Gerçek etiketler `detection.inference` tam korpusu
> etiketledikten sonra gelecek: `build_atomic` kcore'un **her** satırı için
> etiket bekliyor (satır sayısı tutmazsa gürültülü hata veriyor), yani 11.800
> satırlık annotation örneği tek başına yetmez. Zincir: Hafta 4 doğrulama →
> Hafta 5 damıtma + çıkarım → Hafta 6 gerçek deney.

Kurallar:
- Her komut **idempotent** olmalı; çıktı varsa `--force` olmadan yeniden hesaplamamalı.
- Her komut çıktı yolunu ve satır sayısını loglamalı.
- Uzun süren her adım `tqdm` ile ilerleme göstermeli.

---

## 8. YAPILMAYACAKLAR

1. **Veri dosyası commit etmeyin.** `data/`, `models/`, `*.parquet`, `*.jsonl.gz` gitignore'da.
2. **Kendi recommender'ınızı yazmayın.** RecBole var.
3. **LLM çıktısını regex ile parse etmeyin.** Guided decoding + Pydantic doğrulama.
4. **LLM'i tüm korpusa koşturmayın.** 40–60K annotate → ModernBERT distill → tam inference.
5. **Rastgele train/test split kullanmayın.** Zaman bazlı.
6. **Test setinde hiperparametre tuning yapmayın.** Validation split ayrı.
7. **C4'ü atlamayın.**
8. **`asin` ile metadata join etmeyin.** `parent_asin`.
9. **Sonuç beğenilmedi diye koşulları/filtreleri değiştirmeyin.** Değişiklik gerekiyorsa
   `docs/DECISIONS.md`'ye tarih ve gerekçe yazılır.
10. **pandas ile 10M+ satır okumayın.**
11. **Hardcode path yazmayın.** Her şey config üzerinden.
12. **Seed sabitlemeden deney koşmayın.** Yöntem: `seed` config'ten okunur ve
    **açıkça** geçirilir (`sample(seed=...)`, `default_rng(seed)`,
    RecBole `init_seed`). Katman/koşul başına türetilen seed'ler
    `zlib.crc32` ile üretilir — `hash()` DEĞİL: `hash()` `PYTHONHASHSEED`
    ile süreçten sürece değişir ve "seed 42 ile yeniden üretilebilir"
    iddiasını sessizce yalanlar (`sampling._stratum_seed`,
    `conditions._seed_for`). Global `random.seed()` çağıran bir yardımcı
    **yok**; süreç içinde `os.environ["PYTHONHASHSEED"]` yazmak da işe
    yaramaz — o değişken yorumlayıcı başlamadan önce okunur.

---

## 9. Bilinen Edge Case'ler

| Durum | Ne yapılacak |
|---|---|
| **C1'de item'ın tüm etkileşimleri silinir** | Hediye etkileşimleri çıkarılınca bazı item'lar eğitimden tamamen kaybolur ve eval'de cold item olur. Item evreni C0'da sabitlenir; kaybolan item'lar loglanır ve raporlanır. **Bu sessizce geçilecek bir detay değil — C1 vs C0 farkının bir kısmını açıklayabilir.** |
| `timestamp` birimi | Saniye mi ms mi kontrol edilecek; normalize edilip UTC'ye çevrilecek |
| Boş / çok kısa `text` | < 5 kelime → annotation dışı, `unclear` |
| Aynı user+item birden çok review | Deduplicate; en erken timestamp tutulur |
| Tüm sekansı hediye olan kullanıcı | Eval'den çıkarılır (test item'ı `self` olmalı kuralı gereği) |
| `evidence_span` metinde yok | `unclear`'a düşür, sayacı artır, oranı raporla |
| Review tarihi ≠ satın alma tarihi | Mevsimsellik tepeleri gecikmeli olacak. **Beklenen davranış, bug değil.** |
| LLM JSON üretemedi | 1 kez retry, sonra `unclear` + logla. Sessiz atlamak yok. |

---

## 10. Testler (pytest)

Ürün kodu testi değil, **veri bütünlüğü ve deney geçerliliği** testi yazıyoruz:

- `test_schema.py` — Pydantic şeması geçerli/geçersiz JSON'ları doğru ayırıyor mu
- `test_preprocess.py` — 5-core gerçekten 5-core mu; sekanslar kronolojik mi
- `test_conditions.py` — **kritik**: hiçbir koşul C0'ın evrenine kullanıcı/ürün
  EKLEMİYOR mu (tek yönlü kapsama); C4 tam olarak C1 kadar etkileşim mi çıkarıyor
- `test_no_leakage.py` — **kritik**: bölme zaman bazlı mı; test item'ı eğitimde
  geçmiyor mu; test item'ları `self` etiketli mi
- `test_validation.py` — Fleiss κ elle hesaplanmış örneğe eşit mi; `n < 20` sınıf
  genel κ'ya girmiyor mu; beraberlik `tie` olarak mı işaretleniyor
- `test_prevalence.py` — Wilson GA elle hesaplanmış aralığa eşit mi; yaygınlık
  YALNIZCA `main`'den mi okunuyor; iki tanımdan biri sessizce seçilmiyor mu;
  `human_validated` diskten mi okunuyor (elle iddia edilmiyor)

Küçük sentetik fixture'lar `tests/fixtures/` altında. Gerçek veri testte kullanılmaz.

---

## 11. MVP Kapsamı

**MVP (bu sırayla):**
1. Pilot kategoride uçtan uca pipeline (indir → preprocess → örnekle → LLM annotate → oran + mevsimsellik grafiği)
2. İnsan doğrulama akışı + kappa/F1 hesabı
3. ModernBERT distillation + tam inference
4. `high` ve `low` kategorilerde C0 + C1 + C4, SASRec + BPR
5. M1 + M2 pazarlama metrikleri

**MVP DIŞI (zaman kalırsa):**
- `mid` kategori, C2 ve C3 koşulları, GRU4Rec/ItemKNN/Pop
- Gemma 4 E4B ile ikinci annotation ve model-arası uyum (5K alt örneklem)
- İkinci plasebo (etiket shuffle)
- Streamlit annotation arayüzü (başlangıçta CSV + Google Sheets yeterli)
- M3 segment analizi

---

## 12. Başarı Kriterleri

**Pipeline başarılı sayılır eğer:**
- Pilot kategoride uçtan uca hatasız çalışıyorsa
- Aylık hediye oranı grafiğinde **Aralık–Ocak** tepesi görünüyorsa (V2 — sözcüksel vekille ölçüldü, dört kategoride de var; review tarihi satın alma tarihinin gerisinde kaldığı için tepe Kasım değil Aralık–Ocak)
- Kategori sıralaması Toys > Video Games > Grocery çıkıyorsa (V3)
- LLM ile insan etiketi arasında sınıf bazlı F1 raporlanabiliyorsa (V1)
- C0–C4 arası user/item evreni aynı olduğu testle doğrulanıyorsa

**Araştırma sonucu başarılı sayılır eğer:** RQ1–RQ4 cevaplanmışsa.
Cevabın yönü (pozitif/null) başarı kriteri **değildir**.

---

## 13. TBD — Karar Verilmesi Gerekenler

Bunlar henüz kararlaştırılmadı. Claude Code bunları **kendi kafasına göre doldurmasın**,
karşılaşınca sorsun veya `docs/DECISIONS.md`'ye "varsayıldı" notuyla yazsın.

- [ ] **TBD** Örnekleme boyutu kesin sayı: 40K mı 60K mı (pilot sonucuna göre)
- [ ] **TBD** C2'de kullanılacak nihai ağırlık(lar) — üçünü de mi koşacağız yoksa biri mi
- [ ] **TBD** C3'ün RecBole'da nasıl implement edileceği (feature olarak mı, ayrı token mı)
- [ ] **TBD** **`household` C1'e girecek mi — KAVRAMSAL karar, Hafta 6 öncesi.**
  Kayıt eskiden "kappa sonucuna bağlı" diyordu; 2026-08-29'da bağımlılık değişti.
  κ sınıfın *ayırt edilebilir* olduğunu söyler, *kontaminasyon sayılacağını* değil.
  Projenin kendi tanımı *"alan kişi ürünü kendisi için seçmedi"* — bu tanıma göre
  kendi çocuğuna alınan oyuncak da kontaminasyondur.
  **Ölçüldü (2026-08-29):** `household` `main` çerçevesinin **%20,17'si**. C1 ya
  %23,3 ya %43,4 etkileşim çıkarıyor — iki kat fark. İki sınıfın zaman imzası ayrı:
  `gift_given` 1,58× mevsimsel, `household` **1,09×** yani düz. Sınırı çizen şey
  vesile (`recipient=child`'da `household`'un %99'u `occasion=none`).
  **Kod ikisini de koşabilir olacak:** C1 (yalnızca `gift_given`) birincil,
  C1b (+`household`) sağlamlık kontrolü. Aynı kod yolu, farklı süzgeç — karar
  sonuca göre değil, sonuç her iki tanım altında raporlanır
- [ ] **TBD** **`received` sınıfının C1'de silinip silinmeyeceği.** Şema v3 (2026-08-27)
  beşinci sınıfı ekledi ama C1'in tanımı hâlâ yalnızca `gift_given` diyor. Aynı soru,
  aynı gerekçe: alan kişi ürünü kullanıyor ama seçmedi. `household` ile birlikte
  karara bağlanmalı
- [x] ~~**TBD** κ eşiği 5 sınıfta hâlâ 0.60 mı?~~ → **KARARLAŞTI 2026-08-29.**
  **0,60'ta kaldı.** κ sınıf sayısıyla düşme eğiliminde diye eşiği peşinen düşürmek,
  kapıyı koşudan önce gevşetmenin başka bir biçimi olurdu. Bunun yerine iki kural,
  ikisi de sonuç görülmeden yazıldı: sınıf bazlı κ ayrıca raporlanır, ve `n < 20`
  olan sınıf genel κ'ya **katılmaz**, ayrıca listelenir
  (`validation.min_class_n_for_kappa`). Öngörülen vaka `received`. Gerekçe: DECISIONS
- [x] ~~**TBD** `confidence` alanı ne işe yarayacak?~~ → **KARARLAŞTI 2026-08-29.**
  **Prompt v3'te kalındı, alan analizden düşürüldü.** `low` ile `unclear` birebir
  örtüşüyor (870/870, iki yönde %100) — alan bağımsız bilgi taşımıyor. Kök neden:
  prompt bir *kural* değil yalnızca *örnek* veriyor. v4 yazmak ~4 saat Kaggle ve
  Kapı 1'in yeniden koşulması demekti; alanın tek amacı (sınıf dengesi) zaten
  `boost` çerçevesiyle karşılanıyor. **Limitasyon olarak raporlanacak**, sessizce
  düşürülmeyecek. Gerekçe: DECISIONS
- [x] ~~**TBD** Kapı 1'in sayısal geçme ölçütü~~ → **KARARLAŞTI 2026-08-28.** Dört
  ölçüt, `configs/base.yaml` → `gate1:` altında, koşudan önce sabitlendi.
  Gerekçe: `docs/DECISIONS.md`. **Sonuç 2026-08-29: PASS (4/4)** — dördü de kendi
  eşiğini geçti, hiçbiri sonradan değiştirilmedi
- [ ] **TBD** Kaç seed (3 mü 5 mi) — koşu süresine göre
- [x] ~~**TBD** GPU: yerel RTX mi Kaggle mı~~ → **KARARLAŞTI 2026-08-26.** Yerel kart
  GTX 1650 Ti (4 GB, Turing). LLM annotation **Kaggle**'da (2× T4, ~30 sa/hafta);
  preprocess, distillation, RecBole deneyleri **yerelde**. Altı aşamadan yalnızca
  biri kotaya bağlı. Ayrıntı: implementation-plan §1.6
- [ ] **TBD** M2 "kontaminasyon yarı ömrü" için kesin operasyonel tanım
- [ ] **TBD** Hangi ekip üyesi hangi kulvarda (bkz. `docs/PROJECT_SPEC.md` §11)

---

## 14. Çalışma Tarzı

- **Küçük adım, çalışan kod.** Her adım kendi başına koşabilmeli ve çıktı üretmeli.
- **Önce pilot.** `All_Beauty` üzerinde çalışmayan hiçbir şey büyük kategoriye taşınmaz.
  Ama dikkat: `All_Beauty` **sadece pipeline pilotu**. 5-core sonrası sıfır satır bıraktığı
  için üzerinde recsys deneyi koşulamaz — deney pilotu `Toys_and_Games`.
- **Config-driven.** Yeni parametre gerekiyorsa YAML'a ekle, koda gömme.
- **Emin değilsen sor.** Özellikle §5'teki deneysel kurallarla ilgili bir tavizin gerekiyorsa
  sessizce yapma — sor.
- **Kararları yaz.** Spec'ten sapma varsa `docs/DECISIONS.md`'ye tarih + gerekçe.
