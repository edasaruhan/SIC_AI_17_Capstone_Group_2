# Karar Kaydı

Spec'ten (`CLAUDE.md`, `docs/PROJECT_SPEC.md`) her sapma buraya yazılır.
Format: tarih · karar · gerekçe · kim.

Boş bırakmak yerine "değişiklik yok" yazmak da geçerlidir.

---

## Şablon

### YYYY-MM-DD — [Kısa başlık]
**Karar:**
**Gerekçe:**
**Etkilediği bölüm:**
**Kim:**

---

### 2026-08-24 — Repo iskeleti kuruldu
**Karar:** Docker, CI/CD, veritabanı ve API katmanı kapsam dışı bırakıldı.
**Gerekçe:** Proje bir araştırma pipeline'ı; deploy edilecek bir servis yok.
Docker özellikle GPU ortamında kurulum sürtünmesi ekliyor, 3 kişilik ekipte karşılığı yok.
**Etkilediği bölüm:** Repo yapısı
**Kim:** Ekip

### 2026-08-25 — HF veri erişimi `hf_hub_download` ile yapılacak
**Karar:** `datasets.load_dataset(..., trust_remote_code=True)` terk edildi. Ham `.jsonl`
dosyası `huggingface_hub.hf_hub_download` ile doğrudan çekiliyor, polars `scan_ndjson` ile
lazy okunuyor.
**Gerekçe:** `datasets` 4.0 `trust_remote_code` parametresini, 4.5 ise loading-script
desteğini tamamen kaldırdı. McAuley-Lab reposu script tabanlı olduğu için `README.md`,
`docs/PROJECT_SPEC.md §16` ve `docs/ROADMAP.md §13`'teki örnek çağrı bugün `RuntimeError`
veriyor. Doğrudan indirme ayrıca daha hızlı, kesintiden devam ediyor ve idempotency'yi
dosyanın varlığından ücretsiz alıyor.
**Etkilediği bölüm:** `data/download.py`, `requirements.txt` (+`huggingface_hub`).
README ve spec'teki quickstart bloğu güncellenmeli.
**Kim:** Ekip

---

### 2026-08-25 — İki ayrı korpus: `_clean` ve `_kcore`
**Karar:** `preprocess` artık iki parquet üretiyor.
`<slug>_clean.parquet` = verified + min_words + dedup (k-core YOK) — LLM annotation ve
betimsel analizin (RQ1) korpusu. `<slug>_kcore.parquet` = clean + iteratif 5-core —
RecBole deneyinin korpusu.
**Gerekçe:** Amazon Reviews 2023'ün review grafiği beklenenden çok daha seyrek. Pilot
kategoride (All_Beauty) dedup sonrası 537.261 etkileşim / 503.388 kullanıcı = kullanıcı
başına 1.07; kullanıcıların yalnızca **%0.08'i** (382 kişi) 5+ etkileşime sahip. İteratif
5-core bu kategoride 3 iterasyonda TÜM satırları eliyor. Hediye oranı review'lar hakkında
bir soru olduğu için betimsel analizi recsys alt-grafiğinin elemesine tabi tutmak yanlış
olurdu. CLAUDE.md §5 kural 3 (evren C0'da bir kez donar) ihlal edilmiyor — o kural deney
korpusu için geçerli ve `_kcore` üzerinde aynen uygulanıyor.
**Etkilediği bölüm:** `data/preprocess.py`, tüm alt analizler.
**Açık kalan:** `high`/`mid`/`low` kategorilerinde 5-core sonrası ne kaldığı ölçülecek;
recsys deneyi için k düşürmek veya McAuley Lab'in hazır 5-core benchmark split'lerini
kullanmak gerekebilir. Hafta 6 öncesi karara bağlanmalı.
**Kim:** Ekip

---

### 2026-08-25 — Yeni analiz modülleri
**Karar:** `analysis/` altına CLAUDE.md §2'de listelenmeyen dört modül eklendi:
`keyword_scan.py`, `precision_check.py`, `eda.py`, `viz.py`.
**Gerekçe:** Data Research teslimi betimsel istatistik ve görselleştirme istiyor;
`descriptive.py`/`validation.py` ikilisi bu kapsamı karşılamıyor. `keyword_scan.py`
zaten `docs/PROJECT_SPEC.md §16`'da 1. hafta karar aracı olarak tanımlıydı.
**Etkilediği bölüm:** Paket yapısı. CLAUDE.md §2 güncellenmeli.
**Kim:** Ekip

---

### 2026-08-25 — İnsan etiketleme dosyaları git'e girmeyecek
**Karar:** `.gitignore`'daki `!data/annotations/human/*.csv` istisnası kaldırıldı;
`data/annotations/human/` tamamen yok sayılıyor. Git'e yalnızca toplulaştırılmış sonuç
(`reports/results/keyword_precision.json`) giriyor.
**Gerekçe:** Bu dosyalar birebir review metni taşıyor. `concept-note` ve
`implementation-plan §5.1` açıkça taahhüt ediyor: *"No verbatim review text appears in any
published output"* ve *"`evidence_span` values are excluded from repository commits"*.
Eski istisna bu taahhütle doğrudan çelişiyordu.
**Etkilediği bölüm:** `.gitignore`, `analysis/precision_check.py`
**Kim:** Ekip

---

### 2026-08-25 — Ham veri OneDrive dışına yönlendirildi
**Karar:** `repo/data/raw` yolu korundu ama dizin `D:\amazon-reviews-2023`'e bir Windows
junction. Config, kod ve `.gitignore` değişmedi.
**Gerekçe:** Proje dizini OneDrive senkronizasyonu altında; 16.3 GB ham veriyi oraya
indirmek senkronizasyon fırtınası ve kota aşımı demek. OneDrive junction'ları takip etmez.
**Etkilediği bölüm:** Yalnızca yerel kurulum. Ekibin diğer üyeleri kendi makinelerinde
aynı junction'ı kurmalı veya `paths.raw`'a mutlak bir yol vermeli.
**Kim:** Ekip

---

### 2026-08-25 — Çekirdek `.venv` ortamı
**Karar:** Data Research aşaması için `repo/.venv` oluşturuldu ve yalnızca
`requirements.txt` kuruldu. `venv-llm` ve `venv-recsys` ihtiyaç anında (Hafta 3 ve 6).
**Gerekçe:** Bu teslim torch/vLLM/RecBole gerektirmiyor; ağır bağımlılıkları şimdiden
kurmak kurulum sürtünmesi ekliyor. `requirements-llm.txt` zaten `-r requirements.txt`
ile katmanlanıyor, sonradan üzerine kurulabilir.
**Etkilediği bölüm:** `README.md` kurulum bölümü güncellenmeli.
**Kim:** Ekip

---

### 2026-08-25 — T4 precision kontrolü ön geçiş olarak işaretlendi
**Karar:** 100 review'luk keyword precision örneklemi (T4) **yazar destekli ön geçiş**
olarak etiketlendi; teslimde bu açıkça belirtiliyor. 3 annotator'lı insan doğrulaması ve
Fleiss' kappa (V1) `implementation-plan` Hafta 4 planında olduğu gibi duruyor.
**Gerekçe:** Data Research'ün "neden LLM gerekiyor" argümanı sayısal bir precision
tahmini gerektiriyordu. Ekip aynı CSV'yi yeniden etiketleyip
`precision_check score` ile sonucu tazeleyebilir; araç bunun için hazır.
**Etkilediği bölüm:** `data-research/data-research.md` §4, `reports/results/keyword_precision.json`
**Kim:** Ekip

---

### 2026-08-25 — Açık kalan: `distill.max_length`
**Karar:** Değişiklik YOK, not düşülüyor. `configs/base.yaml` `distill.max_length: 1024`
diyor; oysa ModernBERT seçiminin gerekçesi üç dokümanda da 8192 token context
("alıcı bilgisi review'un sonunda geçiyor, 512'de kesilir").
**Gerekçe:** Bu teslimin kapsamı dışında ama sessiz geçilmemeli. F2 (metin uzunluğu
dağılımı) kaç review'un 1024 token'ı aştığını ölçüyor; karar o sayıya bakılarak
Hafta 5 öncesi verilmeli.
**Etkilediği bölüm:** `configs/base.yaml`, `detection/distill.py` (henüz yazılmadı)
**Kim:** Ekip

---

### 2026-08-25 — ÇÖZÜLDÜ: `distill.max_length` ve ModernBERT context gerekçesi
**Karar:** `distill.max_length: 1024` **değiştirilmiyor**; ModernBERT'in 8192 token
context'ine dayanan gerekçe ise **geçersiz sayılıyor** ve teknoloji gerekçesi
"çıkarım verimliliği" üzerinden yeniden yazılmalı.
**Gerekçe:** Ölçüldü (T5, `reports/results/deep_eda_tables.md`). Hediye kanıtının
review gövdesindeki medyan göreli konumu **0.011–0.031** — yani ilk cümlede.
**512 token'da kesmek** (yani düz BERT) kanıtı hediye review'larının yalnızca
**%0.013–0.044'ünde** kaybettiriyor; 1024 token'da kayıp en fazla %0.008.
Ayrıca review'ların yalnızca %0.03–0.45'i 768 kelimeyi aşıyor (T2).
concept-note §4.2, technology-review §4.4 ve implementation-plan §1.3'te geçen
"alıcı bilgisi review'un sonunda geçer, 512'de kesilir" ifadesi bu korpus için
**yanlış** ve düzeltilmeli.
**Uyarı:** Bu ölçüm sözcüksel desenin ilk eşleşme konumudur; desenler review
açılışına yanlı olabilir. Ama fark üç büyüklük mertebesi, sonuç bu uyarıya bağlı değil.
**Etkilediği bölüm:** `configs/base.yaml`, technology-review, concept-note, implementation-plan
**Kim:** Ekip

---

### 2026-08-25 — Annotation şeması revize edilmeli (LLM koşusundan ÖNCE)
**Karar:** `configs/annotation_schema.json` içindeki `recipient` enum'ı
güncellenecek: **`grandchild` ve `sibling` eklenecek**, `colleague` ise `friend`
içine katlanacak.
**Gerekçe:** Ölçüldü (T6). Toys_and_Games'te alıcıların **%26.2'si torun**
(grandson %14.7 + granddaughter %11.5) ve mevcut şemada `child`'a temiz eşlenmiyor.
Kardeşler tutarlı biçimde görünüyor (Video_Games'te brother %3.6, Grocery'de
sister %5.5) ama karşılığı yok. `colleague` ise hiçbir kategoride kayda değer
görünmüyor. Şema sezgiyle tasarlanmıştı; bu onun veriyle ilk teması.
**Etkilediği bölüm:** `configs/annotation_schema.json`, `prompts/gift_detection_v1.md`,
CLAUDE.md §4
**Kim:** Ekip

---

### 2026-08-25 — RQ2'nin birincil kategorisi Toys_and_Games olacak
**Karar:** Recsys deneyinin **birincil** kategorisi Toys_and_Games; Video_Games
yalnızca hızlı iterasyon pilotu. `All_Beauty` deney dışı (5-core sonrası sıfır).
**Gerekçe:** Ölçüldü (T8). 5-core filtresi hediye alıcılarını **sistematik olarak
eliyor**: Video_Games'te hediye oranı %4.40 → %2.68 (**−%39.2**), Grocery'de
%1.85 → %1.34 (**−%27.5**). Toys'ta ise korunuyor (%11.07 → %11.27, +%1.8).
Mekanizma açık: hediye alımı çoğu kez tek seferliktir ve tek seferlik yorumcular
tam da k-core'un sildiği kullanıcılardır. Video_Games ve Grocery'de deney,
ölçtüğümüzden %27–39 daha temiz bir korpusta koşacak ve **null sonuca doğru
yanlı** olacak. Bu, plandaki "yüksek/orta/düşük" seçiminden bağımsız yeni bir
gerekçedir.
**Etkilediği bölüm:** CLAUDE.md §5, §11 (MVP sırası), implementation-plan
**Kim:** Ekip

---

### 2026-08-25 — Sonuç tablosuna "aynı gün" sağlamlık sütunu eklenecek
**Karar:** RecBole sonuç tablosu, held-out item'ı bir önceki etkileşimden
**gerçekten sonra** olan kullanıcı alt kümesinde hesaplanmış ikinci bir metrik
sütunu taşıyacak.
**Gerekçe:** Ölçüldü (T11). Ardışık etkileşimlerin **%34.8–43.0'ı aynı takvim
gününde**; daha kritiği, kullanıcıların **%24.5–31.5'inde leave-one-out test
item'ı bir önceki etkileşimle aynı gün**. Bu durumda model "sonraki alımı" değil
"aynı review oturumundaki başka bir ürünü" tahmin ediyor — farklı ve daha kolay
bir görev. İnsanlar birikmiş alımlarını tek oturumda yorumluyor; bu veri
kümesinin yapısal bir özelliği. Raporlanmazsa RQ2'nin ölçtüğü şey yanlış anlaşılır.
**Etkilediği bölüm:** `recsys/run_experiment.py` (henüz yazılmadı), sonuç raporlaması
**Kim:** Ekip

---

### 2026-08-25 — `keyword_scan` genişletildi, `analysis/deep_eda.py` eklendi
**Karar:** `keyword_scan` artık dört türetilmiş sinyal daha üretiyor:
`kw_in_title`, `kw_pos_rel` (kanıtın göreli konumu), `kw_recipient`, `kw_occasion`.
Bunların üzerine `analysis/deep_eda.py` modülü eklendi (T5–T14, F9–F16).
**Gerekçe:** Data Research'ün "exploratory analysis" bölümünü betimsel
istatistikten çıkarıp, projenin sonraki aşamalarının dayandığı varsayımları
ölçen bir bölüme dönüştürmek. Üç varsayım bu sayede ölçüldü ve ikisi yanlış çıktı
(yukarıdaki girdiler).
**Gizlilik:** Türetilmiş kolonlar birebir metin taşımıyor; `kw_recipient` bir
ilişki etiketi ("daughter"), kişisel veri değil. `title`/`text` yazımdan önce
düşürülüyor.
**Etkilediği bölüm:** Paket yapısı, CLAUDE.md §2
**Kim:** Ekip

---

### 2026-08-26 — Annotator LLM değişti: Qwen3.5-9B → Qwen3-4B-Instruct-2507
**Karar:** Birincil annotator `Qwen/Qwen3-4B-Instruct-2507`, ikincil `google/gemma-4-E4B`
(yalnızca ~5K uyum alt örneklemi). `quantization: awq` → `none`, `dtype: float16` eklendi.

**Gerekçe — seçim kriteri yanlıştı.** Eski tablo modelleri **VRAM**'e göre sıralıyordu
("Qwen3.5-9B ~6-7 GB 4-bit, 8 GB kartta rahat koşar"). Ekibin donanımı ölçüldü:

| | Yerel | Kaggle | Colab |
|---|---|---|---|
| GPU | GTX 1650 Ti 4 GB | 2× T4 16 GB | T4 16 GB |
| Mimari | Turing sm75 | Turing sm75 | Turing sm75 |

Hepsi **pre-Ampere**. Bunun üç sonucu var ve hiçbiri VRAM tablosunda görünmüyor:
bfloat16 compute capability 8.0 istiyor (yok), vLLM `FLASH_ATTN` sm80 istiyor (yok),
ve yeni mimariler Turing backend'lerini düşürüyor. Qwen3.5 ailesi hibrit Gated-DeltaNet
+ vision-language; 9B üyesi fp16'da 20.5 GB, ve Turing'de belgelenmiş workaround
(`--enforce-eager`) 10 token/s altında kalıyor. Yani eski seçim **hiçbir donanımımızda
koşmuyordu** — 4 GB'lık kartta da, T4'te de.

**Yerine seçilen:** dense, text-only, standart GQA — Turing kod yolu olgun olan mimari.
Qwen3-4B-Instruct-2507: 4.0B, 36 katman, GQA 32Q/8KV, 262K context, Apache 2.0,
**non-thinking** (40-60K sınıflandırmada muhakeme token'ı israf etmiyor). fp16 ~8 GB →
T4'e kuantizasyonsuz sığıyor, yani AWQ/Marlin kernel riski de ortadan kalkıyor.

**4B yeterli mi?** Varsayım değil, ölçüm: kanıt gövdenin medyan %1-3'ünde (T5) ve naif
sözcüksel vekil zaten %58.3 precision veriyor (T4). Modelin işi sıfırdan sinyal bulmak
değil, bilinen bir hata desenini düzeltmek. macro-F1 ≥ 0.75 kapısı yetmediği durumu
yakalamak için zaten var; tırmanma yolu Qwen3-8B.

**Ayrıca düzeltildi:** "Gemma 4 12B" diye bir model **yok**. Gemma 4 ailesi E2B / E4B /
26B-MoE / 31B-dense olarak çıktı. Üç teslimde birden bu isimle geçiyordu.
**Etkilediği bölüm:** `configs/base.yaml`, CLAUDE.md §6, PROJECT_SPEC §[B], ROADMAP,
concept-note §4.2, technology-review §4.2/§4.3, implementation-plan §1.3/§1.6
**Kim:** Ekip

---

### 2026-08-26 — GPU kararı: annotation Kaggle'da, geri kalan yerelde
**Karar:** CLAUDE.md §13'teki "GPU: yerel RTX mi Kaggle mı" TBD'si kapandı.
LLM annotation **Kaggle** (2× T4, ~30 sa/hafta, Linux); preprocess, distillation,
RecBole deneyleri **yerel** makinede.
**Gerekçe:** Yerel kart 4 GB — 4B modeli fp16'da tutamaz. Ayrıca vLLM Windows'ta
native desteklenmiyor. Kaggle bir yedek plan değil, **tasarımın parçası**. Buna karşılık
altı pipeline aşamasından yalnızca biri kotaya bağlı: ModernBERT-base 149M parametre,
4 GB'a rahat sığıyor, yani distillation ve tam korpus inference yerelde koşuyor.
Yerelde prompt denemesi için llama.cpp/GGUF (Windows'ta çalışıyor, 4 GB'a sığıyor) —
ama veri setine giren etiketler yalnızca vLLM koşusundan gelir.
**Etkilediği bölüm:** implementation-plan §1.6 (yeni), risk tablosu §4.4
**Kim:** Ekip

---

### 2026-08-26 — Annotation şeması v2: `grandchild` eklendi
**Karar:** `recipient` enum'u yeniden tasarlandı:
`child | grandchild | partner | parent | sibling | extended_family | friend | other | unknown | null`.
`occasion` enum'una `anniversary`, `baby_shower`, `valentines_day`, `mothers_day`,
`fathers_day`, `other` eklendi.

**Gerekçe:** Eski şema veri görülmeden sezgiyle tasarlanmıştı. T6/T6b onun veriyle ilk teması:

- **`grandchild` en kritik eksikti.** Toys_and_Games'te adı geçen alıcıların **%26.2'si
  torun** (grandson %14.7 + granddaughter %11.5). Eski şemada `child`'a düşüyordu — oysa
  torun ayrı hanede yaşar, yani **tanım gereği hediye**, kendi çocuğu ise `household`
  olabilir. Şemanın en zor sınırı (`household` vs `gift_given`) için en bilgilendirici
  ipucu tam da bu ayrımdı ve kayboluyordu. Birincil deney kategorisinin dörtte biri.
- `spouse` → `partner`: Video Games'te alıcıların %5.5'i sevgili, `spouse` kapsamıyordu.
- `extended_family` eklendi: yeğen Toys'ta %12.4.
- `sibling` eklendi: Video Games'te %3.6.
- `colleague` **kaldırıldı** — sözcüksel taramada hiçbir kategoride görünmüyor; `friend`'e katıldı.
- `occasion` için not: metinden yalnızca **%20-29** oranında çıkarılabiliyor, yani `unknown`
  yaygın ve beklenen bir cevap. Prompt'a modeli vesile uydurmaya zorlamama talimatı eklendi.

**Zamanlama:** LLM koşusundan ÖNCE yapıldı. Şema koşu başladıktan sonra değiştirilirse
etiketler karşılaştırılamaz hale gelir.
**Etkilediği bölüm:** `configs/annotation_schema.json`, `prompts/gift_detection_v1.md`,
CLAUDE.md §4. `detection/schema.py` (Pydantic) yazıldığında buna uymalı.
**Kim:** Ekip

---

### 2026-08-26 — `expected_peaks` ölçüme göre düzeltildi
**Karar:** `analysis.expected_peaks: [11, 12, 2, 5]` → `[12, 1]`.
**Gerekçe:** Eski liste Noel/Sevgililer/Anneler Günü varsayımıydı. Ölçüm (T14): tepe dört
kategoride de **Aralık-Ocak**; Şubat ve Mayıs tepesi **yok**. Aralık-Ocak kayması beklenen
davranış — review tarihi satın alma tarihinin gerisinde kalıyor ve bu gecikme concept-note'ta
önceden not edilmişti. V2 doğrulama kapısı buna göre güncellendi (CLAUDE.md §12).
**Etkilediği bölüm:** `configs/base.yaml`, CLAUDE.md §12, concept-note §2.2
**Kim:** Ekip

---

### 2026-08-26 — DeBERTaV3 distillation sağlamlık kontrolü olarak eklendi
**Karar:** `distill.fallback_model: microsoft/deberta-v3-base`. ModernBERT-base birincil
kalıyor.
**Gerekçe:** ModernBERT'in seçim gerekçesi 8192 context'ten verimlilik + 149M ayak izine
kaydırıldı (bkz. 2026-08-25 girdisi). Kalan gerekçe geçerli ama tek taraflı değil: kontrollü
karşılaştırmalar (arXiv 2504.08716) eşit veriyle DeBERTaV3'ün örneklem verimliliğinde ve
nihai kalitede önde olabildiğini, ModernBERT'in avantajının eğitim hızında olduğunu
buluyor. Elimizde yalnızca 40-60K etiket var, yani örneklem verimliliği akademik değil
canlı bir mesele. İkisi de bu veri hacminde dakikalar içinde eğitiliyor — fidelity kapısı
(LLM'e 5 puan yakınlık) kaçırılırsa ucuz bir sigorta.
**Etkilediği bölüm:** `configs/base.yaml`, technology-review §4.4, PROJECT_SPEC, ROADMAP
**Kim:** Ekip

---

### 2026-08-26 — Örneklem tasarımı: iki ayrı çerçeve
**Karar:** `data/sampling.py` iki çerçeve üretir ve `sample_frame` kolonuyla işaretler.

- **`main`** — `month` × `rating` × `text_length_bucket` (180 hücre) üzerinde **orantılı**
  tahsisli katmanlı rastgele örnek. Orantılı olduğu için **kendinden ağırlıklı**:
  yaygınlık düz ortalamayla hesaplanır, ağırlık gerekmez. Kategori başına 10.000
  (`n_annotate: 40000` dörde bölünür).
- **`boost`** — yalnızca `kw_gift_proxy` havuzundan 1.500 ek satır. Distillation eğitim
  setine daha çok pozitif koymak ve hata analizi için. **Yaygınlık hesabına girmez.**
- **`boost_received`** — `kw_gift_received` havuzundan 300 ek satır (2026-08-27
  denetiminde eklendi). Vekilin tanım gereği dışladığı ama LLM'in ayırması gereken en
  zor negatif sınıf. **Yaygınlık hesabına girmez.**

Üç çerçeve **ayrıktır**; bir satır birden fazlasında geçmez ve bu artık tahsis
raporunda *hesaplanarak* yazılıyor, sabit olarak iddia edilmiyor.

**Gerekçe — ölçülen bedel.** Config'teki "ana orana KATILMAZ" notu soyut bir uyarı değil.
Pilot kategoride üç çerçeve havuzlanırsa oran **%2,02 yerine %14,42** görünüyor: **7,1 kat**
şişme. `tests/test_sampling.py::test_pooling_the_frames_inflates_the_rate` bunu sayıyla
kilitliyor, ki `sample_frame` ayrımını kaldırmaya kalkan biri neyi kaybettiğini görsün.

> **Düzeltme (denetim, 2026-08-29).** Bu paragraf önce "%2.13 yerine %15.01" diyordu.
> İki hata vardı: (1) %2,13 *korpusun* vekil oranı, %15,01 ise *örneklemin* havuzlanmış
> oranı — iki farklı popülasyon karşılaştırılıyordu; (2) havuzlanmış oran hiçbir
> kategoride %15,01 çıkmıyor, ölçülen değer %14,42. Doğru karşılaştırma aynı örneklem
> içinde: `main` %2,02 → havuzlanmış %14,42. Örneklemin korpusa yakınsaması ayrı bir
> iddia ve zaten aşağıda duruyor. Şişme dört kategoride: All_Beauty 7,1× · Grocery 7,7× ·
> Video Games 3,6× · Toys 2,0× (Toys'ta düşük çünkü korpus oranı zaten yüksek).

**Doğrulama.** Dört kategoride `main` çerçevesinin vekil oranı, `clean` korpusun gerçek
oranına iki standart hata içinde yakınsıyor: All_Beauty %2.02 vs %2.13 · Toys %11.27 vs
%11.07 · Video Games %4.63 vs %4.40 · Grocery %1.87 vs %1.85.

> Sayılar 2026-08-27 denetiminden sonra yeniden çekilen örnekleme aittir (katman başına
> türetilmiş seed). Önceki çekimde de dördü 2 SE içindeydi; değişim örnekleme
> hatası kadar.

**Kategori başına eşit tahsis** (orantılı değil): RQ1 kategorileri karşılaştırıyor, yani
her kategori eşit kesinlikte tahmin almalı. Havuzlanmış tek bir oran raporlanmadığı için
havuzlama ağırlığı da gerekmiyor.
**Etkilediği bölüm:** `configs/base.yaml` (`sampling.text_length_buckets` eklendi),
CLAUDE.md §2, README
**Kim:** Ekip

---

### 2026-08-26 — Prompt deneme seti doğrulamadan dışlanacak
**Karar:** `data/annotations/human/prompt_trial_200.csv` ile birlikte
`prompt_trial_ids.json` yazılıyor; içinde çekilen 200 satırın `row_id`'leri ve
`exclude_from_validation: true` bayrağı var. Hafta 4'ün 500'lük doğrulama seti bu
satırları **dışlamak zorunda**.

**Gerekçe:** Prompt v2 bu 200 satır okunarak yazılacak. Aynı satırlarla detektörü
doğrulamak, prompt'u kendi test setine fit etmek olur ve raporlanan F1'i anlamsız kılar.
Aynı hatadan bir kez daha kaçınıyoruz: T4 denetiminde `stocking stuffer` desen kusuru
bulunduğunda da desen bilerek düzeltilmemişti (doğrulama etiketleri görüldükten sonra
ayarlama yapmak precision'ı yapay olarak iyileştirirdi).

**Deneme seti bir tahmin örneği DEĞİLDİR:** zor vakaları kasıtlı fazla temsil ediyor
(%50 vekil-işaretli, %26 spekülatif, %24 işaretsiz), çünkü prompt'u kıranlar onlar.
Buradan yaygınlık okunmaz.
**Etkilediği bölüm:** Hafta 4 doğrulama akışı, `analysis/validation.py` (yazılmadı)
**Kim:** Ekip

---

### 2026-08-26 — Elle etiketleme CSV değil xlsx üzerinden yapılır
**Karar:** `data/labelsheet.py` etiketleme CSV'sini xlsx'e çevirir (`--export`), doldurulmuş
sayfayı doğrulayıp UTF-8 CSV'ye geri yazar (`--ingest`). Etiketleyen kişi xlsx'i doldurur,
CSV'ye elle dokunmaz. Kurallar `docs/ETIKETLEME_REHBERI.md`.

**Gerekçe — iki sessiz bozulma.** Türkçe Windows yerel ayarında Excel'in liste ayracı `;`
olduğu için virgüllü CSV tek kolona düşer; ayrıca Excel BOM'suz UTF-8'i tanımayıp cp1254 ile
geri yazar ve review metnindeki karakterler bozulur. İkisi de ancak 2-3 saatlik etiketleme
bittikten sonra fark edilir. xlsx'te ayraç ve kodlama diye bir kavram yok.

**Önyargı kontrolü:** `trial_stratum`, `sample_frame` ve `kw_gift_proxy` kolonları sayfaya
**yazılmaz**. Etiketleyen, sözcüksel vekilin kararını görmeden etiketler; görseydi etiketler
vekilin hatalarını tekrarlar ve Hafta 4'teki insan-vekil karşılaştırması kendi kendini
doğrulayan bir ölçüme dönerdi. Kolonlar `--ingest` sırasında `trial_id` üzerinden geri eklenir.
`tests/test_labelsheet.py::test_export_hides_the_bias_columns` bunu kilitliyor.

**`--ingest` doğrulaması:** satır silinmiş/eklenmiş mi (`trial_id` kümesi), satır kopyalanmış
mı, sözlük dışı etiket var mı. Üçü de hata verir; yarım doldurulmuş dosya hata değil, ilerleme
raporu üretir. Rapor ayrıca insan etiketini `kw_gift_proxy` ile karşılaştırır - prompt v2'nin
asıl girdisi bu tablo.

**Hafta 4'te aynı modül kullanılacak:** 500 öğe, 3 annotator, aynı önyargı kontrolü.
**Etkilediği bölüm:** `requirements.txt` (xlsxwriter, openpyxl), CLAUDE.md §2,
`docs/ETIKETLEME_REHBERI.md` (yeni)
**Kim:** Ekip

---

### 2026-08-26 — Deneme setinin kaynağı: yazar destekli ön geçiş
**Karar:** `prompt_trial_200_labeled.csv` teslimlerde **"yazar destekli ön geçiş"**
olarak anılacak, "insan doğrulaması" olarak değil. Hafta 1'deki T4 keyword precision
örneklemiyle aynı konvansiyon (`data-research.md` §4.9).

**Gerekçe.** 200 satırın etiketleri elle verildi ama `notes` alanı bir dil modeli
yardımıyla dolduruldu: 130 notun 116'sı tek tip resmi bir kalıpta ("The reviewer…"),
21'inde yapıştırma izi (kıvrık tırnak), hiçbirinde Türkçe karakter yok, ve rehberin
birebir istediği `KENDI_COCUGU` / `EMIN_DEGIL` işaretleri hiç geçmiyor.

**Etiketlerin kalitesi bu kararın gerekçesi değil — kalite iyi.** Otomatik denetimde
200 satırda yalnızca 2-3 tartışmalı etiket bulundu, torun kuralı (`grandchild` →
`gift_given`) sıfır hatayla uygulanmış, ve §3.1/§3.7 çelişkisi rehberde yazandan daha
doğru çözülmüş. Sorun doğrulukta değil **bağımsızlıkta**.

**Bu sette etkisi sınırlı, Hafta 4'te ölümcül olurdu.** Deneme setinin işi prompt
yazmak; etiketler doğru olduğu sürece prompt doğru yöne ayarlanır. Hafta 4'ün 500'ü
ise detektörün F1'ini ölçüyor — referans da bir dil modelinden gelirse ölçülen şey iki
modelin birbirine benzerliği olur, doğruluk değil, ve rapordaki "insan doğrulaması"
ifadesi yanlış olur.

**Sayısını bilemediğimiz kayıp:** insan geçişinin asıl değeri LLM'in *kendi* kör
noktalarını yakalamaktır. Model kendi kör noktasını işaretlemez.

**Sonuç:** Hafta 4 doğrulaması model yardımı olmadan yapılacak; bu kural
`docs/GENEL_BAKIS.md` §7'de bozulmaz kurallar listesine eklendi.
**Etkilediği bölüm:** `docs/GENEL_BAKIS.md`, Hafta 4 doğrulama akışı
**Kim:** Ekip

---

### 2026-08-26 — Prompt v2: deneme geçişinden çıkan üç kural
**Karar:** `prompts/gift_detection_v2.md` açıldı, `configs/base.yaml`'daki
`detection.prompt_path` oraya çevrildi. v1 **silinmedi** — hangi annotation hangi
prompt'la üretildiği izlenebilir kalmalı.

**Ölçüm.** Sözcüksel vekilin 200 satırdaki başarısı: kesinlik **0,610**, duyarlılık
**0,762**, F1 **0,678** (39 yanlış pozitif, 19 kaçırma). İki hata kümesi incelendi.

**Üç ekleme:**
1. **Gerçekleşmiş / önerilmiş hediye üçlü ayrımı.** v1 spekülatif ifadeye tek cevap
   veriyordu (`self`). Deneme geçişi iki ayrı durum olduğunu gösterdi: kullanım kanıtı
   varsa `self`, hiçbir kanıt yoksa `unclear`. v1'in kuralı ikincisini `self` sayıp
   yaygınlığı `self` yönünde şişiriyordu.
2. **Alıcının tepkisi tek başına kanıttır.** Kaçırılan 19 satırın 6'sında "hediye"
   sözcüğü hiç geçmiyor; tek kanıt hane dışı birinin tepkisi. Dil modelinin sözcük
   listesinden üstün olması gereken yer burası.
3. **Satın alma fiili gerekmez.** "Torunumuza harika bir hediye oldu" biçimindeki bir
   cümlede `bought/purchased` yok ama gerçekleşmiş bir hediye. Kaçırmaların en büyük
   tek kalıbı buydu.

**Few-shot örnekleri uydurma.** Gerçek review metni depoya girmiyor (CLAUDE.md §8.1);
örnekler ölçülen kalıplara göre yazıldı, kopyalanmadı.
**Etkilediği bölüm:** `configs/base.yaml`, `tests/test_prompting.py` (+2 test),
`docs/ETIKETLEME_REHBERI.md` §3.1
**Kim:** Ekip

---

### 2026-08-26 — İnsan doğrulaması 3 annotator ile yapılacak
**Karar:** Hafta 4'ün 500 öğelik doğrulama seti **3 kişi** tarafından bağımsız
etiketlenecek; uyum **Fleiss' κ** ile raporlanacak, eşik κ ≥ 0.60.

**Gerekçe:** Bu, spec'in başından beri varsaydığı tasarım (`PROJECT_SPEC.md` §6[C],
`concept-note` başarı kriterleri). Belirsiz olan tek şey kaç kişinin gerçekten
etiketleyeceğiydi; karara bağlandı, yedek senaryolara (2 kişi → Cohen's κ, tek kişi →
intra-annotator agreement) gerek kalmadı.

**Sonucu:** Hafta 4 örneklemi 500'de kalıyor (tek kişilik senaryoda 250'ye inecekti),
κ bir kapı olarak korunuyor, ve `household` sınıfının C1'de silinip silinmeyeceği
kararı κ sonucuna bağlı olmaya devam ediyor (CLAUDE.md §13).

**Şart:** üçü de `docs/ETIKETLEME_REHBERI.md`'yi okumuş olmalı ve **model yardımı
almadan** etiketlemeli — aksi halde ölçülen κ, insanlar arası gerçek belirsizliği
değil aynı modelin kendisiyle tutarlılığını gösterir.
**Etkilediği bölüm:** `docs/ETIKETLEME_REHBERI.md` başlığı, `docs/DECISIONS.md`
etiketleme sayfası kaydı
**Kim:** Ekip

---

### 2026-08-27 — Denetim: sekiz kusur bulundu ve düzeltildi
Hafta 2 sonunda kod ve kararlar baştan gözden geçirildi. 16 sayısal iddianın hepsi
üretilmiş çıktılara karşı doğrulandı (korpus boyutları, vekil oranları, yansızlık,
havuzlama şişmesi, %26.2 torun payı). Sekiz kusur bulundu — sonuncusu ilk yedinin
düzeltmesi doğrulanırken ortaya çıktı:

**1 — Gizlilik: 9 commit edilmiş dosyada mutlak yol.** `preprocess.py` ve `download.py`
JSON'lara `str(path)` yazıyordu; içinde işletim sistemi kullanıcı adı ve tam dizin ağacı
vardı. Aynı hata `precision_check.py`'de daha önce düzeltilmişti ama yardımcı fonksiyon
o modülün içinde özel kalmıştı, diğerleri kullanamamıştı. `utils/io.relative_to_repo`
ortak hale getirildi; `tests/test_preprocess.py::test_funnel_json_carries_no_absolute_path`
regresyonu kilitliyor. Mevcut 9 dosya düzeltildi ve düzeltmenin kodun ürettiğiyle
**birebir aynı** olduğu Video_Games yeniden koşularak doğrulandı.

**2 — Deneme/doğrulama katmanları bayrak uzayını kapsamıyordu.** `_trial_strata` üç havuz
tanımlıyordu ve `kw_gift_received` satırları hiçbirine düşmüyordu: 200 satırlık deneme
geçişinde "hediye ALMIŞ" vakasından **sıfır** örnek vardı. Oysa *"receiving a gift is not
giving one"* prompt'un üç kritik ayrımından biri — yani hiç sınanmadan doğrulanmış
sayılacaktı. Korpus genelinde 48.525 review bu sınıfta. Katman sayısı dörde çıkarıldı
(`proxy .35 / speculative .25 / received .15 / unflagged .25`) ve örnekleme
`boost_received` çerçevesiyle 300 satır/kategori takviye ediyor.
`test_trial_strata_partition_the_frame` ayrıklık **ve** tüketicilik şartını kilitliyor.

**3 — Tüm katmanlarda aynı seed.** `pool.sample(take, seed=seed)` her katmanda aynı
seed'i kullanıyordu; aynı boyutlu iki havuz **birebir aynı konumları** seçiyordu
(ölçüldü: iki farklı katman da `[275, 607, 687, 702, 851]`). Nokta tahmini yansız
kalıyordu — ama katmanlar arası bağımsızlık yoktu ve `experiment.bootstrap_iters`
bağımsız çekim varsayıyor. Pratik etki ölçüldü ve küçüktü (yıl dağılımında toplam
varyasyon mesafesi 0.013–0.015). `_stratum_seed()` eklendi: `crc32` tabanlı, süreçler
arası kararlı. `hash()` kullanılamaz — PYTHONHASHSEED ile değişir ve `seed: 42` ile
yeniden üretilebilirlik iddiasını yalanlar.

**4 — Hiçbir test `force=True` kullanmıyordu.** Üç test "idempotent" iddia ediyordu ama
üçü de `should_skip` yoluna girip dosyayı kendisiyle karşılaştırıyordu. CLAUDE.md §7'nin
iddiası iki parçalı ve asıl önemli yarısı — yeniden hesaplama aynı sonucu veriyor mu —
hiç sınanmıyordu. İki determinizm testi eklendi; yeniden hesaplama deterministik çıktı.

**5 — `eda_tables.json` bayattı.** T4 precision kaydını olduğu gibi gömüyor ve düzeltme
öncesi kopyayı taşıyordu. EDA yeniden koşuldu.

**6 — İki ölü config anahtarı.** `sampling.strata` ve `preprocess.dedup` config'te
parametre gibi duruyordu ama hiç okunmuyordu; değiştirmek çıktıyı değiştirmezdi
(CLAUDE.md §8.11 ihlali). `preprocess.dedup` artık okunuyor; `sampling.strata` için
`_check_strata()` eklendi — kod ile ayrışırsa gürültülü hata verir.

**7 — `frames_disjoint: True` doğrulanmadan yazılıyordu.** Tahsis raporuna hesaplanmamış
bir iddia yazmak, kod değiştiğinde sessizce yalan söyleyen bir alan bırakır. Artık
`out["row_id"].n_unique() == out.height` ile hesaplanıyor.

**8 — Dışlama anahtarı belirsizdi.** Doğrulama sırasında ortaya çıktı: `row_id` her
kategoride 0'dan başlıyor, yani kategoriler arasında **çakışıyor** (ölçüldü: dört
kategorinin örnekleri arasında 78 ortak değer). `prompt_trial_ids.json` düz bir `row_ids`
listesi tutuyordu ve Hafta 4'ün doğrulama seti tam olarak o listeyi kullanacaktı — başka
kategorilerde masum satırları da dışlayarak. Kayıt `excluded: {kategori: [row_id...]}`
biçimine çevrildi; anahtar artık `(category, row_id)` çifti.
`test_exclusion_key_is_scoped_by_category` düz listenin geri gelmesini engelliyor.
Mevcut 200 satırlık kayıt etiketli CSV'den yeniden kuruldu — satırlar değişmedi.

**Örneklem yeniden çekildi.** Kategori başına 11.800 (10.000 main + 1.500 boost + 300
received), toplam **47.200**. Yansızlık dört kategoride de korunuyor (hepsi 2 SE içinde).
Hafta 3 henüz başlamadığı için maliyet sıfıra yakındı — 46.000 satır Kaggle'da
etiketlendikten sonra aynı düzeltme o kotayı çöpe atardı.

**Deneme setinin 200 satırı yeniden çekilmedi.** Görevi tamamlandı (prompt v2 yazıldı) ve
yeniden etiketlemek 2-3 saat insan emeği demek. Yeni örneklemin alt kümesi değil artık
(200'den yalnızca 5'i içinde), ama dışlama listesi `row_id` üzerinden çalıştığı için
Hafta 4'te işlevini görmeye devam ediyor. **Açık kalan:** `received` kuralı hâlâ hiçbir
insan etiketiyle sınanmadı.
**Etkilediği bölüm:** `data/sampling.py`, `data/preprocess.py`, `data/download.py`,
`utils/io.py`, `configs/base.yaml`, testler (117 → 127), `reports/results/` (9 dosya),
`data/annotations/human/prompt_trial_ids.json`
**Kim:** Ekip

---

### 2026-08-27 — Şema v3: `received` beşinci sınıf, `recipient`'ta `null` kaldırıldı

Şema, LLM koşusundan **önce** eleştirel okundu. Bir kusur 40.000 satır etiketlendikten
sonra geri alınamaz: yeniden annotation Kaggle kotasını ikinci kez harcamak demek.

**1 — `received` beşinci sınıf oldu.** v2 "hediye alan"ı `self`'e katlıyordu. Ama
`household` sınıfının var olma gerekçesi *"hediye değil ama alıcının kendi tercihi de
değil"* — hediye **alan** da tam olarak bu durumda; aynı mantık iki vakaya farklı
uygulanıyordu. Asıl sorun bilgi kaybı: bir kez `self` yazıldıktan sonra geri gelmez ve
C1/C2/C3 sonradan karar veremez. Annotation anında bilgi yok etmiyoruz.

Yan etki yok: mevcut 200 etiketli satırda `kw_gift_received` sıfır, yani yeniden eşleme
kaybı yok. (Vekilin duyarlılığı mükemmel olmadığı için birkaç satır kaçmış olabilir;
ölçülemez ve etkisi ihmal edilebilir.)

**2 — `recipient`'tan `null` kaldırıldı.** Enum'da hem `null` hem `"unknown"` vardı ve
hangisinin ne zaman kullanılacağı hiçbir yerde yazmıyordu; 40.000 satırda model ikisi
arasında rastgele gidip gelir ve analiz karışırdı. Artık tüm alanlar string — guided
decoding de sadeleşti. `received` kaydında `recipient` **vereni** gösterir.

**3 — `occasion`'da `none` / `unknown` kuralı prompt'a yazıldı.** Ayrım anlamlıydı
(`none` = hediye değil, vesile kavramı geçersiz; `unknown` = hediye ama vesile yazmıyor)
ama yalnızca örneklerden çıkarılabiliyordu.

**4 — `confidence`'ın kullanım amacı yazıldı.** Alan duruyordu ama ne işe yarayacağı
hiçbir dokümanda yoktu. Amaç: distillation eğitim setini filtrelemek ve hata analizini
önceliklendirmek. LLM öz-beyanı ve kalibrasyonu zayıf olduğu için **kapı değildir** —
bu da yazıldı.

**Prompt v3 açıldı**, v2 silinmedi. v2'nin üç kuralı aynen taşındı.

**Ayrıca düzeltildi:** `PROJECT_SPEC.md` ve `ROADMAP.md` hâlâ **v1** şemasını gösteriyordu
(`spouse`, `colleague`) — 26 Ağustos'taki v2 revizyonunda atlanmışlar.
**Etkilediği bölüm:** `configs/annotation_schema.json`, `detection/schema.py`,
`data/sampling.py` (LABELS), `prompts/gift_detection_v3.md`, CLAUDE.md §4,
PROJECT_SPEC, ROADMAP, ETIKETLEME_REHBERI §1-3.2
**Kim:** Ekip

---

### 2026-08-27 — Ürün metadata'sı indiriliyor ve prompt'a giriyor

**Bulgu.** `configs/base.yaml` başından beri `meta_prefix: raw_meta_` ve
`join_key: parent_asin` tanımlıyordu ama `download.py`'de metadata diye bir şey yoktu.
Yani config bir yeteneği ilan ediyordu, kod onu hiç uygulamıyordu — denetimde bulunan
"ölü config anahtarı" sınıfının bir örneği daha.

**Sonucu.** LLM bir review'ı etiketlerken *"she loved it"* cümlesindeki "it"in ne
olduğunu bilmiyordu; yalnızca dataset kategorisini (`Toys_and_Games`) görüyordu.

**Karar.** Dört kategorinin metadata'sı indiriliyor (**4,35 GB**: Toys 2,5 · Grocery 1,3
· Video Games 0,4 · All_Beauty 0,2). `data/metadata.py` jsonl'i parquet'e çeviriyor ve
**yalnızca ihtiyaç duyulan alanları** okuyor — `description`, `features`, `images`,
`videos`, `details` hiç ayrıştırılmıyor. Tutulanlar: `parent_asin`, `title`,
`main_category`, `store`, `price`, `average_rating`, `rating_number`, `categories`.

Örneklem `product_title` ve `product_category` olarak join'liyor (`main_category` →
`product_category`: dataset kategorisiyle ve review başlığıyla karışmasın). Prompt v3'ün
USER bloğuna `Product: {product_title}` satırı girdi.

**Ürün adı bağlamdır, kanıt değildir.** Prompt bunu açıkça yasaklıyor: oyuncak olduğu
için `gift_given` demek en bariz yeni hata yolu. `evidence_span` yalnızca review
metninden alınabilir. `test_prompt_forbids_inferring_the_label_from_the_product_type`
bu yasağı kilitliyor.

**Kapsam ölçülüyor, varsayılmıyor.** Eşleşmeyen `parent_asin` ve metadata'da boş gelen
başlıklar birlikte sayılıp tahsis raporuna `n_missing_product_title` olarak yazılıyor.
Pilot kategoride: 11.800 satırda 2 boş başlık, 0 eşleşmeme.

**Kapsam dışı:** RecBole `.item` dosyaları bu blokta yazılmadı. Metadata Hafta 6'da item
feature olarak da kullanılabilir ama SASRec/BPR ID tabanlı çalıştığı için MVP
gerektirmiyor.
**Etkilediği bölüm:** `data/download.py` (`--meta`), `data/metadata.py` (yeni),
`data/sampling.py`, `detection/prompting.py` (`REQUIRED_FIELDS`), `configs/base.yaml`,
CLAUDE.md §2-3
**Kim:** Ekip

---

### 2026-08-27 — Test fixture'ları gitignore'a takılıyordu

**Bulgu.** `tests/fixtures/` altında **hiçbir dosya takip edilmiyordu.** `.gitignore`'daki
`*.jsonl` kuralı — 15 GB'lık ham veriyi dışarıda tutmak için yazılmış — sentetik test
fixture'larını da yutuyordu. Yani temiz bir clone'da `mini_reviews.jsonl` yok ve **136
testin tamamı kırılıyordu.**

Metadata fixture'ı (`mini_meta.jsonl`) eklenirken `git status`'ta görünmemesi üzerine
fark edildi.

**Neden önemli:** CLAUDE.md §10 testlerin sentetik fixture ile koşmasını şart koşuyor ve
projenin teslim vaadi "yeniden üretilebilir kod". Fixture repoda yoksa ikisi de geçersiz.

**Düzeltme:** `!tests/fixtures/` + `!tests/fixtures/**` istisnası. Gerçek veriyi dışarıda
tutan kural aynen duruyor.
**Etkilediği bölüm:** `.gitignore`, `tests/fixtures/mini_reviews.jsonl`,
`tests/fixtures/mini_meta.jsonl`
**Kim:** Ekip

---

### 2026-08-27 — Hafta 3 öncesi: prefix caching, inference kapsamı, açık kararlar

Kararlar baştan sorgulandı. Çoğu sağlam çıktı; üç şey eklendi, bir de **kendi hatam
düzeltildi**.

**Düzeltme — `kcore` kayması zaten ölçülmüştü.** Denetim sırasında "k-core korpusunda
hediye oranı farklı ve bu ölçülmemiş" diye bir bulgu bildirdim. **Yanlıştı.** T8
(`table_corpus_comparison`) bunu ölçüyor, `data-research` §4.11 tam bir bölüm ayırıyor
("deney null sonuca doğru yanlı olacak"), ve 2026-08-25 tarihli "RQ2'nin birincil
kategorisi Toys" kararının gerekçesi doğrudan bu. Mevcut işi yeniden keşfetmişim.
Yapılan tek gerçek ekleme: `GENEL_BAKIS`'a bir uyarı kutusu — o doküman "buradan
başlayın" girişi ve `clean` oranlarını gösterip `kcore` farkından hiç söz etmiyordu.

**1 — Prefix caching zorunlu hale getirildi.** Ölçüm: prompt v3'ün SYSTEM bloğu
**~2.450 token**, review medyanı **~30 token**. Her satırın prefill'inin **%99'u aynı**.
`enable_prefix_caching` kapalıysa 47.200 satırda ~**116M gereksiz prefill token**
üretilir. vLLM ayarları hiçbir dokümanda geçmiyordu; CLAUDE.md §6'ya yazıldı
(`enable_prefix_caching=True`, `max_model_len=4096`).

> **Config'e anahtar EKLENMEDİ, bilinçli olarak.** `detection/llm_annotate.py` henüz
> yazılmadı; anahtarı şimdi eklemek onu okuyansız bırakırdı — denetimin 6. bulgusu tam
> olarak buydu (`sampling.strata`, `preprocess.dedup`, ve haftalarca ölü duran
> `meta_prefix`). Anahtarlar onları okuyan kodla birlikte gelecek.

**2 — Inference kapsamı: önce `kcore`, sonra `clean`.** Dokümanlar her yerde "tam korpus
inference" diyordu ama sıra hiç kararlaştırılmamıştı. RQ2–RQ4 yalnızca k-core'a etiket
istiyor: **4,97M satır** (~3–5 saat), 27M değil (~15–25 saat). RQ1 zaten `main`
çerçevesinden güven aralığıyla cevaplanabiliyor; `clean` koşusu betimsel eğrileri
keskinleştiriyor ama deneyi bloklamamalı.

**3 — İki açık karar kaydedildi** (CLAUDE.md §13):
- `received` sınıfı C1'de silinecek mi? Şema v3 beşinci sınıfı ekledi, C1'in tanımı hâlâ
  yalnızca `gift_given` diyor. `household` ile aynı soru, aynı gerekçe.
- κ eşiği 5 sınıfta hâlâ 0.60 mı? Fleiss' κ sınıf sayısı arttıkça düşer. Hafta 4
  **öncesinde** karara bağlanmalı — sonuç görüldükten sonra eşik düşürmek olmaz.

**4 — Hafta 5 için gereklilik: damıtmanın işe yaradığı ölçülecek.** ModernBERT'in gerekli
olduğu şu an bir varsayım. Aynı 47.200 etiketle TF-IDF + lojistik regresyon eğitilip aynı
doğrulama setinde karşılaştırılacak. Fark küçükse rapora girer ve CPU'da koşan bir
yedeğimiz olur; büyükse damıtma gerekçesi sayıyla desteklenir. Şu an bu yalnızca bir
gereklilik notu — kod Hafta 5'te.
**Etkilediği bölüm:** `CLAUDE.md` §6 ve §13, `docs/GENEL_BAKIS.md` §5
**Kim:** Ekip

---

### 2026-08-28 — Hafta 3: pilot kategori Toys, Kapı 1 eşikleri koşudan önce sabitlendi

**Karar 1 — Pilot annotation `Toys_and_Games`'te koşuyor, `All_Beauty`'de değil.**
Roadmap §10 "pilot kategori" diyor ve `pilot` rolü `All_Beauty`. Bu seçim, inference'ın
pahalı olduğu varsayımından geliyordu; prefix caching ölçüldükten sonra üç kategorinin
maliyeti aynı (dosya başına 11.800 satır). Toys'a geçmenin üç gerekçesi var:

- Deneyin gücü yalnızca orada: %11 yaygınlık, 2,16M k-core etkileşimi. All_Beauty'nin
  k-core'u **boş**, yani C0–C4'e hiç girmiyor (2026-08-25 kararı).
- ~1.300 `gift_given` satırı → aylık eğri ve sınıf bazlı hata analizi için yeterli
  kütle. All_Beauty %2,13 ile ~250 satır bırakırdı, ayda ~20.
- Kapı 1'in koruma amacı: All_Beauty'de çalışıp Toys'ta patlayan bir detektörü
  All_Beauty kapısı yakalayamaz.

`All_Beauty` **pipeline pilotu olarak kalıyor** (CLAUDE.md §14) — kuru koşu ve yol
doğrulaması orada yapılabilir.

**Karar 2 — Kapı 1'in dört ölçütü `configs/base.yaml` → `gate1:` altında, sonuç
görülmeden yazıldı.** Roadmap "Aralık–Ocak tepesi görünüyor mu?" diyordu; sayısal eşik
yoktu ve "tepe var gibi" diyerek bozuk bir detektörle devam etme riski açıktı. Her ölçüt
farklı bir arıza tipini yakalıyor:

| # | Ölçüt | Eşik | Yakaladığı arıza |
|---|---|---|---|
| 1 | (Ara+Oca) ÷ (Haz–Eyl) oranı, bootstrap %95 GA 1,0'ı dışlıyor | ≥ 1,25 | **Sinyalsizlik** — etiket gerektirmeyen dış geçerlilik testi (V2) |
| 2 | Vekilin işaretlemediği satırlarda `gift_given` oranı | ≥ %1 | **Anahtar kelime taklidi** — LLM bedava regex'in işini pahalıya tekrar ediyor |
| 3 | parse hatası / `evidence_span` düşürme | < %1 / < %10 | **Şema çöküşü** |
| 4 | 200 satırlık deneme setiyle uyum | ≥ %70 | **Aşırı tetikleme** |

Eşik 1 için gerekçe: sözcüksel vekil dört kategoride de 1,34–1,90× veriyor (T14). LLM en
az bu kadarını görmeli, ama eşik alt sınırın (Toys 1,34) biraz altında bilerek — LLM
vekilin kaçırdığı mevsimsel-olmayan vakaları da yakalarsa oran doğal olarak seyrelir.

**4. ölçüt bir doğrulama DEĞİL.** Prompt v2/v3 tam o 200 satır okunarak yazıldı
(2026-08-26 kararı), dolayısıyla çıkan sayı F1 olarak raporlanamaz. Yalnızca "model her
şeye hediye mi diyor" sorusunu yanıtlar. Deneme koşusu yoksa ölçüt **atlanır** ve karar
`PASS` değil `INCOMPLETE` olur — atlanan ölçüt sessizce geçmiş sayılmaz.

**Sonucu gördükten sonra eşik gevşetmek yasak.** κ eşiğinde reddettiğimiz şeyin aynısı.
Gevşetme gerekiyorsa buraya tarih + gerekçe yazılır.

**Karar 3 — 200 satırlık deneme seti şema v3'te geçerli, yeniden etiketlenmiyor.**
Ölçüldü (2026-08-28): `kw_gift_received` taşıyan satır **0/200**; dar alıcı-tarafı
regex'i **0 eşleşme**; geniş regex 6 satır buldu, altısı da doğru şekilde `gift_given`.
`self` etiketli 50 satırın tamamı birinci ağızdan satın alma/kullanım. Yani v2
rehberinin "hediye aldıysa `self` yaz" kuralı bu 200 satırda hiç tetiklenmedi.

Nedeni tesadüf değil: denetimin 2. bulgusu (2026-08-27) deneme setinin katmanlarının
`kw_gift_received`'ı kapsamadığıydı. **Bedeli:** `received` kuralı hâlâ hiçbir insan
etiketine karşı sınanmadı; ilk sınavı Hafta 4'ün 500'lük setinde olacak (katman payı
%15 → ~75 satır, 3 annotator).

**Yan bulgu:** denetim sonrası yeniden çekim, deneme satırlarının 195'ini örneklemden
çıkardı (All_Beauty'de 5 ortak, diğer üçünde 0). `row_id` ham jsonl satır indeksi
olduğu için `prompt_trial_ids.json` hâlâ doğru review'ları gösteriyor ve Hafta 4'ün
dışlama kuralı sağlam. Ama 200 insan etiketi LLM koşusuyla karşılaştırılmak istenirse
deneme CSV'si **ayrıca** koşturulmalı (200 satır, bedava).

**Etkilediği bölüm:** `configs/base.yaml` (`gate1`, `detection`), `analysis/gate1.py`,
`detection/llm_annotate.py`, `CLAUDE.md` §2/§7
**Kim:** Ekip

---

### 2026-08-28 — Kaggle duman testi (200 satır): ne çalıştı, ne bulundu

İlk gerçek LLM koşusu. Kaggle 2× T4, vLLM 0.28, Qwen3-4B-Instruct-2507, prompt v3.
**Boru hattı uçtan uca çalıştı**: iki GPU veri-paralel, 100+100 satır, birleştirme
temiz, parse hatası %0, `evidence_span` düşürme %1 (2/200), beş sınıfın hepsi üretildi.

**Ölçümler (artık tahmin değil):** SYSTEM prompt'u 2.277 token / 9.859 karakter;
üretim 2,8 satır/sn (1,4/GPU); çıktı 133 token/sn; motor kurulumu ~181 sn.
Uzatma: Toys 11.800 satır ≈ 73 dk, dört kategori ≈ 4,9 saat.

> **Çıktı hızı tahmini 5–15× iyimserdi.** `technology-review` 1.000–2.000 token/sn
> diyordu, gerçek 133. Sebep: T4 compute capability 7.5 → FlashAttention yok
> (`TRITON_ATTN` backend'i), KV cache 4,45 GiB → aynı anda yalnızca ~8 istek. Proje
> planı yine de bozulmuyor çünkü 47.200 satır Kaggle kotasının %16'sı.

**İki kod kusuru duman testiyle ortaya çıktı ve düzeltildi:**

1. **Kurulum süresi üretim hızına karışıyordu.** Rapor 0,79 satır/sn diyordu, gerçek
   üretim 2,8 satır/sn idi — 252 sn'nin 181'i motor kurulumuydu. Küçük koşudan tam
   koşuya uzatma 4 kat yanlış çıkardı. Rapor artık `generation_s` / `startup_s` /
   `rows_per_s_generating` ayrımını yapıyor.
2. **`--limit` çıktısı tam koşuyu sessizce atlatıyordu.** 200 satırlık parquet diskte
   kalınca idempotans devreye giriyor ve tam koşu atlanıyordu; backend kontrolü bunu
   yakalamıyor (ikisi de `vllm`). Artık satır sayısı kontrol ediliyor ve rapor
   `is_partial` / `limit` taşıyor.

**İki ölçüm bulgusu — kod kusuru değil, veriye dair:**

3. **`kw_gift_received` deseninin precision'ı düşük.** `boost_received` çerçevesinden
   gelen 8 satırın yalnızca **2'si** gerçekten alıcı tarafı; 6'sı veren tarafı
   ("Got this as a gift **for my little brother**"). Desen `got this ... as a gift`
   kalıbını yakalıyor ama yönü ayırt edemiyor. **Sonucu:** Hafta 4'ün 500'lük setinde
   `received` katmanına ayrılan %15 (~75 satır) gerçekte ~19 alıcı-tarafı satır
   getirecek. Hâlâ sıfırdan iyi ama beklenti buna göre kurulmalı. Desen **bilerek
   düzeltilmedi**: doğrulama etiketleri görüldükten sonra ayar yapmak precision'ı
   yapay olarak iyileştirir (aynı gerekçe: 2026-08-26, `stocking stuffer`).
4. **LLM'de bir yön hatası (1/8).** *"This was a present from my son"* → `gift_given`
   yazdı, `received` olmalıydı. Ayrıca `recipient` alanında en az 2 hata: eş →
   `sibling` (olmalıydı `partner`), erkek kardeş → `child` (olmalıydı `sibling`).
   `recipient` betimsel bir alan (T6), kapı değil — ama Hafta 4 hata analizinde
   bakılacak listeye girdi.

**Beklenmedik ve iyi haber:** vekilin işaretlemediği `main` satırlarında LLM %19,9
`gift_given` buluyor (Kapı 1 eşiği %1). Bulunanların büyük kısmı **torun** deseni —
*"great game for my grandson"*, *"Our grandson spends hours playing with these"* —
içinde hiç hediye kelimesi geçmiyor, yani sözcüksel vekilin yapısal olarak
göremeyeceği vakalar. Şema v2'de `grandchild`'ı ayrı sınıf yapmanın gerekçesi buydu
ve ilk kez veriyle doğrulandı.

`main` çerçevesinde `gift_given` %27,1 (45/166) çıktı; vekil %11,07 diyordu. 200
satırlık bir alt küme, güven aralığı geniş — ama fark tam koşuda doğrulanırsa RQ1'in
cevabı sözcüksel tahminin iki katından fazla demektir.

**Etkilediği bölüm:** `CLAUDE.md` §6, `detection/llm_annotate.py`
**Kim:** Ekip

### 2026-08-28 — Toys tam koşusu: Kapı 1 sonucu ve ölçülen özellikler
**Karar:** Kapı 1 Toys_and_Games'te **INCOMPLETE**. Üç ölçüt geçti, dördüncüsü
(deneme setiyle uyum) henüz koşulmadı. 3/3 görüp "PASS" demek reddedildi.

**Gerekçe:** Kapının dört ölçütü koşudan **önce** sabitlendi. Sonucu gördükten sonra
ölçüt sayısını düşürmek, kapıyı kurmamış olmakla aynı şey. Atlanan ölçüt `passed:
null` + `skipped: true` olarak raporlanıyor ve karar beklemede.

Koşu: 11.800 satır, 2× T4, 57 dakika (kurulum 196 sn ayrı raporlandı), üretim 3,64
satır/sn, 516K çıktı token. Kod sürümü `add95f3` rapora yazıldı.

| Ölçüt | Ölçülen | Eşik | |
|---|---|---|---|
| 1 mevsimsellik | 1,576 · GA [1,434 – 1,730] | ≥ 1,25 | ✅ |
| 2 vekil ötesi | %18,0 | ≥ %1 | ✅ |
| 3 şema sağlığı | parse %0,03 · span %2,77 | < %1 · < %10 | ✅ |
| 4 deneme uyumu | ölçülmedi | ≥ %70 | ⏳ |

**Dördüncü ölçüt neden ayrı bir koşu istiyor:** 200 deneme satırı annotation
örneğinin **içinde değil**. `build_trial` onları 2026-08-26 tarihli örnekten
çekmişti; örnek ertesi gün `boost_received` çerçevesi eklenince yeniden çekildi ve
200 satırın hiçbiri yeni çekilişte kalmadı (ölçüldü: 50 Toys satırının 0'ı). Bu bir
hata değil — 16M satırlık korpustan 11.800 çekilişte 50 satırın beklenen kesişimi
0,04. Çözüm: `sampling --trial-source 200` satırları korpustan geri kurar (`row_id`
eşleşmesi metin birebir karşılaştırılarak doğrulanıyor), `llm_annotate --trial 200`
onları **gerçek koşuyla aynı** prompt/parse yolundan geçirir (`_label_chunk` iki
akışta da ortak; ayrı bir kopya zamanla üretimden sapardı ve bunu fark etmenin yolu
olmazdı).

**Kapatılan açık:** `trial_agreement` backend'i kontrol etmiyordu. Kuru koşu
çıktısıyla denendi ve kapı gerçek görünen bir **FAIL** üretti. Taklit etiketler
rastgele olduğu için ölçüt her iki yöne de kayabilirdi. `load_joined` ile aynı
koruma eklendi.

---

**Ölçülen üç özellik — kod kusuru değil, veriye ve modele dair:**

1. **Sözcüksel vekil sanılandan çok daha zayıf.** LLM'i referans alırsak `main`
   çerçevesinde vekilin **recall'ı %31,3**, **precision'ı %64,5**. Yani hediyelerin
   üçte ikisini kaçırıyor ve işaretlediklerinin üçte biri hediye değil. Hediye oranı:
   vekil %11,07 → LLM **%23,25** (2,1 kat). 200 satırlık duman testindeki %27,1
   tahmini tam koşuda %23,25'e oturdu.

   Kaçırılanların kaynağı ölçüldü: `gift_given` satırlarının **%43,6'sı torun**
   (`grandchild`), ve bu satırlarda çoğunlukla hiç hediye kelimesi geçmiyor
   ("great game for my grandson"). Şema v2'de `grandchild`'ı ayrı sınıf yapma kararı
   ilk kez veriyle doğrulandı. **RQ1 için sonuç:** kontaminasyon oranı sözcüksel
   tahminin iki katından fazla; RQ2'nin doz ekseni de bu sayılarla çizilmeli.

2. **`confidence` alanı bağımsız bilgi taşımıyor.** `low` ile `unclear` **birebir
   örtüşüyor** (870/870, her iki yönde %100); `medium` yalnızca 551 satırda. Prompt
   confidence için bir *kural* vermiyor, yalnızca örnek veriyor ve model alanı
   etiketin yeniden yazımına indirgemiş. **Sonucu:** DECISIONS 2026-08-27'de Hafta 5
   için not edilen "sınıf dengesi gerekirse `confidence` filtre olarak kullanılabilir"
   planı işe yaramaz — `confidence != low` filtresi `unclear`'ı atmakla aynı şey.
   Prompt **şimdi değiştirilmedi**: tamamlanmış 11.800 satırlık koşuyu ve onunla
   birlikte Kapı 1'i geçersiz kılardı. Hafta 4 kararı.

3. **`kw_gift_received` precision'ı tam koşuda %11.** `boost_received` çerçevesindeki
   300 satırın **%82'si `gift_given`** (veren tarafı), yalnızca %11'i gerçekten
   `received`. Duman testindeki 2/8 bulgusu ölçekte doğrulandı. `main` çerçevesinde
   `received` yaygınlığı **%0,60** (60 satır). **Sonucu:** Hafta 4'ün 500'lük setinde
   `received` katmanına ayrılan %15 (~75 satır) gerçekte ~8 alıcı-tarafı satır
   getirecek — önceki ~19 beklentisinden de düşük. κ eşiği tartışılırken bu sınıfın
   neredeyse boş kalacağı hesaba katılmalı. Desen yine **bilerek düzeltilmedi**
   (aynı gerekçe: 2026-08-26, `stocking stuffer`).

**Bir doğrulama:** `span_downgraded` satırlarının **%100'ü `unclear`** (327/923).
Mekanizma tam tasarlandığı gibi çalışıyor: modelin metinde birebir bulunmayan bir
`evidence_span` uydurduğu satırlar `unclear`'a düşürülüyor. Diğer dört sınıfta
downgrade oranı %0 — yani düşürülen hiçbir satır etiketini korumuyor.

**Etkilediği bölüm:** `analysis/gate1.py`, `detection/llm_annotate.py`,
`data/sampling.py`, `docs/GENEL_BAKIS.md` §6, `CLAUDE.md` §13
**Kim:** Ekip

---

### 2026-08-29 — Kapı 1 kapandı: PASS (4/4). Dördüncü ölçüt bir yanlılık gösterdi
**Karar:** 200 satırlık deneme seti `--trial 200` ile etiketlendi (kod `b9d8664`,
tek süreç, 122 sn üretim). Uyum **%83,5** (167/200), eşik %70 → **geçti**. Kapı 1
kararı **PASS (4/4)**; tam annotation'a geçilebilir.

**Gerekçe:** Dört ölçüt de koşudan önce sabitlenmişti ve dördü de kendi eşiğini
kendi başına geçti. Hiçbir eşik sonuç görüldükten sonra değiştirilmedi.

**Ama sayı bir uyarı taşıyor — geçme kararından ayrı tutulmalı:**

33 uyuşmazlığın **14'ü** (%42) tek bir yönde toplanıyor: insan `household` demiş,
LLM `gift_given`. Sınıf bazında:

| sınıf | insan | LLM | precision | recall |
|---|---:|---:|---:|---:|
| `gift_given` | 80 | 98 | 0,796 | 0,975 |
| `household` | 45 | 29 | **1,000** | **0,644** |
| `self` | 50 | 52 | 0,846 | 0,880 |
| `unclear` | 25 | 21 | 0,762 | 0,640 |

`household` precision'ı 1,000: LLM bu etiketi **yanlış yere koymuyor**, yalnızca
**az koyuyor** ve boşluğu `gift_given` ile dolduruyor. Yön tek taraflı —
`gift_given → household` hatası **sıfır**. Yani model, ev halkı için alınmış
ürünleri sistematik olarak hediye sayıyor.

**Üç sonucu var:**

1. **Tam koşudaki %23,25'lik hediye oranı bir ÜST SINIR sayılmalı.** `gift_given`
   precision'ı 0,796; model hediyeyi fazla çağırıyor. Üstelik bu 200 satır
   prompt'un **yazılırken okunduğu** satırlar (2026-08-26) — sayı iyimser tarafta.
   Görülmemiş veride fazla çağırma daha kötü olabilir, daha iyi değil.
2. **`household`/`gift_given` TBD'si artık veriye dayanıyor.** İki sınıf C1'de
   birleştirilirse uyum %83,5 → **%90,5** olur. Bu, birleştirme için bir gerekçe
   *değil* — ölçüyü iyileştirmek için sınıf birleştirmek, ölçütü sonradan
   gevşetmenin başka bir biçimi. Ama kararın maliyetini gösteriyor: ayrı tutulurlarsa
   `household` sınıfının etiket gürültüsü Hafta 4'ün κ'sına doğrudan yansıyacak.
3. **Ölçüt "duman testi", doğrulama değil.** Prompt'un gördüğü satırlarda %83,5,
   detektörün *çalıştığını* gösterir — *ne kadar iyi* çalıştığını değil. Gerçek
   ölçüm Hafta 4'ün bağımsız 500 satırı.

**Rapora eklenen:** `trial_agreement` artık yalnızca tek bir uyum yüzdesi değil,
`disagreements` (yön yön uyuşmazlık sayıları) ve `per_class` (sınıf bazında
precision/recall) da yazıyor. Tek bir yüzde hatanın yönünü gizliyordu:
`household → gift_given` ile `gift_given → household` aynı uyum sayısını verir
ama bambaşka iki sorundur ve yalnızca ilki C1'in tanımını ilgilendirir.

**Bir yan bulgu:** `received` deneme setinde **hiç üretilmedi** (0/200) — insan
etiketlerinde de 0. Şema v3'ün beşinci sınıfını uyuşmazlık sayma kararı (bkz.
`test_llm_only_received_label_counts_as_disagreement`) bu koşuda bir maliyet
doğurmadı; koruma yine de yerinde kalıyor.

**Etkilediği bölüm:** `analysis/gate1.py`, `docs/GENEL_BAKIS.md` §6, `CLAUDE.md` §13
**Kim:** Ekip

---

### 2026-08-29 — Hafta 4 öncesi üç karar + denetimden çıkan ölçümler

Kapı 1 geçtikten sonra proje baştan gözden geçirildi. Üç açık TBD karara bağlandı ve
hiçbiri sonuç görüldükten sonra ayarlanmadı. Ayrıca denetim sırasında ilk kez ölçülen
üç şey aşağıda kayda geçiyor — o güne kadar hiçbir belgede yoktu.

**1 — `confidence` alanı v3'te bırakılıyor, analizden düşürülüyor.**
Ölçüm: `low` ile `unclear` **birebir örtüşüyor** (870/870, her iki yönde %100).
Alan bağımsız bilgi taşımıyor; model onu etiketin yeniden yazımına indirmiş. Kök
neden: prompt `confidence` için bir *kural* vermiyor, yalnızca *örnek* veriyor.
Şemaya alan eklerken amacını yazmak yetmiyor — prompt'ta bir karar kuralı yoksa alan
boş çıkıyor.

Prompt v4 yazılıp koşu tekrarlanabilirdi (~4 saat Kaggle, Kapı 1 baskıdan sonra
yeniden koşulurdu) ama yapılmıyor. Alanın tek amacı distillation eğitim setini
sınıf dengesi için filtrelemekti; o iş zaten `boost` çerçevesiyle karşılanıyor.
Bedeli olmayan bir kayıp. **Limitasyon olarak raporlanacak**, sessizce
düşürülmeyecek: "öz-beyan güven alanı kalibre çıkmadı" kendi başına raporlanabilir
bir bulgudur.

**2 — Fleiss' κ eşiği beş sınıfta da 0,60'ta kalıyor.**
κ sınıf sayısı arttıkça düşme eğilimindedir. Bunu bilerek eşiği peşinen düşürmek,
kapıyı koşudan önce gevşetmenin başka bir biçimi olurdu. Bunun yerine iki kural
yazıldı — ikisi de sonuç görülmeden:

- **Sınıf bazlı κ ayrıca raporlanır** (bire-karşı-hepsi). Tek bir toplu sayı,
  sınıflar arasında çok farklı uyum varsa yanıltıcıdır.
- **`n < 20` olan sınıf genel κ'ya KATILMAZ**, ayrıca listelenir
  (`validation.min_class_n_for_kappa`). Öngörülen vaka `received`: `main`
  çerçevesinde %0,60, orantılı çekilseydi 500'lük sette **3 satır** düşerdi.
  Üç satırda κ da F1 de anlamsızdır.

**3 — RecBole iskelesi Hafta 6'yı beklemeden şimdi kuruluyor.**
`src/gift_contamination/recsys/` tamamen boştu ve RQ2/RQ3/RQ4 ile M1–M3'ün hepsi
orada yaşıyor. CLAUDE.md §10'un **kritik** dediği iki test (`test_conditions`,
`test_no_leakage`) de yazılmamıştı. Atomic file üretimi ve C0 baseline dil modeli
etiketi **gerektirmiyor** — sözcüksel vekille uçtan uca koşturulabilir. Etiketler
gelince yalnızca sütun değişir.

Asıl gerekçe zamanlama değil, öğrenme sırası: **C0 ile C4 arasında fark çıkmamalı.**
Çıkarsa deney kurulumunda hata var demektir ve bunu Hafta 6'da öğrenmek, Hafta 5'te
öğrenmekten pahalıdır. Duman koşusunun çıktısı `label_source: proxy` taşıyacak ve
modül vekil kaynaklı girdiden raporlanabilir sonuç üretmeyi reddedecek (`gate1`'in
backend guard'ıyla aynı desen).

---

**Denetimde ilk kez ölçülenler.** Üçü de Toys'un `main` çerçevesinden (10.000 satır),
dil modeli etiketleriyle:

**A — Gerçek sınıf dağılımı.** `self` %47,28 · `gift_given` **%23,25** ·
`household` **%20,17** · `unclear` %8,70 · `received` %0,60.

`household` korpusun **beşte biri**. Bu sayı C1'in tanımını doğrudan ilgilendiriyor:
projenin kendi kontaminasyon tanımı *"alan kişi ürünü kendisi için seçmedi"* ve bu
tanıma göre kendi çocuğuna alınan oyuncak da kontaminasyondur. C1 ya **%23,3** ya
**%43,4** etkileşim çıkarıyor — iki kat fark.

**B — İki sınıfın zaman imzası bambaşka.** (Aralık+Ocak) ÷ (Haziran–Eylül):

| sınıf | oran |
|---|---:|
| `gift_given` | **1,58×** |
| `household` | **1,09×** |
| `self` | 0,76× |

`household` mevsimsel **değil**. Bu iki yönlü bir bulgu: bir yandan sınıfın gerçek
olduğunu gösteriyor (uydurma olsaydı, yani yanlış etiketlenmiş hediyelerden ibaret
olsaydı, Aralık tepesi çıkardı); öte yandan **farklı türde** bir kontaminasyon
olduğunu — kategoriye göre yapısal ama zamana göre düz.

**C — Sınırı çizen şey vesile.** `recipient=child` olan 2.499 satırda:
`household`'un **%99'unda** `occasion=none`; `gift_given`'ın %31'i doğum günü,
%27'si Noel. Model keyfi bölmüyor, *vesile var mı* diye bölüyor. Bu, etiketleme
rehberi §3.4'ün ("önce hane sınırını çizin") kurduğu kuralın veriyle karşılığı.

**Sonucu — `household` TBD'sinin bağımlılığı değişti.** Kayıt bu kararı "kappa
sonucuna bağlı" diyordu. Yanlış bağımlılık: κ sınıfın *ayırt edilebilir* olup
olmadığını söyler, *kontaminasyon sayılıp sayılmayacağını* değil. İkincisi kavramsal
bir karar. Karar Hafta 6 öncesine bırakılıyor ama **kod ikisini de koşabilir hale
getirilecek**: C1 (yalnızca `gift_given`) birincil, C1b (`gift_given` + `household`)
sağlamlık kontrolü. Aynı kod yolu, farklı süzgeç. Rapordaki cümle her iki sonuçta da
güçlenir — *"iki tanım altında da aynı yönde"* ya da *"sonuç tanıma duyarlı"*.

**Etkilediği bölüm:** `configs/base.yaml` (`validation:` bölümü), CLAUDE.md §13,
`data/sampling.py`, `data/labelsheet.py`, `analysis/validation.py` (yeni),
`recsys/` (yeni)
**Kim:** Ekip

---

### 2026-08-29 — RecBole AYRI bir ortama kuruldu; C0/C1/C4 duman koşusu geçti

**Karar:** RecBole ve torch ana `.venv`'e **kurulmuyor**; `.venv-recbole` altında
ayrı bir ortamda duruyorlar. Pin'ler `requirements-recbole.txt`'te.

**Gerekçe — ölçülerek bulundu, tahmin değil.** RecBole 1.2.0 numpy 1.x döneminden.
Ana ortam numpy 2.5 / pandas 3.0 üzerinde ve RecBole `compatibility_settings()`
içinde `np.float_` kullanıyor — numpy 2.0'da kaldırılmış, ilk satırda çöküyor.
Ana ortama kurmak zinciri geri çekerdi: `numpy<2` denendiğinde scipy 1.18 kırıldı
(`np.long` yok), o da scikit-learn'ü kırdı. Üçü birlikte çözülmek zorunda
(`numpy<2` + `scipy<1.14` + `scikit-learn<1.6` + `pandas 2.x`) ve bu yığın
polars/statsmodels tarafını riske atardı.

Ayırmanın bedeli yok çünkü **iki taraf zaten dosyayla konuşuyor**: `.inter`
dosyası arayüz. Ana ortam veriyi hazırlar, RecBole ortamı modeli koşar.
Ana ortamın sürümleri kurulum sonrası doğrulandı — değişmedi, 253 test geçiyor.

**Üç uyumsuzluk daha kayda geçiyor** (hepsi RecBole 1.2.0 ↔ yeni sürümler):

1. `recbole.quick_start` koşulsuz `ray` import ediyor. Kullanmıyoruz — alt seviye
   API (`Config` / `create_dataset` / `Trainer`) yeterli ve bölmeyi kontrol etmek
   için zaten gerekli.
2. `general_recommender/__init__` bütün modelleri toplu import ediyor;
   `ldiffrec` `kmeans_pytorch` istiyor. Kuruldu.
3. torch 2.6'da `torch.load` varsayılanı `weights_only=True` oldu; RecBole
   checkpoint'e kendi `Config` nesnesini de gömüyor ve dosya reddediliyor.
   `run_experiment._torch_load_full()` daraltıcı bir context manager ile
   çözüyor. Güvenli olmasının sebebi dosyanın kaynağının **biz** olmamız —
   saniyeler önce kendi `saved/` klasörümüze yazıldı.

**Bölmeyi RecBole yapmıyor, biz veriyoruz.** `benchmark_filename` ile üç ayrı
dosya (train/valid/test) veriliyor. RecBole kendi bölmesini yapsaydı C1'de
satırlar eksildiği için kullanıcının "son alımı" değişir ve C0 ile C1 **farklı
test setleri** üzerinde karşılaştırılırdı. Test dosyasına yalnızca `label == self`
satırları yazılıyor — CLAUDE.md §5 kural 2 böylece **yapısal** olarak uygulanmış
oluyor, bir kontrol koduna bırakılmıyor.

---

**Duman koşusu sonucu — Video_Games, BPR, 5 epoch, tek seed, CPU:**

| koşul | eğitim satırı | Recall@10 | NDCG@10 | C0'a fark |
|---|---:|---:|---:|---:|
| C0 | 273.150 | 0,0321 | 0,0171 | — |
| C1 | 265.976 | 0,0322 | 0,0172 | +%0,3 |
| C4 | 265.976 | 0,0310 | 0,0164 | **−%3,4** |

**Kapı 2'nin provası geçti:** boru hattı uçtan uca çalışıyor ve C0 ile C4 arasında
kurulum hatasına işaret eden bir sıçrama yok. C4 tam olarak C1 kadar (7.174) satır
çıkardı ve **farklı** 5 ürünü eğitimden düşürdü.

**Bu sayılar BULGU DEĞİL ve sonuç tablosuna giremez.** Üç sebep, çıktıda da
`reportable: false` olarak duruyor:

1. Etiketler **sözcüksel vekilden** — dil modelinden değil. Vekilin recall'ı
   %31,3 (2026-08-28), yani C1'in çıkardığı satırların çoğu gerçek hediye değil.
2. **5 epoch, tek seed, güven aralığı yok.** Recall@10 = 0,032 üzerinde %3'lük
   bir fark seed gürültüsünün içinde kalır.
3. Doğrulanmamış bir detektörün etiketleriyle koşuldu; Hafta 4 daha bitmedi.

Gerçek koşu Hafta 6-7'de: LLM etiketleri, tam epoch, 3-5 seed, bootstrap GA.

**Bir yan bulgu — üç bağımsız çapraz doğrulama.** `atomic.py` Video_Games için
değerlendirilebilir kullanıcı oranını **%97,2** hesapladı; `self` kuralı test
setinden **%2,77** düşürdü. İkisi de Hafta 1'de bambaşka bir kod yolundan üretilen
T7 tablosundaki sayılarla birebir aynı. Bölme ve uygunluk kuralı iki yerde aynı
şekilde uygulanıyor.

**Etkilediği bölüm:** `requirements-recbole.txt` (yeni), `.gitignore`,
`recsys/run_experiment.py` (yeni), CLAUDE.md §6
**Kim:** Ekip

---

### 2026-08-29 — Dört kategori tamamlandı: RQ1 ölçüldü, doğrulama seti yeniden çekildi

Kalan üç kategori Kaggle'da koştu (All Beauty, Video Games, Grocery). Dört
kategorinin tamamı artık etiketli: **4 × 11.800 = 47.200 satır**, hepsi
`backend: vllm`, `prompt v3`, `is_partial: false`, boş kolon yok, çerçeve
tahsisi dördünde de aynı (main 10.000 / boost 1.500 / boost_received 300).

**Koşular farklı `code_version`'larda yapıldı — bu denetlendi.** Toys `add95f3`,
All Beauty `0384602`, Video Games ve Grocery `efae0ea`. Aradaki 245 satırlık
`llm_annotate.py` değişikliği **saf refactor**: üretim yolu (`build_prompts` →
`generate` → `parse_one` → tekrar deneme → `_attach_labels`) birebir aynı kodla
`_label_chunk`/`_attach_labels`'a taşındı, üstüne `--trial` girişi eklendi.
`prompting.py` ve prompt dosyası hiç değişmedi. Bağımsız kanıt: dört koşu da
`shared_prefix_chars: 9859` ve `system_prompt_tokens: 2277` raporluyor — prompt
baytı bayta aynı. Dört kategori karşılaştırılabilir.

#### RQ1 — hediye yaygınlığı (`main` çerçevesi, n = 10.000/kategori)

| kategori | rol | dar (C1) % | %95 GA | geniş (C1b) % | vekil % | LLM/vekil |
|---|---|---:|---|---:|---:|---:|
| Toys and Games | high | **23,25** | 22,43–24,09 | 43,42 | 11,07 | 2,10× |
| Video Games | mid | **7,86** | 7,35–8,40 | 14,48 | 4,40 | 1,79× |
| All Beauty | pilot | **4,70** | 4,30–5,13 | 8,00 | 2,13 | 2,21× |
| Grocery and Gourmet Food | low | **4,28** | 3,90–4,69 | 8,52 | 1,85 | 2,32× |

**Üç bulgu, üçü de kararı etkiliyor:**

1. **Rol ataması bir ÖNGÖRÜYDÜ ve tuttu.** `high/mid/low/pilot` etiketleri
   Hafta 1'de sözcüksel vekilden atandı — hiçbir dil modeli veriyi görmeden
   önce. Ölçülen sıra: Toys ≫ Video Games > {All Beauty, Grocery}. Toys ve
   Video Games'in güven aralıkları birbirinden ve alt ikiliden **ayrık**.
   All Beauty ile Grocery'nin aralıkları örtüşüyor — o ikisi **ayrılamaz** ve
   öyle raporlanacak.

2. **Sözcüksel vekil sistematik olarak ~2,1× kaçırıyor** (1,79–2,32 aralığı,
   dört kategoride). Sıra korunuyor ama seviye korunmuyor. Bu, alandaki
   anahtar-kelime temelli çalışmaların hediye oranını yaklaşık **yarı yarıya**
   eksik saydığı anlamına geliyor ve damıtma yığınının gerekçesi tam olarak bu.

3. **Kontaminasyon tanımı sonucu ikiye katlıyor.** `household` eklenince Toys
   %23,25 → %43,42'ye çıkıyor. Karar hâlâ **kavramsal** ve ekipte (CLAUDE.md
   §13). Kod hiçbirini seçmiyor: `analysis/prevalence.py` ikisini de yan yana
   yazıyor, `recsys/conditions.py` ikisini de koşuyor (C1 / C1b).

**Dar tanım bir ÜST SINIR.** Deneme koşusundaki 33 uyuşmazlığın 14'ü tek yönde
(insan `household`, model `gift_given`); ters yön sıfır. Rapor bunu `caveats`
altında taşıyor.

#### Kapı 1 dört kategoride de PASS (4/4)

| kategori | mevsimsellik | GA alt | vekil-ötesi | parse-fail | span-downgrade |
|---|---:|---:|---:|---:|---:|
| Toys and Games | 1,58× | 1,43 | %18,01 | %0,03 | %2,77 |
| Video Games | 2,08× | 1,74 | %4,85 | %0,16 | %3,08 |
| Grocery and Gourmet Food | 1,71× | 1,33 | %3,10 | %0,06 | %2,49 |
| All Beauty | 2,06× | 1,61 | %3,27 | %0,15 | %1,72 |

Eşikler değiştirilmedi (mevsimsellik ≥ 1,25, vekil-ötesi ≥ %1, parse-fail
≤ %1, span-downgrade ≤ %10). Dört kategorinin tamamında Aralık-Ocak tepesi
güven aralığının alt ucuyla birlikte eşiğin üstünde.

**Ters yönlü bir örüntü:** Toys en DÜŞÜK mevsimselliğe (1,58×) ama en YÜKSEK
vekil-ötesi orana (%18) sahip. Tutarlı bir okuma: oyuncakta hediye yıl boyu
(doğum günü) veriliyor, o yüzden Aralık tepesi oransal olarak küçük kalıyor;
ve hediye o kadar yaygın ki büyük kısmı açık anahtar kelime taşımıyor.

#### Doğrulama seti yeniden çekildi — öncekiler tek kategoridendi

Önceki çekim (2026-08-29 01:47) **500 satırın tamamını Toys'tan** almıştı,
çünkü o an yalnızca Toys etiketliydi. RQ1 kategorileri karşılaştırdığı için
tek kategoriden çekilmiş bir doğrulama seti detektörün kategoriler arası
genellenip genellenmediğini ölçemez. Yeniden çekildi:

- **Toys 200 · Video Games 100 · Grocery 100 · All Beauty 100** (tasarım payı)
- Katmanlar tasarıma birebir: `gift_given` 150, `household` 150, `self` 100,
  `unclear` 60, `received` 40
- Deneme setiyle kesişim: **0** (`(category, row_id)` çifti üzerinden)
- Üç sayfa (A/B/C) aynı satır sırasında — κ hizalaması buna bağlı
- Körleme doğrulandı: sayfalarda yalnızca `val_id, category, title, text,
  label, notes`; `purchase_type`, `kw_gift_proxy` ve diğer önyargı kolonları
  **yok**

Üzerine yazmadan önce üç sayfanın da **sıfır** dolu etiket taşıdığı kontrol
edildi — kimsenin emeği kaybolmadı.

#### Yan düzeltme: rapor sırası kararsızdı

Kapı 1 yeniden koşulduğunda Toys raporu değişti — ama **hiçbir sayı
değişmeden**. Sebep: karışıklık matrisi yalnızca `-adet`e göre sıralanıyordu ve
eşit sayılar `group_by`'ın rastgele sırasına kalıyordu. Aynı veriden iki farklı
dosya çıkıyordu; o gürültünün içinde gerçek bir değişikliği görmek imkânsız.
İkincil anahtar (etiket çifti) eklendi — `gate1.py` ve `validation.py`. İki
ardışık koşu artık **birebir aynı** dosyayı üretiyor. Testi yazıldı.

**Etkilediği bölüm:** `analysis/prevalence.py` (yeni), `tests/test_prevalence.py`
(yeni, 14 test), `analysis/gate1.py`, `analysis/validation.py`,
`tests/test_gate1.py`, CLAUDE.md §2/§6/§10
**Kim:** Ekip

---

### 2026-08-29 — Tam denetim: kod, config ve belgeler gerçeğe karşı sınandı

Depo baştan sona okundu; her belgelenmiş komut, her config anahtarı ve her ölçüm
iddiası gerçek koda ve gerçek veriye karşı kontrol edildi. **Bulunan on kusurun
tamamı düzeltildi.** Sağlam çıkanlar da aşağıda — neyin sınandığını bilmek,
neyin düzeldiğini bilmek kadar önemli.

#### Doğrulanan sağlamlık

- **271 test geçiyor**; 28 modülün hepsi temiz import ediliyor.
- **Bit düzeyinde yeniden üretilebilirlik:** `deep_eda` yeniden koşuldu; 11 tablo
  ve 8 figürün **hepsi** bayt bayt aynı çıktı, çalışma ağacı tertemiz kaldı.
- **Hafta 4 zinciri uçtan uca koştu** — 500 gerçek satırda, sentetik etiketle,
  geçici klasörde: `export → doldur → ingest → Fleiss κ → rapor`. Üç kişi 12 saat
  harcamadan önce boru hattının çalıştığı biliniyor.
- Kod hijyeni: çıplak `except` yok, mutable default yok, sessiz `pass` yok,
  `TODO/FIXME/HACK` yok, commit edilmiş sır yok.
- 17 belgelenmiş komuttan 13'ü sorunsuz; kalan 4'ü aşağıda.

#### Düzeltilen kusurlar

**1 · README'deki `mklink` komutu bozuktu — dosyada kontrol karakteri vardı.**
Heredoc içinde yazılırken `\r` ve `\a` kaçışları yorumlanmış: `data\raw` →
`data<CR>aw`, `D:\amazon` → `D:<BEL>mazon`. Komut kopyalanamaz hâldeydi ve dosya
bir BEL karakteri taşıyordu. Bu depoda tekrarlayan bir tuzak; düzeltme bu kez
kabuk üzerinden değil, ayrı bir Python dosyasıyla yapıldı.

**2 · CLAUDE.md olmayan bir bayrak belgeliyordu.**
`data.sampling --n 10000` — böyle bir bayrak yok, komut `unrecognized arguments`
ile düşüyor. Örneklem boyutu config'ten (`sampling.n_annotate`) geliyor.

**3 · CLAUDE.md olmayan bir modül belgeliyordu.** `analysis.descriptive` hiç
yazılmadı; karşılığı `analysis.eda` (+ `deep_eda`). Komut listesi ayrıca
yazılmış olanlarla yazılmamışları karışık tutuyordu; **ikiye ayrıldı** ve
yazılmamışlar açıkça öyle işaretlendi.

**4 · `requirements-recsys.txt` kurulduğunda ÇALIŞMAYAN bir ortam üretiyordu.**
İçindeki `-r requirements.txt` `numpy>=1.26`'yı sınırsız bırakıyor, yani numpy
2.x çekiliyor; RecBole 1.2.0 ise `np.float_` kullanıyor ve o ad numpy 2.0'da
kaldırıldı. Dosya `requirements-recbole.txt` tarafından zaten geçersiz kılınmıştı
(ölçülmüş pin'lerle, `.venv-recbole` altında çalıştığı doğrulandı: numpy 1.26.4,
recbole 1.2.0, torch 2.13.0+cpu). **Silindi**, README ve CLAUDE.md yeniden
yönlendirildi.

**5 · `requirements-llm.txt` hiç kullanılmayan bir ortamı anlatıyordu.**
Etiketlerin hiçbiri ondan gelmedi: LLM Kaggle'da koşuyor ve ortamı
`scripts/kaggle_annotate.py` kendisi kuruyor. Dosya silinmedi (yerel GPU'su olan
biri için geçerli) ama **rolü yazıldı** ve `vllm` alt sınırı Kaggle betiğiyle
hizalandı (`>=0.6` → `>=0.7`).

**6 · `utils/seeding.py` ölü koddu — ve içindeki düzeltme işe yaramıyordu.**
Hiçbir yerden çağrılmıyordu. Dahası `os.environ["PYTHONHASHSEED"] = ...` satırı
**süreç içinde etkisizdir**; o değişken yorumlayıcı başlamadan okunur. Yani modül
var olmayan bir garanti ima ediyordu. Projenin gerçek yöntemi zaten farklı ve
doğru: `seed` config'ten açıkça geçiriliyor, türetilmiş seed'ler `zlib.crc32`
ile üretiliyor (`hash()` `PYTHONHASHSEED` ile değişir). Modül **silindi**,
CLAUDE.md §8 kural 12 yöntemi anlatacak şekilde genişletildi.

**7 · Config'te ölü anahtarlar vardı — ve biri ölçüme dokunuyordu.**
`analysis.expected_peaks` DECISIONS'ta "ölçüme göre düzeltildi" diye kayıtlıydı
ama **hiçbir yerden okunmuyordu**; `eda.py` aynı değeri kendi içinde `(12, 1)`
olarak taşıyordu. Aynı şekilde `deep_eda.SUMMER = (6,7,8,9)`,
`gate1.trough_months`un kopyasıydı. Birisi config'i düzeltip figürün
değişmediğini görecekti. İkisi de config'ten okunur hâle getirildi; çıktının
**bayt bayt aynı kaldığı** doğrulandı. `dataset.meta_prefix` ve
`analysis.seasonality_months` gerçekten karşılıksızdı, silindi. Geri kalanlar
(Hafta 5/7 anahtarları) config'in kendi kuralına göre **"⚠️ HENÜZ OKUNMUYOR"**
diye işaretlendi.

**8 · Ölü anahtarın geri gelmesini engelleyen test yazıldı.**
`tests/test_config_keys.py`: her yaprak anahtar ya kodda okunuyor ya da
işaretli olmalı. Test yazılır yazılmaz bir eksik yakaladı —
`detection.secondary_sample_n`'in notu "aşağıdaki **iki** anahtar" diye düz yazı
olduğu için makine çözemiyordu; işaret her anahtarın kendi satırına taşındı.

**9 · Ölçülen bir sayı iki belgede yanlıştı.**
"Çerçeveleri havuzlamanın bedeli" üç belgede geçiyordu ve ikisinde **%15,01**
yazıyordu — hiçbir kategoride çıkmayan bir değer. Ayrıca karşılaştırma iki farklı
popülasyonu yan yana koyuyordu: %2,13 *korpusun* oranı, havuzlanmış oran ise
*örneklemin*. Yeniden ölçüldü: aynı örneklem içinde `main` **%2,02** →
havuzlanmış **%14,42**, yani **7,1×**. Dört kategoride: All_Beauty 7,1× ·
Grocery 7,7× · Video Games 3,6× · Toys 2,0×. README, GENEL_BAKIS ve DECISIONS
düzeltildi; DECISIONS'a ne değiştiği ayrıca yazıldı.

**10 · Fleiss κ commit edilen JSON'a 17 basamaklı float olarak yazılıyordu.**
Rapor değeri 4 basamağa yuvarlandı — ama **kapı ham değeri karşılaştırmaya devam
ediyor**, ayrı bir `kappa_exact` alanı üzerinden. Yuvarlanmış bir değerle
karşılaştırmak 0,59996'yı 0,60 eşiğinden geçirirdi; okunabilirlik için yapılan
bir işlem kapının kararını değiştiremez. Testi yazıldı.

#### Belgelenen sınırlar (düzeltilmedi, kayda geçti)

- **Öneri deneyi henüz gerçek etiketle koşamıyor.** `build_atomic` bir `labels`
  parametresi taşıyor ama onu dolduran CLI yolu yok ve hiçbir çağıran vermiyor.
  Bu bir eksiklik, arıza değil: zincir Hafta 4 doğrulaması → Hafta 5 damıtma +
  tam korpus çıkarımı → Hafta 6 gerçek deney. 11.800 satırlık annotation örneği
  tek başına yetmiyor; `build_atomic` kcore'un **her** satırı için etiket
  istiyor ve tutmazsa gürültülü hata veriyor.
- `data/raw` boş (~16 GB temizlendi); `download` ve `preprocess` yerelde yeniden
  koşulamıyor. Sonraki aşamalar `data/interim/` üzerinden çalışıyor.
- `detection/distill.py`, `detection/inference.py`, `recsys/marketing_metrics.py`
  yazılmadı. Config'teki karşılıkları işaretli.

#### Belgelerde yapılan yapısal değişiklik

`GENEL_BAKIS.md`'ye **§8 "Şu an ne çalışıyor, sırada ne var"** eklendi: neyin
doğrulandığı, neyin kısmi, neyin yazılmadığı, bilinen sınırlar ve **öncelik
sırasıyla sıradaki adımlar**. `PROJECT_SPEC.md` ile `ROADMAP.md`'ye "bu belge
PLANDIR, durum raporu değildir" uyarısı kondu — SPEC'in durum satırı hâlâ "ekip
onayı bekliyor" diyordu.

**Etkilediği bölüm:** `README.md`, `CLAUDE.md` (§2/§6/§7/§8), `configs/base.yaml`,
`docs/GENEL_BAKIS.md` (§5/§8/§9), `docs/PROJECT_SPEC.md`, `docs/ROADMAP.md`,
`analysis/eda.py`, `analysis/deep_eda.py`, `analysis/validation.py`,
`tests/test_config_keys.py` (yeni), `tests/test_validation.py`;
`requirements-recsys.txt` ve `utils/seeding.py` **silindi**
**Kim:** Ekip

---

### 2026-09-14 — Hafta 4 tek etiketleyiciyle kapanıyor: sapma kaydı ve ÖNCEDEN kayıt

> **Bu kayıt, A'nın etiketleri modelin etiketleriyle karşılaştırılmadan ÖNCE
> commit edildi.** A'nın dosyasına yalnızca bütünlük için bakıldı: satır sayısı,
> kimlikler, sözlük dışı etiket, metnin değişip değişmediği, marjinal etiket
> dağılımı ve notlar. LLM etiketiyle tek bir satır bile birleştirilmedi. Aşağıdaki
> her yöntem ve eşik o yüzden "sonucu görmeden" yazılmış sayılır.

#### Ne oldu

Hafta 4'ün tasarımı üç bağımsız etiketleyici ve Fleiss κ ≥ 0,60 kapısıydı
(2026-08-26, 2026-08-29). **Yalnızca A** 500 satırı etiketledi; B ve C
yapmayacak. A'nın dosyası 2026-09-14 02:28'de teslim edildi.

Bütünlük kontrolü: **500/500 dolu** · başlıklar ve `val_id` kümesi bozulmamış ·
metin hiçbir satırda değişmemiş · sözlük dışı etiket yok · dağılım `household`
170 · `self` 144 · `gift_given` 129 · `unclear` 46 · `received` 11 · 131 satırda
not var, **hepsi "kendi çocuğu"** (rehberin istediği `KENDI_COCUGU` kodu değil) ·
`EMIN_DEGIL` hiç kullanılmamış.

#### Karar — güvenilirlik ölçülmeyecek

2026-08-26 kaydı tek kişi senaryosu için bir yedek adlandırmıştı:
**intra-annotator agreement** (aynı kişinin bir alt kümeyi yeniden etiketlemesi).
Kullanıcı 2026-09-14'te bunu da, ikinci bir etiketleyiciyi de **uygulamama**
kararı verdi.

Sonuçları:

1. **Hafta 4 kapısı INCOMPLETE kalır — PASS yazılmaz, FAIL da yazılmaz.**
   Ölçülmemiş bir ölçüt `passed: null` taşır; bu projede atlanmış bir ölçüt hiçbir
   zaman geçmiş sayılmadı (Kapı 1'le aynı kural).
2. Proje bu **tarihli kararla** ilerliyor. CLAUDE.md §12'nin başarı ölçütü V1
   ("LLM ile insan etiketi arasında sınıf bazlı F1 raporlanabiliyorsa") tek
   etiketleyiciyle **karşılanabilir**; κ ise karşılanamaz.
3. Raporda sınırlılık olarak: tek kişinin yargısı referans; etiket gürültüsü
   ölçülmedi; A belirsizlik işareti hiç kullanmadı.
4. ROADMAP §11'deki "κ < 0,6 → `household`'ı `unclear` ile birleştir" yedeği ile
   rehber/`validation.py`'deki "`household` ile `gift_given` birleşir" ifadesi
   çelişiyordu. **İkisi de artık tetiklenemez** — κ ölçülmeyecek. Tek
   etiketleyicinin sonucuna bakılarak şema değiştirilmeyecek.

#### ÖNCEDEN KAYIT — Hafta 4 ölçüm yöntemi (eşik YOK)

- **Referans:** A'nın etiketi. Uzlaşı / beraberlik adımı yok.
- **Örneklem düzeyi (500 satır, ağırlıksız):** sınıf bazlı P/R/F1, makro-F1,
  doğruluk, yön yön uyuşmazlık. Doğrulama LLM etiketine göre **bilerek dengesiz**
  çekildi; bu sayılar `household`/`gift_given` yönünde yanlıdır ve öyle etiketlenir.
- **Popülasyon düzeyi (yalnızca `main` satırları, n=339, ters olasılık ağırlıklı):**
  (kategori c, LLM sınıfı k) hücresindeki bir satırın ağırlığı
  `N_main(c,k) / n_doğrulama_main(c,k)`. `n_doğrulama_main(c,k) < 5` olan hücrenin
  popülasyon kütlesi, o sınıfın **bütün kategorilerdeki** doğrulama `main`
  satırlarına eşit bölünür. Hiç doğrulama satırı olmayan sınıfın kütlesi
  "kapsanmayan pay" olarak ayrıca yazılır. Popülasyon: dört kategorinin `main`
  çerçeveleri (kategori başına 10.000, yani eşit ağırlık — RQ1'le tutarlı).
- **Güven aralığı:** yüzdelik bootstrap, 2.000 tekrar, `seed` config'ten; satırlar
  (kategori × LLM sınıfı) hücreleri içinde yeniden örneklenir. Hem ağırlıksız hem
  ağırlıklı ölçümler için.
- **Az örnekli sınıf:** insan sayısı `min_class_n_for_kappa` (20) altındaki sınıf
  raporda `sparse: true` işaretlenir (öngörülen: `received`, n=11). Hesaptan
  çıkarılmaz, işaretlenir.
- **`KENDI_COCUGU`:** "kendi çocuğu" notu ingest'te koda normalize edilir; ham not
  ayrıca saklanır. İşaretli satırlarda A'nın ve LLM'in etiket dağılımı raporlanır.
- **Sözcüksel vekil vs insan:** mevcut hediye ekseni ölçümü, değişmeden.

#### ÖNCEDEN KAYIT — insan kalibrasyonlu yaygınlık (RQ1)

Doğrulama LLM etiketine göre katmanlı çekildiği için her LLM sınıfının "insanın
gözünde gerçekte ne olduğu" tahmin edilebilir. Tanım D için
(dar = `gift_given`; geniş = `gift_given` + `household` + `received`):

- `PPV_k^D = P(A'nın etiketi ∈ D | LLM etiketi = k)` — `main` satırlarından,
  kategoriler havuzlanarak (`main` satır sayıları: `gift_given` 62 · `household`
  104 · `received` 19 · `self` 100 · `unclear` 54).
- `düzeltilmiş oran_c^D = Σ_k pay_main,c(k) · PPV_k^D`.
- Bir sınıfın `main` doğrulama satırı 5'ten azsa o sınıf için tüm çerçevelerin
  satırları kullanılır ve işaretlenir (şu an hiçbir sınıf bu durumda değil).
- GA: 2.000 bootstrap; PPV için doğrulama satırları LLM sınıfı içinde, paylar için
  `main` satırları kategori içinde yeniden örneklenir.
- **Varsayım** (PPV kategoriden bağımsız) Toys'un kendi `main` satırlarıyla
  sınanır ve raporlanır; seçim için kullanılmaz. 500 satırın tamamıyla hesaplanan
  PPV duyarlılık analizi olarak yazılır.
- `DEFINITIONS["broad"]`'a `received` eklenir (aşağıdaki C1b tanımıyla tutarlı).

#### ÖNCEDEN KAYIT — damıtma sadakat kapısı (eğitimden önce)

- Öğrenci vs öğretmen, katmanlı ayrılmış %10 LLM etiketi üzerinde:
  **C1 ekseni** (`gift_given` vs geri kalan) F1 ≥ **0,85** **ve**
  **C1b ekseni** (`gift_given`∪`household`∪`received` vs geri kalan) F1 ≥ **0,85**.
- Öğrenci vs insan (500 doğrulama satırı — eğitimden **dışlanır**): öğrencinin C1
  ekseni F1'i, öğretmeninkinden en fazla **0,05** düşük.
- Ayrılmış kümenin 5-core'a düşen kısmında aynı sayılar ayrıca raporlanır, kapıya
  girmez (clean → 5-core dağılım kaymasının ölçümü).
- ModernBERT geçemezse DeBERTa-v3 bir kez denenir; o da geçemezse deney **durur**
  ve karar kullanıcıya döner.
- 0,85 bir yargıdır ve öyle kaydediliyor: damıtmanın deneyin kullandığı ikili
  kararlara %15'ten fazla F1 kaybı eklememesi.

#### ÖNCEDEN KAYIT — Kapı 2 (deney koşularından önce)

Kurulum geçerliliği ölçer, **etkiyi ölçmez**. Kategori × model başına:

1. Test (kullanıcı, ürün) çiftleri bütün koşullarda **birebir aynı** (hash).
2. Her koşulun gerçek ürün evreni C0'ın **alt kümesi** (gölge sonek soyulup).
3. **Plasebo taban çizgisini geçmiyor:** kullanıcı başı Recall@10 farkının
   (C4 − C0) eşli bootstrap %95 GA'sının alt ucu ≤ 0. Aynısı C4b − C0 için.
   Rastgele veri silmenin anlamlı iyileştirme getirmesi kurulum hatasına işaret.
4. **Seed kararlılığı:** C0 Recall@10'un seed'ler arası değişim katsayısı < 0,10.

Karar deseni Kapı 1'le aynı: eksik koşu/seed varsa INCOMPLETE.

#### Koşul tanımları (CLAUDE.md §13 TBD'leri kapanıyor)

| kod | ne yapar | rol |
|---|---|---|
| C0 | bütün etkileşimler | taban |
| **C1** | `gift_given` eğitimden çıkar | **birincil** (RQ2) |
| C4 | C1 kadar rastgele satır çıkar | C1'in plasebosu |
| **C3** | `gift_given` satırları eğitimde **gölge token** olur | **RQ3** |
| C1b | `gift_given` + `household` + `received` çıkar | sağlamlık |
| C4b | C1b kadar rastgele satır çıkar | C1b'nin plasebosu |
| C2 | — | **uygulanmadı**; çağrılırsa açık hata |

- **`household` ve `received`:** kavramsal karar kullanıcıda (2026-09-14). Birincil
  tanım literatürdeki hediye (`gift_given`); "alıcı ürünü kendisi seçmedi" tanımı
  (C1b) sağlamlık kontrolü. İkisi de raporlanır.
- **C4b neden var:** C1b, C1'in yaklaşık iki katı satır çıkarıyor. C1b'yi C4'le
  karşılaştırmak iki farklı veri kaybını karıştırırdı.
- **C3 — "feature mı, ayrı token mı" → AYRI TOKEN.** Eğitim bölümündeki
  `gift_given` satırlarının `item_id`'si `<id>::gift` olur; valid/test satırlarına
  dokunulmaz; değerlendirmede gölge ürünlerin skoru `-inf` yapılır. Gerekçe:
  bayrağı feature olarak eklemek hediye ürününü eğitimde **tahmin hedefi**
  bırakırdı — kirlilik çıkış katmanından öneri listesine geri sızardı. Gölge token
  hediye olayını sekansta tutar ama gerçek ürünün ne gömülmesini ne hedefini
  kirletir. Bedeli: gölge ile gerçek ürün gömülmesi bilgi paylaşmaz. C3, C1 ile
  **aynı etiket kümesini** kullanır (RQ3 "silmek mi söylemek mi" karşılaştırması).
- **C2:** mevcut kod bir `weight` kolonu ekliyordu ama hiçbir şey onu okumuyordu;
  koşulsa C0'ın aynısı eğitilir ve "C2" diye raporlanırdı. `NotImplementedError`
  ile kilitleniyor.

#### Analiz karşıtlıkları (yön ne çıkarsa çıksın raporlanır)

- **RQ2:** C1 − C4 (asıl), C1 − C0 · doz–yanıt: Toys etkisi vs Grocery etkisi
- **RQ3:** C3 − C1, C3 − C0
- **Sağlamlık:** C1b − C4b
- Hepsi seed'ler üzerinden ortalanmış kullanıcı başı metriklerle **eşli bootstrap**.

**Etkilediği bölüm:** `configs/base.yaml` (`validation`, `distill.fidelity`,
`gate2`, `experiment.conditions`), CLAUDE.md §13, `implementation-plan.md` §5.4
**Kim:** Kullanıcı (kararlar) · Ekip

---

### 2026-09-14 — Hafta 4 sonuçları: model vekilden iyi, C1 etiketi gürültülü, RQ1 kalibre edildi

> Yöntemler `8c697a8`'de, onları uygulayan kod `eed7f35`'te — ikisi de bu sonuçlar
> üretilmeden **önce** push'landı. Aşağıdaki hiçbir eşik ya da tanım sonuca
> bakılarak değiştirilmedi. Sonuçtan sonra eklenen tek analiz aşağıda **post-hoc**
> diye işaretli ve birincil sayının yerine geçmiyor.

#### Karar: INCOMPLETE (önceden kayıtlı)

Tek etiketleyici; güvenilirlik ölçülmedi. Kapı ölçütü `passed: null`.

#### LLM vs A — beş sınıf

| | örneklem (500, ağırlıksız) | popülasyon (`main`, ağırlıklı, 339 satır) |
|---|---|---|
| doğruluk | 0,640 | **0,727** [0,661–0,789] |
| makro-F1 | 0,540 | 0,494 [0,438–0,558] |

| sınıf | örneklem F1 | popülasyon K | popülasyon D | popülasyon F1 |
|---|---:|---:|---:|---:|
| `gift_given` | 0,731 | 0,649 | 0,799 | 0,716 |
| `self` | 0,582 | 0,776 | 0,909 | 0,837 |
| `household` | 0,757 | 0,764 | 0,485 | 0,594 |
| `unclear` | 0,359 | 0,332 | 0,148 | 0,205 |
| `received` (az örnekli, n=11) | 0,274 | 0,091 | 0,180 | 0,121 |

Makro-F1'i iki küçük/bulanık sınıf aşağı çekiyor. En büyük uyuşmazlıklar:
**A `household` → LLM `gift_given` 33** (bilinen sınır) · A `self` → LLM `unclear` 32 ·
A `self` → LLM `received` 18. `KENDI_COCUGU` işaretli 131 satırın **hepsi** A'da
`household`; LLM 102'sinde aynı, 19'unda `gift_given` diyor.

Sözcüksel vekil (hediye ekseni): K 0,41 · D 0,51 · **F1 0,46**. LLM belirgin iyi.
Vekilin önceki 0,58/0,61 sayıları "yazar destekli ön geçiş"ti; bu ilk bağımsız ölçümü.

#### Deneyin kullandığı ikili eksenler — öğretmen vs A

| eksen | örneklem F1 [GA] | popülasyon K · D · F1 [GA] |
|---|---|---|
| **C1** `gift_given` | 0,731 [0,672–0,783] | 0,649 · 0,799 · **0,716** [0,618–0,805] |
| **C1b** `gift_given`+`household`+`received` | 0,889 [0,865–0,910] | 0,842 · 0,742 · **0,789** [0,719–0,855] |

Damıtma kapısının önceden kayıtlı üçüncü ölçütü öğretmenin **örneklem C1 F1'ini**
(0,731) referans alıyor: öğrenci ≥ 0,681 olmalı.

#### RQ1 — insan kalibrasyonlu yaygınlık

| kategori | dar ham | **dar kalibre** [GA] | *post-hoc* | geniş ham | **geniş kalibre** [GA] | *post-hoc* |
|---|---:|---|---:|---:|---|---:|
| Toys | 23,25 | **18,99** [15,76–22,31] | *21,86* | 44,02 | **44,22** [40,41–48,35] | *54,37* |
| Video Games | 7,86 | **7,90** [5,83–10,57] | *4,02* | 15,23 | **22,74** [17,81–28,10] | *12,97* |
| Grocery | 4,28 | **5,43** [3,39–8,26] | *2,61* | 9,57 | **18,29** [13,38–24,05] | *14,02* |
| All Beauty | 4,70 | **5,54** [3,44–8,41] | *4,05* | 8,84 | **17,97** [12,81–23,45] | *6,86* |

Geniş ham oranlar önceki kayıttan (%43,42 vb.) biraz yüksek: `received` artık geniş
tanımda.

**Okuma:**
1. Toys'un ham dar oranı (%23,25) kalibre GA'nın **dışında**. Deneme koşusundan beri
   yazılı "üst sınır" uyarısı ölçümle doğrulandı.
2. Doz–yanıt: Toys hâlâ ayrık. **Video Games artık alt gruptan ayrılamıyor** (GA'lar
   örtüşüyor) — ham oranlarda ayrıktı.
3. **Kalibrasyonun varsayımı Toys'ta tutmuyor** (önceden kayıtlı kontrol): kendi
   PPV'siyle geniş %54,37, havuzlanmış PPV'yle %44,22. Düşük hediyeli kategorilerde
   geniş oranın ~ikiye katlanması büyük ölçüde havuzlanmış `PPV_geniş(self) = 0,12`'den
   geliyor — o değeri Toys'un çocuk alımları taşıyor.
4. **Post-hoc duyarlılık** (sonuç görüldükten sonra eklendi, birincil DEĞİL): her
   kategorinin kendi PPV'siyle geniş oran All Beauty'de %6,86, Grocery'de %14,02. Kategori
   başına doğrulama satırı az olduğu için gürültülü; ama birincil geniş oranların düşük
   hediyeli kategorilerde **yukarı yanlı olabileceğini** gösteriyor. Rapor aralığı
   ikisiyle birlikte verir.

#### Plana etkisi

- **Önceden kayıt tutuyor.** F1 bir ölçümdü, kapı değil; proje ilerliyor.
- **Prompt v4 yok; 500 etiket hiçbir ayar için kullanılmaz.** Kullanılırsa projenin tek
  bağımsız referansı yok olur — deneme setinin F1 olarak raporlanamamasıyla aynı hata.
- **C1b önceliği (kullanıcı, 2026-09-14, deney koşulmadan):** C1 birincil kalıyor; ama
  C1b ve C4b artık kesme sırasında **değil**, çekirdekte. C1'in etiketi gürültülü; C1'de
  fark çıkmazsa daha iyi ölçülen eksen elimizde olmalı. Tanım değişmedi, öncelik değişti.

#### ÖNCEDEN KAYIT — deney sonuçlarının yorum kuralı (HİÇBİR DENEY KOŞULMADAN)

Etiket gürültüsü koşulu plaseboya doğru çeker: C1'in çıkardığı satırların popülasyonda
~%35'i hediye değil, hediyelerin ~%20'si hiç çıkarılmıyor.

- **C1 ≈ C4** (eşli bootstrap GA sıfırı içeriyor) → *"bu etiket hassasiyetiyle
  saptanamadı"*. **"Etki yok" yazılmaz.**
- **C1 > C4** (GA sıfırın üstünde) → gerçek etkinin **alt sınırı**.
- **C1 < C4** → olduğu gibi raporlanır, yorumlanmaya zorlanmaz.
- Aynı kural C1b/C4b için. Sonuç tablosunda her eksenin ölçülen K/D değeri yanında durur.

#### Bu fazda bulunan ve düzeltilen hatalar

1. **Makro-F1 şişiriliyordu.** `compare()` `if precision and recall` diyordu; TP=0 olan
   sınıfta precision `0.0` (falsy) olduğu için F1 `None` oluyor ve sınıf ortalamadan
   **sessizce düşüyordu** — tamamen kaçırılan bir sınıf makro-F1'i yükseltiyordu. Aynı
   desen ağırlıklı ve ikili ölçümde de vardı. `F1 = 2TP/(2TP+FP+FN)`'ye geçildi. Gerçek
   nokta tahminleri etkilenmedi (her sınıfta en az bir TP vardı), ama **ağırlıklı
   makro-F1'in GA üst ucu 0,629 → 0,558** düştü: bootstrap tekrarlarında `received`'ın
   TP=0 kaldığı örneklerde sınıf ortalamadan atılıyordu. Hata sonuçlar commit edilmeden
   bir test tarafından yakalandı.
2. **Kalibrasyon GA'ları işleme sırasına bağlıydı.** Tek rng kategorilere sırayla
   dağıtılıyordu; `--category all` ile tek tek çağrı farklı GA veriyordu. Seed artık
   (kategori, tanım) başına türetiliyor (`sampling._stratum_seed` deseni). Testi var.
3. **Rapor yanlış korpusu söylüyordu** — "oranlar k-core korpusuna aittir". Örnekleme ve
   vekil **clean** korpustan okuyor; Toys annotation satırlarının yalnızca %17,6'sı
   5-core'da. Bu cümleyi önceki bir oturumda doğrulamadan yazmıştım.
4. **Bir test tesadüfen geçiyordu** — "geniş − dar = household" `received` eklenince
   yanlıştı ama fixture'ın küçük `main` çerçevesi `received` etiketine hiç ulaşmadığı için
   geçiyordu. Kurgu düzeltildi; test artık kendi kurgusunu da doğruluyor.

Kontaminasyon tanımları tek yere taşındı: `detection.schema.CONTAMINATION`. Yaygınlık,
doğrulama eksenleri ve (Faz 3'te) deney koşulları aynı kümeyi okuyor.

**Etkilediği bölüm:** `analysis/validation.py`, `analysis/prevalence.py`,
`detection/schema.py`, `reports/results/validation_500.json`, `prevalence.json`, F18, F19
**Kim:** Kullanıcı (C1b önceliği) · Ekip

---

### 2026-09-14 — Hafta 5 kodu: damıtma ve tam korpus çıkarımı (HİÇBİR MODEL EĞİTİLMEDEN)

> Sadakat kapısının eşikleri `8c697a8`'de. Bu kayıttaki her karar **eğitimden önce**
> verildi; hiçbir öğrenci eğitilmedi, hiçbir öğrenci tahmini görülmedi.

#### Ne kuruldu

| adım | nerede | ne yapar |
|---|---|---|
| `distill prepare` | yerel, CPU | LLM etiketi + örnek metni → eğitim/ayrılmış paket; 500 doğrulama satırı `(category,row_id)` ile dışlanır |
| `distill train` | Kaggle, tek T4 | ModernBERT-base, sabit 3 epoch, **model seçimi yok** → sadakat kapısı |
| `inference prepare` | yerel, CPU | 5-core satırı → öğretmenin gördüğü dört alan |
| `inference run` | Kaggle, kategori başına bir T4 | 5-core'un her satırına etiket + 5 sınıf olasılığı; 200.000'lik parçalar, kaldığı yerden devam |

Akış tek hücrede: `scripts/kaggle_distill.py`. Kapı PASS değilse çıkarım adımı koşmaz.

#### Ölçülen (yerelde, gerçek veriyle)

- 47.200 LLM etiketi − 500 doğrulama satırı − 45 ayrıştırma hatası = **46.655**
  (eğitim 41.988 · ayrılmış 4.667; katman = kategori × etiket). Ayrılmış kümenin 5-core'a
  düşen kısmı **546** satır — dağılım kayması ölçümü bu kadar satıra dayanacak.
- **Sızıntı koddan bağımsız sayıldı:** paket ∩ doğrulama `(category,row_id)` = **0**;
  500 insan satırının metniyle birebir aynı metin taşıyan paket satırı = **0**.
- Eğitim etiketleri: `self` 26.877 · `gift_given` 7.868 · `household` 4.260 ·
  `unclear` 2.482 · `received` **501**. `received` öğrenci için de zayıf sınıf olacak.
- Çıkarım girdisi: Toys **2.164.018** (22 satırda ürün adı yok) · Grocery **2.434.594**
  (393). Kimlik kümesi 5-core'la birebir. Aynı satır için eğitim paketindeki metin ile
  çıkarım girdisindeki metin **aynı** (300/300 örnek) — öğrenci eğitimde ve çıkarımda aynı
  girdiyi görüyor. Öğrenci metninin medyanı ~305 karakter.

#### Eğitimden önce verilen kararlar

1. **Rapor model başına** (`distill_report_base.json`, `distill_report_fallback.json`).
   Tek dosya olsaydı yedek modelin koşusu ana modelin FAIL kanıtını ezerdi.
2. **"Yedek yalnızca ana model FAIL olursa, bir kez" kuralı KODDA.** Önceden kayıtlı kural
   (bu günün önceki kaydı) yalnızca yazılıydı; `train --model fallback` artık ana modelin
   raporu FAIL değilse duruyor (INCOMPLETE de dahil). İki modeli eğitip iyisini seçmek,
   kapıyı aynı kümede ölçerken model seçmek olurdu.
3. **Eğitim tek kartta, DDP değil** (CLAUDE.md §6'daki öngörüden sapma). 42K satırlık
   eğitimde kazanç küçük; ve iki kart görünürken Trainer sessizce DataParallel'e geçip
   etkin grup boyunu 64'e çıkarıyor — config'te yazan 32 koşmamış olurdu. Raporda
   `n_gpu` ve `effective_batch_size` ayrıca yazılıyor. Çıkarım **kategori başına bir kart**.
4. **`distill.gradient_checkpointing: true`.** `distill.max_length: 1024`'ün "bedava"
   gerekçesi (2026-08-25) ModernBERT'in dolgu kaldırmasına dayanıyordu; o yalnızca
   FlashAttention'la çalışıyor ve T4'te FlashAttention yok. 32 × 1024 token'lık bir grup
   kaba hesapla 16 GB'ı aşıyor (**ölçülmedi**). Tavan 1024'te kaldı — öğretmen 6.000
   karakter gördü ve girdi eşitliği sadakat ölçümünün ön koşulu. Checkpointing sonucu
   değiştirmez, eğitimi ~%30 yavaşlatır.
5. **`warmup_ratio: 0.1`, `weight_decay: 0.01` koddan config'e taşındı** ("kodda
   hardcode parametre yok"). Değerler transformers'ın yaygın varsayılanları; ayar yapılmadı.
6. **Tahminde token bütçeli gruplar** (grup başına ≤ 65.536 token). Sabit 256 satırlık
   grup, uzunluğa göre sıralı listenin sonunda 256 × 1024 token kurardı. Mühendislik
   ayarı: CPU'da aynı satırların tek seferde ve parça parça tahmini arasındaki en büyük
   olasılık farkı 1,2e-7.
7. **transformers sürümüne bağımsızlık.** Kaggle imajındaki sürüm bilinmiyor; v5
   `warmup_ratio`'yu ve `group_by_length`'i yeniden adlandırdı. Yanlış ad, model
   indirildikten **sonra** hata verirdi. Argümanlar kurulu sürümün alanlarına bakılarak
   kuruluyor (testi var).

#### Doğrulanan / doğrulanmayan

- **Doğrulandı:** 331 test (stub öğrenciyle sızıntı, kapı mantığı, devam ettirme, eksik
  satır hatası) · CPU duman testi, transformers 5.17 + torch 2.14: gerçek ModernBERT-base
  ve deberta-v3-base config'lerinin küçültülmüş, **rastgele ağırlıklı** kopyalarıyla
  eğitim → kapı → rapor → yedek kuralı → 3 parçalı çıkarım uçtan uca geçti. DeBERTa SDPA
  desteklemediği için eager dikkate düşüyor (beklenen).
- **Doğrulanmadı:** T4 üzerinde fp16 davranışı, bellek, gerçek hız. Eğitim kaybı NaN
  olursa kod durur ve `FP16 = False` ile yeniden koşmayı söyler.

#### Bilinen sınırlar

- Öğrenci **clean** korpusun örneğiyle eğitiliyor, **5-core**'u etiketliyor. Ayrılmış
  kümenin 5-core kısmı (546 satır) ayrıca raporlanıyor ama kapıya girmiyor.
- Çıkarım raporundaki "LLM'le örtüşme" **iyimser**: o satırların çoğu eğitimdeydi.
  Tarafsız sayı damıtma raporundaki `student_vs_teacher_holdout_in_kcore`.

**Etkilediği bölüm:** `detection/distill.py`, `detection/inference.py`,
`scripts/kaggle_distill.py`, `configs/base.yaml` (`distill`, `paths.models`),
`reports/results/distill_bundle.json`, CLAUDE.md §2/§6/§7/§10, README, GENEL_BAKIS §6/§8
**Kim:** Ekip

---

### 2026-09-14 — Hafta 6 kodu: koşullar gerçek etikete bağlandı, C3 gölge token, değerlendirme maskesi (HİÇBİR DENEY KOŞULMADAN)

> Damıtılmış etiketler henüz yok (Kaggle koşusu bekliyor), yani gerçek etiketle **hiçbir**
> deney koşulmadı. Bu kayıttaki her karar sonuç görülmeden verildi ve yalnızca sentetik
> duman verisiyle sınandı.

#### Ne kuruldu

| parça | ne yapar |
|---|---|
| `detection/contamination.py` | C1 (`narrow`) ve C1b (`broad`) etiket kümeleri **tek yerde**, bağımlılıksız. `schema` yeniden dışa aktarıyor; yaygınlık, doğrulama ve koşullar aynı kümeyi okuyor |
| `recsys/conditions.py` | C0 · C1 · C4 · **C1b** · **C4b** (C1b kadar rastgele eğitim satırı) · **C3** (gölge token) · C2 → `NotImplementedError` · raporlanabilir kaynak `{llm, distilled}` · bayat koşul koruması |
| `recsys/atomic.py` | `--labels distilled`: öğrencinin 5-core etiketleri, üç kontrolle (aşağıda) · bayat atomic koruması · vekil etiketi `proxy` dışında bir adla damgalanamaz |
| `recsys/run_experiment.py` | gölge ürün maskesi · C0 alınmış ürün maskesi · kullanıcı başı metrik + top-K (parquet) + RecBole toplamıyla tutarlılık kontrolü · SASRec yolu |

#### Koşudan önce verilen kararlar

1. **C2 kilitli.** Önceki kod bir `weight` kolonu ekliyordu ama RecBole onu hiç okumuyordu —
   koşulsa C0'ın aynısı "C2" diye raporlanırdı. Artık çağrılırsa açık hata veriyor.
   `experiment.soft_weights` bu yüzden config'ten kaldırıldı.
2. **C3 = gölge token.** C1'in etiket kümesi (`gift_given`) aynen kullanılır — RQ3'ün
   "silmek mi, söylemek mi" karşılaştırması aynı satırlar üzerinde. Yalnızca **eğitim**
   satırlarında ürün `<id>::gift` olur; valid/test'e dokunulmaz. Sıralı modelde gölge
   token geçmişte kalır ve eğitimde hedef de olabilir (gölge gömmesi böyle öğreniliyor);
   gerçek ürünün gömmesi ve hedefi kirlenmez. Değerlendirmede (valid + test) gölge
   ürünlerin skoru `-inf`. Gerçek kimliğiyle eğitimden kaybolan ürün C1'deki gibi sayılır.
3. **C4b'nin satır sayısı C1b'den türetilir**, elle girilmez; seed `crc32` ile koşul
   adından (`conditions._seed_for`).
4. **Alınmış ürün maskesi bütün koşullarda C0'ın (genel modeller: BPR).** Duman testinde
   bulundu. RecBole tam sıralamada kullanıcının eğitimde gördüğü ürünleri öneriden
   çıkarıyor — ama **koşulun kendi** eğitiminden. C1/C4/C1b/C4b'de silinen satırın ürünü
   ve C3'te gölgeye dönen gerçek kimlik maskeden düşüp top-K'da yer kaplıyordu. Yani koşul
   yalnızca eğitimi değil **değerlendirmeyi** de değiştiriyordu; ve bu C1−C4'te bile
   simetrik değil (silinen hediye ürünleri rastgele silinen ürünlerden farklı bir
   popülerlik dağılımına sahip olabilir). Düzeltme: C0 eğitiminde olup koşulun eğitiminde
   gerçek kimliğiyle olmayan (kullanıcı, ürün) çiftleri maskeye eklenir. **C0'ın sonucu
   değişmez.** Maskelenen bir çift valid/test pozitifiyle çakışırsa koşu durur (kullanıcı
   + ürün tekil olduğu için olmamalı); maskedeki ürün top-K'ya girerse koşu durur.
   Sentetik veride etkisi küçük değildi: BPR × C3 Recall@10 0,1016 → 0,1049 (bu bir sonuç
   değil, yalnızca maskenin sayıyı değiştirdiğinin kanıtı). **Sıralı modellerde RecBole hiç
   maskelemiyor** — her koşulda aynı, dokunulmadı.
5. **SASRec yolu — SASRec bu projede hiç koşmamıştı.** İki kurulum hatası:
   (a) SASRec'in kaybı CE ve RecBole'un genel varsayılanı (1 negatif) CE ile birlikte
   verilince kurulumda hata veriyor → `train_neg_sample_args: None`, modelin kendi
   `properties` dosyasındaki `loss_type`'tan okunarak. (b) `benchmark_filename` verilince
   RecBole sekansları kendisi genişletmiyor; her satır hazır `item_id_list` taşımalı →
   `run_experiment.sequential_parts`, RecBole'un kendi genişletme kuralıyla (eğitim: her
   ürün önceki eğitim ürünleriyle; valid: bütün eğitim ürünleri; test: eğitim + valid).
6. **`experiment.max_item_list_length: 50`** (RecBole varsayılanı). Kırpma bizim tarafta ve
   **son** 50 ürün tutuluyor; RecBole'un `seq_len`'i ilk N'i tutardı (`dataset.py`'de
   `seq[:seq_len]`, okundu).
7. **Boş geçmişli valid satırı.** C1/C1b bir kullanıcının bütün eğitim satırlarını
   silebilir; SASRec boş geçmişi işleyemez. O satır **yalnızca erken durdurma setinden**
   düşer ve `n_valid_dropped_empty_history` olarak raporlanır. Test satırının geçmişi
   valid ürününü içerdiği için hiçbir zaman boş değil — test çiftleri koşullar arasında
   aynı kalıyor (boşsa koşu durur).
8. **Kullanıcı başı çıktı.** Eşli bootstrap ve M1/M2 kullanıcı başı sayı istiyor, RecBole
   yalnızca ortalama veriyor. Test değerlendirmesi sırasında RecBole'un **kendi** skor
   tensöründen aynı `topk` çağrısıyla yakalanıyor; kullanıcı başı ortalama RecBole'un
   toplamına 5e-7 içinde eşit değilse (`metric_decimal_place: 6`) koşu durur. Parquet
   `data/processed/recbole/<kategori>/peruser/` altında (git'e girmez); rapora yalnızca
   dosya **adı** yazılıyor, mutlak yol değil.
9. **`atomic --labels distilled` üç kontrol:** etiketlerin kaynağı gerçek model (stub
   değil); çıkarım raporu kapıyı PASS kaydetmiş; **ve** o modelin kendi damıtma raporu
   diskte PASS diyor. Satır sayısı raporla tutmazsa durur.
10. **Bayat dosya korumaları.** Vekille üretilmiş atomic dosya varken `--labels distilled`
    istenirse ya da koşul dosyası başka bir etiket kaynağından/satır sayısından kalmışsa
    "zaten var" diye atlanmıyor, `--force` isteniyor. Atlansaydı deney sessizce vekil
    etiketle koşardı.

#### Bulunan hata: koşucu RecBole ortamında ilk satırda çöküyordu

`conditions` etiket kümesini `detection.schema`'dan okumaya başlayınca import zinciri
pydantic'i çekti; `.venv-recbole`'da pydantic yok. Ana ortamdaki testler bunu göremezdi,
ancak RecBole duman testi yakaladı. Kümeler bağımlılıksız `detection/contamination.py`'ye
taşındı; koşucunun pydantic/matplotlib/statsmodels/scipy çekmeden import edildiği taze bir
yorumlayıcıda testle kilitli.

#### Doğrulanan / doğrulanmayan

- **Doğrulandı:** 357 test (`.venv`) · RecBole duman testi (`.venv-recbole`, CPU, sentetik
  400 kullanıcı / 120 ürün, 3 epoch, `label_source: distilled` damgalı sentetik etiket):
  BPR ve SASRec × C0/C3/C4b. Altı koşunun altısında kullanıcı başı ortalama RecBole
  toplamına **birebir** eşit (en büyük fark 0,0) · 119 gölge ürün maskelendi, top-K'da 0 ·
  C0 maskesine eklenen çift sayısı C3'te gölgeye dönen satır sayısına (618) ve C4b'de
  silinen satır sayısına (862) eşit · C0'ın sayısı maske eklenmeden öncekiyle aynı.
- **Doğrulanmadı:** gerçek veriyle hiçbir koşu, GPU, süre. Genel modellerde RecBole'un
  `eval_batch_size`'ı (varsayılan 4.096) ürün sayısına bölünüyor; ürün sayısı 4.096'yı
  aşarsa tam sıralama **kullanıcı başına bir grup** koşar ve yavaş olabilir. Zaman sondası
  (Faz 4.1) ölçecek; bu sonuçtan bağımsız bir mühendislik parametresi.

#### Bilinen sınırlar

- C3'te gölge ürün ile gerçek ürünün gömmesi bilgi paylaşmıyor (tasarımın bedeli).
- SASRec son 50 ürünü görüyor; daha uzun geçmişli kullanıcının erken ürünleri modele girmez
  (her koşulda aynı kural).
- C1/C1b'de gerçek kimliğiyle eğitimden tamamen kaybolan ürün değerlendirmede hâlâ cold
  item; sayısı koşul raporunda (`n_items_lost_from_training`).

**Etkilediği bölüm:** `detection/contamination.py`, `detection/schema.py`,
`recsys/atomic.py`, `recsys/conditions.py`, `recsys/run_experiment.py`,
`configs/base.yaml` (`experiment.conditions`, `experiment.max_item_list_length`,
`soft_weights` kaldırıldı), testler (`test_conditions`, `test_no_leakage`,
`test_experiment_labels`, `test_run_experiment`), CLAUDE.md §2/§7/§10, README,
GENEL_BAKIS §6/§8
**Kim:** Ekip

---

### 2026-09-14 — Faz 4 kodu: Kapı 2 değerlendiricisi, eşli bootstrap, Kaggle deney betiği (HİÇBİR DENEY KOŞULMADAN)

> Kapı 2'nin ölçütleri ve karşıtlıklar bu günün ilk kaydında önceden yazılmıştı. Aşağıdakiler
> o kaydın **yazmadığı uygulama ayrıntıları**; hepsi gerçek etiketle tek bir koşu yapılmadan,
> sentetik veriyle sınanarak verildi.

#### Ne kuruldu

| parça | ne yapar |
|---|---|
| `analysis/experiment_stats.py` | koşu raporları + kullanıcı başı dosyalar → `gate2.json` (kategori × model başına dört ölçüt) ve `experiment_stats.json` (karşıtlıklar, doz–yanıt, etiket kalitesi) |
| `recsys/run_experiment.py` | test çiftlerinin özeti · koşu süresi · kod sürümü · bitmiş koşuyu atlama (`--force`) · iki GPU hatası (aşağıda) · checkpoint silme |
| `scripts/kaggle_experiment.py` | RecBole'u ayrı klasöre kurar, koşulları atomic dosyadan üretir, koşuları iki T4'e dağıtır; `PLAN = "probe"` zaman sondası (Faz 4.1), `"matrix"` matris |

#### Uygulama kararları

1. **Kapı 2, ölçüt 1:** test (kullanıcı, ürün) çiftlerinin sıra bağımsız sha256 özeti, koşu
   anında RecBole'a verilen test dosyasından. Bir model içinde bütün koşullar ve seed'ler
   aynı olmalı. (Duman testinde BPR ile SASRec'in özeti de aynı çıktı.)
2. **Ölçüt 1–2 hangi koşullara bakar:** çekirdek C0 · C1 · C4 · C1b · C4b koşmamışsa ölçüt
   `null`. C3 koştuysa kontrole girer; koşmadıysa kapıyı bekletmez — Grocery'de C3 kesme
   sırasında.
3. **Ölçüt 3:** kullanıcı başı metrik önce iki koşulun **ortak** seed'leri üzerinden ortalanır,
   sonra kullanıcılar yeniden örneklenerek eşli bootstrap (`gate2.bootstrap_n`, `gate2.ci`).
   Bootstrap seed'i `crc32` ile (kategori, model, karşıtlık) adından.
4. **Ölçüt 4:** RecBole'un C0 test metriği; config'teki **her** seed koşmuş olmalı (en az 2),
   yoksa `null`. Değişim katsayısı = örneklem standart sapması (ddof=1) ÷ ortalama.
5. **Karşıtlıklar** (C1−C4, C1−C0, C3−C1, C3−C0, C1b−C4b): `experiment.bootstrap_iters`
   (1.000) ve yeni anahtar `experiment.ci` (0,95). Bütün metrikler **aynı** kullanıcı
   örneklemiyle yeniden örneklenir. Göreli fark da yazılır. Plasebo karşıtlıklarına (C1−C4,
   C1b−C4b) önceden kayıtlı yorum etiketi eklenir — `alt_sinir` / `saptanamadi` / `negatif`;
   diğerlerine yalnızca yön.
6. **Doz–yanıt:** plasebo karşıtlığının kullanıcı başı farkı Toys'ta ve Grocery'de; iki
   kategori farklı kullanıcılar olduğu için **bağımsız** bootstrap, fark GA'sı. Önceden kayıt
   C1−C4'ü adlandırıyordu; aynı hesap C1b−C4b için de yazılıyor (sonuç görülmeden eklendi).
7. **Kapı 2 PASS değilse** o kategori × modelin karşıtlıkları yine hesaplanır ama
   `interpretable: false` damgası taşır.
8. Sonuç dosyasında iki eksenin Hafta 4'te ölçülen popülasyon K/D/F1'i yanında durur
   (yorum kuralının istediği gibi). Koşulardan biri bile `llm`/`distilled` dışındaysa
   iki dosya da `reportable: false`.

#### GPU koşusundan ÖNCE RecBole kaynağını okuyarak bulunan hatalar

1. **İki paralel koşu aynı karta düşerdi.** RecBole `gpu_id`'yi (varsayılan `'0'`)
   `CUDA_VISIBLE_DEVICES`'a **yazıyor**; ikinci karta verilen koşu sessizce birinci karta
   geçerdi. `gpu_id` artık ortamdaki değerden veriliyor.
2. **BPR'nin tam sıralaması kullanıcı başına bir grup koşardı.** RecBole'un
   `eval_batch_size`'ı (4.096) genel modellerde ürün sayısına bölünüyor; ölçülen 5-core ürün
   sayısı Toys **103.618**, Grocery **94.863** (kullanıcı 268.652 / 268.991). Her doğrulama
   turu ~269 bin grup olurdu. `experiment.eval_score_cells: 67108864` (grup başına en fazla
   64M skor hücresi ≈ 256 MB) — mühendislik ayarı; sıralama skorları aynı.
3. **Koşu idempotent değildi** (CLAUDE.md §7). Rapor ve kullanıcı başı dosya varsa atlanıyor.
4. **Checkpoint'ler diski doldururdu** (~70 koşu × 100–300 MB, Kaggle 20 GB). Model yeniden
   kullanılmıyor — M1/M2 kullanıcı başı top-K listelerinden — koşu bitince siliniyor.
5. `code_version` `utils.io`'ya taşındı: RecBole ortamı `llm_annotate`'i (pydantic) import
   edemez.

#### Kaggle kurulumu

Kaggle imajının numpy 2 / pandas 3 yığını RecBole'u ilk satırda çökertiyor. Tarif: `.venv-
recbole`'un `pip freeze`'indeki pinler (torch hariç) `pip install --no-deps --target` ile ayrı
bir klasöre; alt süreçlerde `PYTHONPATH`'in başına. **Yerelde sınandı:** torch 2.14 + numpy
2.5 + pandas 3 kurulu bir ortamda bu klasörle 24 duman koşusu `.venv-recbole` (torch 2.13) ile
**birebir aynı** sayıları verdi. Betiğin akışı yerelde CPU'da tek işçiyle simüle edildi:
kurulum kontrolü → koşul üretimi → sonda → küçük matris → önceki oturumun çıktısını atlama.

#### Doğrulanan / doğrulanmayan

- **Doğrulandı:** 374 test · 24 koşuluk RecBole duman testi (BPR + SASRec × altı koşul × iki
  seed, sentetik): özet hepsinde aynı, kullanıcı başı ortalama RecBole'a eşit, bitmiş koşu
  1,4 sn'de atlandı; `experiment_stats` bu çıktılardan `gate2.json`/`experiment_stats.json`
  üretti (sentetik 3 epoch'ta seed kararlılığı beklendiği gibi FAIL).
- **Doğrulanmadı:** Kaggle imajında kurulum, CUDA, iki GPU'nun paralel işleyişi, gerçek
  boyutta süre ve bellek. Zaman sondası (Kaggle, `PLAN = "probe"`) matrisi bütçelemeden önce
  koşulacak.

**Etkilediği bölüm:** `analysis/experiment_stats.py`, `recsys/run_experiment.py`,
`utils/io.py`, `detection/llm_annotate.py`, `scripts/kaggle_experiment.py`,
`configs/base.yaml` (`seeds`, `experiment.categories/ci/eval_score_cells`, `gate2` okunuyor),
`tests/test_experiment_stats.py`, CLAUDE.md §2/§7/§10, README, GENEL_BAKIS §8
**Kim:** Ekip
