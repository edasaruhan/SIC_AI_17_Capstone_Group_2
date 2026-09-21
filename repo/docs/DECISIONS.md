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

---

### 2026-09-14 — Faz 5 kodu: pazarlama metrikleri M1 / M2 / M3 — tanımlar ÖNCEDEN kayıt (HİÇBİR DENEY KOŞULMADAN)

> CLAUDE.md §13'te M2'nin operasyonel tanımı TBD'ydi ve "kafaya göre doldurulmasın, sorulsun
> ya da **varsayıldı** notuyla yazılsın" deniyordu. Aşağıdaki tanımlar **varsayıldı**:
> gerçek etiketle tek bir deney koşulmadan, sentetik veriyle sınanarak yazıldı. Ekip
> değiştirmek isterse sonuçlar görülmeden, bu kayda tarihli bir ekle yapılmalı.

#### Ortak tanımlar

- **Geçmiş H_u** = C0 dosyasındaki `train` + `valid` satırları (test ürünü geçmiş değil).
  Geçmiş **her koşulda C0'ın**: koşul modelin ne gördüğünü değiştirir, kullanıcının ne
  aldığını değil.
- **Alt kategori** = metadata `categories[1]` (`marketing.subcategory_level`). Ölçüldü:
  Toys'ta 223 farklı değer ("Preschool", "Games & Accessories", "Puzzles"…), Grocery'de 336
  ("Pantry Staples", "Snacks & Sweets", "Beverages"…); 0. seviye kategori adının kendisi.
  Listesi kısa olan ürünün alt kategorisi yok → hiçbir kümeye girmez.
- **Hediye-yalnız alt kategori S_u** = H_u'da o alt kategoride en az bir hediye satırı olan
  ve **hediye olmayan hiçbir satırı olmayan** alt kategori. `unclear` satırı da "hediye
  değil" sayılır (muhafazakâr: kullanıcının kendi ilgisi olabilecek alt kategori dışarıda).
  Hediye = eksenin etiket kümesi (`narrow`: `gift_given`; `broad`: + `household` + `received`).
- K = `experiment.topk[0]` = 10. Kullanıcı başı pay önce seed'ler üzerinden ortalanır; GA
  kullanıcılar yeniden örneklenerek (`experiment.bootstrap_iters`, `experiment.ci`).

#### M1 — retargeting israf payı

Top-10'un S_u'dan gelen payı, **S_u boş olmayan test kullanıcılarında** (hepsi üzerinden
değil — hediye geçmişi olmayan kullanıcının israfı tanım gereği sıfır ve oranı sulandırır;
maruz kullanıcı sayısı ve toplam test kullanıcısı ayrıca yazılır). Karşıtlıklar, pozitif =
soldaki liste daha çok slot harcıyor: dar eksen **C0−C1**, **C4−C1** (plaseboya karşı),
**C0−C3**; geniş eksen **C0−C1b**, **C4b−C1b**.

#### M2 — kontaminasyon yarı ömrü (varsayıldı)

- Kapsam: en az bir `gift_given` satırı olan test kullanıcısı; **son** hediye g'nin alt
  kategorisi S_g hediye-yalnız olmalı (değilse öneriler kullanıcının kendi ilgisini de
  yansıtabilir ve ayrıştırılamaz).
- n_u = H_u'da g'den **sonraki** `self` satır sayısı (diğer etiketler sayılmaz).
- Fazla pay e_u = pay_C0(S_g) − pay_C1(S_g): kirli modelin, hediyeyi hiç görmemiş modele
  göre o alt kategoriye fazladan verdiği slotlar. Aynı kullanıcılar, eşli.
- Kova: n = 0 … 9 ayrı, ≥ 10 birleşik (`marketing.m2_max_self_after_gift`). Birleşik kova
  ve 30'dan az kullanıcılı kova (`marketing.m2_min_users_per_bucket`) uyuma girmez, raporda
  görünür.
- Uyum: e(n) = A·exp(−λn), kova büyüklüğüyle ağırlıklı en küçük kareler (A ∈ [0, 1]).
  **Yarı ömür = ln 2 / λ etkileşim.** GA: kullanıcılar yeniden örneklenip yeniden uyum;
  tekrarların yarısından azı uyum verirse GA yazılmaz.
- **Tanımsız** yazılır (uydurulmaz): uyuma giren kova 3'ten azsa, ilk kovadaki fazla ≤ 0
  ise ya da uyum yakınsamazsa. "Yarı ömür yok" bir bulgudur.
- Haftaya çeviri: test kullanıcılarının ardışık etkileşim aralığının kullanıcı başı
  medyanının medyanı (gün) × yarı ömür ÷ 7. **Yaklaşık** — review tarihi alım tarihi değil.
  (Planın "T15 medyan alımlar arası süre" dediği tablo projede yok; bu sayı aynı işi görüyor
  ve modülün içinde hesaplanıyor.)
- Birincil model SASRec (sıra bilgisini kullanan tek model). BPR için de hesaplanır ama
  BPR geçmişin sırasını görmediği için n'ye bağlı bir azalma beklenmez; öyle etiketlenir.

#### M3 — segment

C1−C4 ve C1b−C4b kullanıcı başı metrik farkı, |H_u|'nun üçte birlik dilimlerinde. Kesim
noktaları test kullanıcılarının geçmiş uzunluğunun 1/3 ve 2/3 yüzdelikleri; kesim
noktasına **eşit** uzunluk alttaki dilime gider (dilimler bu yüzden eşit büyüklükte
olmayabilir; boyutlar ve kesim noktaları yazılır). Önceden kayıtlı yorum etiketi her dilime.

#### Bulunan hata (testte, commit'ten önce)

Alt kategori haritası ilk yazımda yalnızca **geçmişte geçen** ürünlerle sınırlıydı; top-K'daki
geçmişte olmayan bir ürünün alt kategorisi boş kalıyor ve M1/M2 payı sessizce **düşük**
çıkıyordu (elle kurulmuş testte 0,50 yerine 0,15). Gerçek veride öneri evreni C0'ın ürünleri
olduğu için çoğu zaman görünmezdi. Harita artık kategorinin bütün ürünlerini kapsıyor; test
bu durumu kilitliyor.

#### Doğrulanan / doğrulanmayan

- **Doğrulandı:** 6 test (M1 elle hesaba eşit; M2 kurgulanmış fazla payda A = 0,8 ve yarı
  ömür = 1 etkileşimi, 1 gün aralıkta 1/7 haftayı buluyor; fazla yoksa tanımsız; M3 dilimleri
  ve etiketler) · 24 RecBole duman koşusunun gerçek biçimli top-K dosyalarıyla uçtan uca
  (sentetik metadata): M1 beş karşıtlık, M2 kovaları (30'u geçen kova olmadığı için beklendiği
  gibi tanımsız), M3 dilimleri.
- **Doğrulanmadı:** gerçek veriyle hiçbir şey; M2'nin gerçek kova doluluğu bilinmiyor.

**Etkilediği bölüm:** `recsys/marketing_metrics.py`, `configs/base.yaml` (`marketing`),
`tests/test_marketing_metrics.py`, CLAUDE.md §2/§7/§10/§13, README, GENEL_BAKIS §8
**Kim:** Ekip (M2 tanımı "varsayıldı" — ekip onayı bekliyor)

---

### 2026-09-14 — Sonuç figürleri F20–F23: kodlama kuralları (HİÇBİR SONUÇ GÖRÜLMEDEN)

`analysis/result_figures.py` yalnızca raporlanmış JSON'ları çiziyor; hiçbir sayı figürde
üretilmiyor. Sunum kararları sonuç görülmeden verildi ki figür sonuca göre şekillenmesin:

- **F21 yalnızca birincil metriği** (`gate2.metric`, Recall@10) gösterir; öteki metrikler
  `experiment_stats.json`'da. Satır sırası önceden kayıtlı karşıtlık sırası (C1−C4 en üstte).
- **Kapı 2 PASS olmayan nokta içi boş** çizilir ve alt başlık bunu söyler; M1/M2'de Kapı 2
  değerlendirilmemişse de içi boş.
- `reportable: false` girdiden (vekil/sentetik) çizilen figürün başlığında **"SMOKE TEST —
  NOT REPORTABLE"** yazar; stub öğrencinin F20'si de öyle.
- **Renk:** kategori `viz.ROLE_COLOR` (Toys mavi, Grocery aqua), eksen C1/C1b doğrulanmış
  sıradaki komşu slotlar (mavi/turuncu). İki kategori rengi, beceri paketinin doğrulayıcısının
  Python'a aktarılmış hesabıyla **ölçüldü**: CVD ΔE 23,1 (eşik 8), normal görüş ΔE 24,0 (eşik
  15) — geçti. Grocery'nin aqua'sı açık yüzeyde 2,74:1 (3:1 altı) → her seride **marker şekli**
  ikinci kodlama ve ilk satırda doğrudan etiket.
- Duman çıktısıyla çizildi ve **göz ile** kontrol edildi; ilk hâlinde üç yerleşim hatası vardı
  (başlık ile alt başlık çakışıyordu, legend veri noktalarının üstüne biniyordu, M2 eğrisi
  kullanıcısı olmayan kovayı köprülüyordu) — düzeltildi. F20 ayrıca yerel model klasörünün
  mutlak yolunu başlığa yazıyordu (kullanıcı adı sızıntısı); artık yalnızca son parça.

**Etkilediği bölüm:** `analysis/result_figures.py`, `tests/test_result_figures.py`, CLAUDE.md
§2/§7/§10, README, GENEL_BAKIS §8
**Kim:** Ekip

---

### 2026-09-15 — Hafta 5 sonucu: damıtma kapısı PASS, deney korpusunun tamamı etiketlendi

#### Ne koşuldu

Kaggle, "Save & Run All", kod `79e6b80` (raporların `code_version`'ı). transformers 5.0.0,
torch 2.10.0+cu128, `sdpa` dikkat. **Eğitim** tek T4, 3 epoch, fp16, gradient checkpointing,
etkin grup 32: 1.683 sn, son eğitim kaybı 0,276. Model seçimi yok (sabit epoch, ara kontrol
noktası seçilmedi). **Çıkarım** iki T4'te paralel: Toys 2.164.018 satır 5.940 sn (364 satır/sn),
Grocery 2.434.594 satır 7.385 sn (330 satır/sn). Yedek model (DeBERTa-v3) **denenmedi** —
önceden kayıtlı kural yalnızca ana model FAIL olursa.

#### Sadakat kapısı — PASS (3/3); eşikler 2026-09-14'te, eğitimden önce

| ölçüt | ölçülen [%95 GA] | eşik | |
|---|---|---|---|
| 1 · C1 ekseni, öğrenci vs öğretmen (ayrılmış 4.667) | F1 0,910 [0,897–0,923] | ≥ 0,85 | ✓ |
| 2 · C1b ekseni, öğrenci vs öğretmen | F1 0,931 [0,922–0,941] | ≥ 0,85 | ✓ |
| 3 · C1 ekseni, öğrenci vs insan (500) | F1 0,742 [0,685–0,793]; öğretmen 0,731 | düşüş ≤ 0,05 | ✓ |

Kapıya **girmeyen** ölçümler (raporda ayrı):
- **clean → 5-core kayması:** ayrılmış kümenin 5-core'daki 546 satırında C1 F1 0,913
  [0,872–0,951], C1b 0,925 [0,894–0,954]. Kayma saptanmadı; aralık geniş.
- **Öğrenci insana karşı öğretmenin gürültüsünü taşıyor:** C1 kesinliği 0,661, duyarlılığı
  0,845; C1b F1 0,891 [0,868–0,914]. Öğrenci öğretmeni kopyalıyor, düzeltmiyor — beklenen.
  2026-09-14'ün yorum kuralı (C1 ≈ C4 → "saptanamadı", C1 > C4 → alt sınır) aynen geçerli.
- Beş sınıf (ayrılmış): doğruluk 0,887, makro-F1 0,743; en zayıf `unclear` (F1 0,38),
  `received` 0,68 (n = 56).
- LLM etiketiyle örtüşen 5-core satırlarında uyum Toys 0,970 (n = 2.072), Grocery 0,973
  (n = 2.532) — **iyimser**: bu satırların çoğu eğitimdeydi.

#### 5-core etiket payları (öğrenci, ham — kalibre DEĞİL)

| | gift_given = C1 | household | received | self | unclear | C1b |
|---|---:|---:|---:|---:|---:|---:|
| Toys | %24,8 | %27,3 | %0,3 | %42,3 | %5,4 | %52,4 |
| Grocery | %3,1 | %4,0 | %0,5 | %89,0 | %3,4 | %7,6 |

Mevsimsellik (Aralık–Ocak ÷ Haziran–Eylül): C1 Toys 1,36 · Grocery 1,75 (Kapı 1'in referansı
≥ 1,25); C1b 1,17 / 1,30. RQ1'in insan kalibrasyonlu oranı (clean `main` çerçevesi) **bu
tablonun yerine geçmez** ve bu tablo onun yerine geçmez: farklı korpus, farklı etiketleyici.
Deney koşulları tanım gereği ham öğrenci etiketini kullanıyor.

#### Deney girdileri (yerelde, gerçek etiketle) — ölçülenler

- `atomic --labels distilled`: Toys 268.652 kullanıcı · 103.618 ürün · değerlendirilebilir
  (test ürünü `self`) **%43,7** (117.386); Grocery 268.991 · 94.863 · %88,8.
- Koşulların çıkardığı eğitim payı: Toys C1 **%24,7**, C1b **%52,8** (5.251 ürün, %5,1,
  eğitimde gerçek kimliğiyle kalmıyor); Grocery C1 %3,1, C1b %7,6. C4/C4b aynı sayıda satır
  çıkarıyor. **Sonuç için anlamı:** Toys'ta C1b−C0 farkı yarı yarıya veri kaybı da içerir;
  önceden kayıtlı birincil karşıtlıkların C1−C4 / C1b−C4b olması tam bunun için.
- C3 (Toys) 73.683 gölge ürün ekliyor; SASRec'in çıkış katmanı ~177 bin ürüne çıkıyor.
- Tekrar eden (kullanıcı, ürün) çifti **yok** (iki kategoride de); C0 alınmış ürün maskesi
  hiçbir koşulda valid/test pozitifiyle **çakışmıyor** — koşucu çakışmada durur, Kaggle'dan önce
  sayıldı.

#### Kaggle betikleri — sonucu etkilemeyen düzeltmeler

- `d1b809e` — `DATASET_PATH` elle yazılmıyor; dosyalar `/kaggle/input` altında aranıyor.
- `79e6b80` — damıtma hücresi kapı PASS değilse hata vermeden biter (hata veren versiyonun
  çıktısı saklanmayabilir). Yedek model yeni oturumda ana modelin FAIL raporunu **depodan**
  okur: FAIL raporu önce commit edilmeli.
- Deney betiği (bu commit):
  - Depo + RecBole paketleri `/tmp`'de: "Save & Run All" `/kaggle/working`'in tamamını çıktı
    diye kaydediyor, paket klasörü on binlerce dosya.
  - **Zaman bütçesi:** yeni koşu, aynı (kategori, model) için ölçülen en uzun süre × 1,15
    `MAX_HOURS`'a sığmıyorsa başlatılmaz. Kaggle 12 saatte keser ve kesilen versiyonun
    çıktısı kaybolabilir; eski kural yalnızca başlangıç saatine bakıyordu.
  - İlk iki koşu çökerse yeni koşu başlatılmaz (sistematik hata, kota yakılmaz).
  - Önceki oturumun çıktısı `/kaggle/input` altında kendiliğinden bulunur (yalnızca
    `peruser/<slug>/`).
  - `out/oturum_ozeti.json`: biten / bitmeyen koşular ve ölçülen süreler.
  - **`PLAN = "matrix"` varsayılan.** Matrisin öncelik sırası zaman sondasının iki koşusuyla
    (Toys × C0 × {SASRec, BPR} × ilk seed) başlıyor, sıra kesilmez çekirdekle devam ediyor;
    ayrı sonda oturumu kotaya bir şey kazandırmıyor, bir tur kaybettiriyor. Plan Faz 4.1'in
    **usulü** değişti, içeriği değil — süreler aynı koşulardan ölçülüyor.
  - Pip `--only-binary=:all:`. Pinlerin hepsinin Linux cp311/cp312 wheel'i var (`pip install
    --dry-run` ile denetlendi); **Python 3.13'te numpy 1.26.4 yok** → betik anlaşılır bir
    mesajla durur. Kaggle imajının Python sürümü bilinmiyor.
  - Hepsi betiğin kendi kodu okunarak sahte `/kaggle/input` ve sahte koşularla denendi.
    **Kaggle'da henüz koşmadı.**

#### Bulunan hata — paralel koşularda checkpoint çarpışması (Kaggle'dan ÖNCE, RecBole kaynağından)

RecBole 1.2.0 en iyi modeli `checkpoint_dir/<model>-<Ay-Gün-Yıl_SS-DD-ss>.pth` diye kaydediyor
(`trainer.py:132`, `get_local_time` saniye çözünürlüklü) ve koşucu hepsine aynı `saved/`
klasörünü veriyordu. İki T4'te **aynı saniyede** kurulan aynı model aynı dosyaya yazar: biri
ötekinin ağırlığını yükler — ürün sayısı aynıysa **sessizce yanlış sonuç** — sonra dosyayı
siler ve öteki çöker. Matrisin sırası aynı (kategori, koşul, model) üçlüsünü aynı anda koşturmuyor,
ama farklı seed'ler ve eşit ürün sayılı koşullar için olasılık sıfır değildi. Düzeltme:
`checkpoint_dir = saved/<koşu kimliği>` (`run_experiment.checkpoint_dir`); koşu bitince klasör
de silinir. Test: 12 farklı koşu 12 farklı klasör. Sentetik veriyle BPR + SASRec'te dosyanın
koşuya özel klasöre yazıldığı ve temizlendiği RecBole'un kendi `_save_checkpoint`'i izlenerek
görüldü; kullanıcı başı tutarlılık farkı 0,0.

#### Yerel ön uçuş (gerçek veri, CPU, 1 epoch — sonuç DEĞİL, raporlara yazılmadı)

- **BPR · Toys · C1b:** uçtan uca geçti. 547 sn (eğitim + valid 244 sn, test 231 sn), ~3,5 GB RAM.
  117.386 kullanıcı; C0 maskesine 853.979 çift; kullanıcı başı ortalama RecBole toplamına
  **birebir** eşit (fark 0,0); test çiftleri özeti `3bed366e…` (Kaggle'daki her koşulda aynı
  çıkmalı).
- **SASRec · Toys · C3:** uçtan uca geçti. Eğitim **30 grupla sınırlandı** (grup 512; tam epoch
  CPU'da grup başına ~3 sn, ~2 saat sürecekti — ilk deneme 1.261. grupta durduruldu), valid ve
  test **tam boyut**: 1.134 sn (valid dahil 721, test 292), veri yüklemede en fazla ~3,8 GB RAM.
  177.231 ürün (73.683 gölge ürün maskelendi), değerlendirme grubu 378 kullanıcı; kullanıcı başı
  ortalama RecBole toplamına birebir eşit; test çiftleri özeti BPR C1b ile **aynı** (`3bed366e…`)
  — Kapı 2'nin 1. ölçütü farklı koşul ve modelde gerçek veride tutuyor. Checkpoint koşuya özel
  klasöre yazıldı ve temizlendi.
- **GPU süresi hakkında bilgi vermez.** Zaman sondası matrisin ilk iki koşusu. RecBole'un
  varsayılanı erken durdurmalı (300 epoch tavanı, 10 epoch sabır) — SASRec'in Toys'ta koşu başına
  saatler sürmesi olası; matris büyük ihtimalle birden fazla oturum ister.

**Etkilediği bölüm:** `reports/results/distill_report_base.json`, `inference_*.json`,
`condition_{Toys_and_Games,Grocery_and_Gourmet_Food}_*.json`, F20, `scripts/kaggle_experiment.py`,
`recsys/run_experiment.py` (`checkpoint_dir`), `tests/test_run_experiment.py`, GENEL_BAKIS
§5/§6/§8, CLAUDE.md §7/§10, README
**Kim:** Ekip

### 2026-09-16 — Kaggle deney matrisi 1. deneme: RecBole'un import'u protobuf yüzünden çöktü (SONUÇ YOK)

Kaggle'daki ilk deney oturumu 96 saniyede bitti: **hiçbir koşu tamamlanmadı, hiçbir sonuç
üretilmedi.** Aşağıdaki hiçbir sayı sonuç değildir.

**Belirti.** Kurulum doğrulaması geçti (`numpy 1.26.4 | torch 2.10.0+cu128 | recbole 1.2.0 |
cuda True`, Python 3.12, görünen GPU 2), altı koşul iki kategori için yazıldı (Toys C1b
eğitimin **%52,82**'sini çıkardı — yereldeki sayının aynısı), sonra başlatılan üç koşunun
üçü de aynı yerde çöktü:
`google.protobuf.runtime_version.VersionError: … gencode 6.31.1 runtime 5.29.5`, zincir
`recbole.config` → `recbole.utils` → `torch.utils.tensorboard` → `tensorboard.compat.proto`.
Fail-fast devreye girdi: kalan 69 koşu başlatılmadı, GPU kotasından ~2 dakika gitti, `out/`
(loglar + `oturum_ozeti.json` + `condition_*.json`) yine de yazıldı.

**Kök neden.** Pin listemizdeki tensorboard 2.21.0 protobuf **6.31.1 gencode**'uyla derlenmiş;
Kaggle imajının protobuf'u **5.29.5**. Kendi protobuf 7.36.0'ımız `--target` klasöründe ve
PYTHONPATH'te önde olmasına rağmen **görünmüyor**: imajın `google` paketi düzenli paket
(`__init__.py` var), bizim `google/protobuf` portumuz namespace parçası olarak hiç devreye
girmiyor. Pin'i yükseltip düşürmek çözmezdi — imajın protobuf'u her hâlükârda kazanıyor.

**Kurulum doğrulaması neden yakalamadı.** `_dogrula()` `import recbole` yapıyor; RecBole'un
`__init__.py`'si yalnızca sürüm satırı, tensorboard zincirine hiç girmiyor. Ortam "kurulu"
göründü, koşular çöktü. Bu zayıflık **duruyor**: betik kullanıcının yapıştırdığı hücrenin
kendisi (değiştirmek yeniden yapıştırma demek) ve koşuların fail-fast'i ucuz — iki traceback,
~2 dakika.

**Düzeltme (depo kodunda; yapıştırılan hücre değişmiyor, klon her koşuda güncelleniyor).**
`run_experiment.stub_tensorboard_if_broken()`: RecBole `torch.utils.tensorboard`'ı koşulsuz
import ediyor, biz TensorBoard kaydını hiç okumuyoruz — import kırıksa yerine hiçbir şey
yapmayan bir `SummaryWriter` konuyor. Ölçülen hiçbir sayı değişmez; yalnızca okumadığımız
event dosyası yazılmaz. Çalışan bir tensorboard varsa **dokunulmaz**; torch'un kendisi yoksa
yama **yapılmaz** (gerçek hata gizlenmesin). Durum sessiz kalmıyor: koşu raporunda
`meta.tensorboard` = `ok` ya da sebep metni (sürüm numaraları taşır, mutlak yol taşımaz).

**Doğrulama (bu makinede gerçekten koşturuldu).**
- Kaggle hatası yerelde birebir üretildi: `tensorboard/compat/proto/event_pb2` importunda aynı
  `VersionError`'ı veren sahte bir tensorboard paketi PYTHONPATH'e konunca
  `from recbole.config import Config` aynı mesajla çöküyor.
- Aynı kırık tensorboard altında sentetik veriyle iki **gerçek** RecBole koşusu (BPR/C0 ve
  SASRec/C3, 1 epoch) uçtan uca geçti: kullanıcı başı ortalama RecBole toplamına birebir eşit
  (fark 0,0), gölge ürün maskesi çalıştı (119 gölge ürün, top-K'ya sızma yok), test çiftleri
  özeti iki koşuda aynı, raporlarda `meta.tensorboard` sebep metnini taşıyor.
- RecBole'un `get_tensorboard`'ı boş yazıcıyı aldı; `add_scalar`/`close` sessizce geçti, yeni
  event dosyası yazılmadı.
- 388 test geçiyor (yeni üçü: kırık import → boş yazıcı ve aynı süreçteki ikinci koşu "ok"
  demez · çalışan tensorboard'a dokunulmaz · torch yoksa yama yok).

**Hâlâ doğrulanmadı:** Kaggle imajında import zincirinin tensorboard'dan sonraki kısmı
(`recbole.model`, `recbole.trainer`), GPU'da koşu süresi ve belleği, iki kartta paralellik ve
deney sonuçlarının kendisi.

**Etkilediği bölüm:** `recsys/run_experiment.py` (`stub_tensorboard_if_broken`,
`meta.tensorboard`), `tests/test_run_experiment.py`, GENEL_BAKIS §8
**Kim:** Ekip

---

### 2026-09-18 — Deney matrisi Kaggle'da koştu: 72 koşunun 67'si bitti ve denetlendi, 5 koşu eksik

> Bu kayıt **kurulumun ve çıktıların** denetimidir. Kapı 2'nin kararı ve karşıtlıklar 72 koşu
> tamamlanınca hesaplanıp ayrı bir kayıtta yazılacak; aşağıdaki ara hesaplar commit edilmedi.

#### Ne koşuldu

Kod `e1c7f89` (67 raporun hepsinde aynı `code_version`), Kaggle, 2× T4, `PLAN = "matrix"`.
Matris birden fazla hesapta paralel yedi oturuma bölündü; her oturum hücrenin başındaki
`CATEGORIES` / `MODELS` / `SEEDS` satırlarıyla ayrık bir dilim aldı, kod değişmedi.

| oturum | dilim | süre | biten |
|---|---|---:|---:|
| D0 | tam matris, öncelik sırasıyla (fiilen seed 42 + BPR seed 1337'nin dört hücresi) | 10,9 sa | 23 |
| D1 | SASRec · Toys · seed 1337 | 5,4 sa | 6 |
| D2 | SASRec · Grocery · 1337 | 7,5 sa | 6 |
| D3 | SASRec · Toys · 2024 | 5,0 sa | 6 |
| D4 | SASRec · Grocery · 2024 | 8,2 sa | 6 |
| D5 | BPR · iki kategori · 1337 | 2,5 sa | 12 |
| D6 | BPR · iki kategori · 2024 | 2,2 sa | 12 |

Toplam 41,6 oturum saati. Koşu başına duvar süresi: SASRec Toys 24–162 dk (medyan 91),
Grocery 117–195 dk (medyan 150); BPR Toys 19–49 dk, Grocery 13–23 dk. Hiçbir oturum 12 saat
sınırına değmedi.

#### Eksik 5 koşu — hata değil, zaman bütçesi

Hepsi seed 42: **Toys C4b SASRec · Toys C3 SASRec · Grocery C4b SASRec · Grocery C3 SASRec ·
Grocery C3 BPR.** D0 bunları başlatmadı: kalan süreye sığmıyorlardı (`MAX_HOURS` kuralı,
oturum özetinde "başlatılmadı" diye yazıyor). Bölme planı bu tamamlama turunu öngörüyordu.

Etkisi: SASRec'in C4b ve C3 hücreleri ile Grocery BPR C3 şu an **iki** seed'li. C4b ve
Toys × SASRec × C3 kesilmez çekirdekte → **tamamlanmadan Kapı 2 ve karşıtlıklar kesin değil.**

Tamamlama: tek oturum, `CONDITIONS = ["C4b", "C3"]` · `SEEDS = [42]` → 8 koşu, tahmini ~6
saat. Üçü (BPR Toys C4b, Grocery C4b, Toys C3) D0'da zaten var ve yeniden koşacak — ~1 saat
fazla kota karşılığında tek hücre, tek düzenleme; tekrar determinizmi bir kez daha sınar.
Depo o sırada bu commit'te olacak: `code_version` farklı çıkar ama `src/`, `scripts/` ve
`configs/` `e1c7f89`'dan beri **değişmedi** (bu commit yalnızca rapor ve belge).

#### Denetim — indirilen yedi `out/` klasörünün tamamı, bu makinede koşturuldu

- Dosya adı ↔ `meta` birebir; 67 raporun hepsi `label_source: distilled`,
  `reportable: true`, `device: cuda`, `epochs: 300`. Matris dışı koşu yok.
- **Test çiftlerinin özeti kategori içinde tek** (Kapı 2, ölçüt 1'in önkoşulu): iki model,
  altı koşul, üç seed. Test kullanıcısı Toys 117.386, Grocery 238.756 — yereldeki sayıların
  aynısı. Kullanıcı başı dosyaların kullanıcı kümesi de kategori içinde tek.
- Kullanıcı başı dosyalar koşucudan bağımsız yeniden hesaplandı: satır sayısı = test
  kullanıcısı; altı metriğin ortalaması raporla ≤ 5·10⁻⁷ içinde; her top-K 20 **farklı**
  ürün; top-K'da gölge (`::gift`) ürün **0**.
- Maske meta'sı tasarlandığı gibi: gölge maskesi yalnızca C3'te; BPR'de C0 geçmiş maskesi
  (`c0_train`), SASRec'te `none_sequential`.
- Koşul raporları (12 dosya × 5 oturum): birbirine ve depodaki raporlara **içerikçe eşit**
  (yalnızca satır sonu farkı); C4 = C1 ve C4b = C1b kadar satır çıkarıyor; evren ⊆ C0.
- **Oturumlar arası determinizm Linux'ta ölçüldü.** D0 ve D5, BPR × seed 1337'nin dört
  hücresini (Toys C0, Grocery C0/C1/C4) ayrı oturumlarda koştu: kullanıcı başı dosyalar ve
  top-K listeleri **birebir aynı**. 2026-09-16'da yalnızca Windows'ta sınanıp Linux için
  çıkarım olarak kalan "koşul dosyaları süreçler arası aynı" varsayımı böylece doğrudan
  doğrulandı — matrisi hesaplara bölmek güvenliydi. İkiz koşuların süresi %7'ye kadar
  oynuyor (donanım).
- 81 log tarandı: beklenen tensorboard uyarısı (2026-09-16) ve RecBole/pandas
  `FutureWarning`'leri dışında hata, NaN, bellek hatası yok.

#### Hata değil — ama sonuç okunurken bilinmeli

- **SASRec'in validasyon kümesi koşula göre küçülüyor.** Eğitim geçmişi boşalan
  kullanıcının valid satırı erken durdurma setinden düşüyor (2026-09-14 tasarımı; SASRec
  boş geçmişi işleyemez). Toys: C1b 45.981 kullanıcı (%17), C4b 19.233, C1 15.666, C4 1.532;
  Grocery en fazla 802. Test kümesi değişmiyor; erken durdurma farklı kümelerde karar
  veriyor. BPR'de valid bütün koşullarda aynı.
- **Kaç epoch eğitildiği bilinmiyor.** RecBole epoch satırlarını kök logger'a yazıyor ve
  alt süreçte INFO düzeyinde handler yok — loglara düşmedi, rapora da yazılmıyordu. 300
  tavanına değen koşu olup olmadığı söylenemez. SASRec'te süre koşullar arasında satır
  oranından çok daha fazla oynuyor (Toys C1b 24 dk, C0 ~2,5 sa; eğitim satırı oranı 2,5×)
  → erken durdurma en azından bazı koşularda devrede. Protokol bütün koşullarda aynı olduğu
  için karşıtlıkları bozmaz; mutlak metriklerin tam yakınsadığı **iddia edilemez**. Matris
  bitene kadar koşucu değiştirilmiyor (5 koşu aynı kodla koşmalı).

#### Belgelerdeki eski Kapı 2 ifadesi — düzeltildi

ROADMAP, PROJECT_SPEC ve GENEL_BAKIS §6 Kapı 2'yi "C0 ile C4 arasında anlamlı fark
**olmamalı**" diye anlatıyordu. Bağlayıcı olan 2026-09-14 önceden kaydı (commit `8c697a8`,
ilk koşudan iki gün önce) ölçütü "plasebo C0'ı **geçmemeli**" diye yazdı: rastgele silme
veriyi azaltır ve doğruluğu düşürmesi beklenir; **iyileştirmesi** kurulum hatasıdır. Eski
ifade bu yüzden 2026-09-14'ten beri geçersizdi ama belgeden silinmemişti.

Açıkça yazılsın: **ara hesapta (67 koşu) Toys'ta C4, C0'dan anlamlı düşük** — eski ifadeyle
Kapı 2 FAIL olurdu. Önceden kayıtlı ölçütle geçiyor. Bu, sonuç görüldükten sonra ölçüt
değiştirmek değil (ölçüt koşulardan önce commit'liydi), ama sonuç raporu bu farkı ayrıca
yazacak. GENEL_BAKIS §6 önceden kayda göre düzeltildi; ROADMAP ve PROJECT_SPEC tarihî
belge olarak bırakıldı, bu kayıt çelişkiyi işaretliyor.

#### Analiz hattı gerçek veride denendi (ara — commit EDİLMEDİ)

67 koşuyla `experiment_stats` (9 dk), `marketing_metrics` (3 dk) ve `result_figures`
hatasız koştu; F21/F23 gerçek sayılarla çizilip göz ile kontrol edildi (yerleşim sorunu yok).
M2'nin kovaları iki kategoride de dolu (her kova ≥ 30 kullanıcı; Toys 29.577, Grocery
11.614 kullanıcı). Eksik seed'li hücreler yüzünden `gate2.json`, `experiment_stats.json`,
`marketing_metrics.json` ve F21–F23 **commit edilmedi**; 72 koşu tamamlanınca yeniden
üretilecek. **M2'nin değerlerine bakılmadı** — tanım hâlâ "varsayıldı", ekip onayı bekliyor.

**Commit edilen:** 67 koşu raporu (`reports/results/experiment_*.json`). Kullanıcı başı
dosyalar `data/processed/recbole/<slug>/peruser/` altında (git'e girmez). Tekrarlanan dört
hücrenin D5 kopyası kullanıldı (D0'ınkiyle birebir aynı).

**Etkilediği bölüm:** `reports/results/experiment_*.json` (67), GENEL_BAKIS §6/§8
**Kim:** Ekip

---

### 2026-09-19 — M2 tanımı onaylandı (DEĞERLERİ GÖRÜLMEDEN)

Kullanıcı (ekip adına), 2026-09-14 "Faz 5 kodu" kaydındaki M2 tanımını **olduğu gibi**
onayladı. Tanımda hiçbir değişiklik yok: son hediyenin hediye-yalnız alt kategorisi,
sonraki `self` sayısı n'ye göre C0−C1 fazla payı, `A·exp(−λn)` uyumu, yarı ömür = ln 2/λ
etkileşim, haftaya çeviri yaklaşık; uyum yoksa TANIMSIZ.

Onay anında M2'nin **fazla pay, uyum ve yarı ömür değerleri görülmemişti**: 67 koşuluk ara
hesapta `marketing_metrics` koştu; M2'den yalnızca kova doluluğu kontrol edildi, F22
açılmadı (2026-09-18 kaydı). Etiket
"varsayıldı" → **"onaylandı 2026-09-19"**. CLAUDE.md §13 buna göre güncellendi.

**Etkilediği bölüm:** CLAUDE.md §13, GENEL_BAKIS §8
**Kim:** Kullanıcı (ekip adına)

---

### 2026-09-19 — Deney matrisi tamam (72/72): Kapı 2 PASS 4/4, sonuçlar ve sonuçtan sonra bulunan bir tasarım etkisi

> Sonuç raporu: `docs/SONUCLAR.md`. Bu kayıt tamamlama oturumunun denetimini, kararları ve
> sonuç görüldükten sonra yapılan her şeyi yazar.

#### Tamamlama oturumu (D7)

`CONDITIONS = ["C4b", "C3"]` · `SEEDS = [42]`, kod `f3183df` (`src/`, `scripts/`, `configs/`
`e1c7f89` ile aynı), 5,7 saat, 8 koşu, hepsi bitti. Eksik beş koşu geldi; üç BPR koşusu
(Toys C4b, Grocery C4b, Toys C3) D0'da zaten vardı ve **bit bit aynı** çıktı. Böylece oturumlar
arası birebir aynılık 7 hücrede ve **iki farklı kod sürümünde** ölçülmüş oldu. Aynı denetim
betiği sekiz klasörün tamamına yeniden koşturuldu: 72/72 hücre, test çiftleri özeti kategori
içinde tek, kullanıcı başı ortalamalar raporla ≤ 5·10⁻⁷, top-K'da gölge ürün 0, koşul
raporları altı klasörde birbirine ve depodakine eşit, 91 log temiz. Betiğin "valid koşullar
arasında farklı" uyarısı yalnızca SASRec'in düşen valid satırları: her koşuda
`valid + n_valid_dropped_empty_history` kategorinin valid sayısına eşit (2026-09-18 tasarımı).
Depoya yalnızca eksik beş koşu kopyalandı; tekrarların D0 kopyası kaldı.

#### Sonuçlar (ayrıntı ve GA'lar SONUCLAR'da)

- **Kapı 2: dört hücrede PASS** (`gate2.json`). Plasebo her yerde C0'ın altında ya da ondan
  ayırt edilemiyor; seed değişim katsayısı 0,002–0,025.
- **RQ2 birincil C1 − C4 (Recall@10):** Toys SASRec +0,545 puan [+0,482, +0,610], BPR +0,482
  [+0,436, +0,529]; Grocery SASRec +0,036 [+0,005, +0,068] → hepsi "alt sınır"; Grocery BPR
  +0,020 [−0,009, +0,049] → "saptanamadı" (NDCG@10'da alt sınır). Doz–yanıt dört
  karşılaştırmada da Toys > Grocery.
- **C1 − C0 modele bağlı:** Toys BPR +%4,4 (silmek iyileştiriyor), Toys SASRec −%7,1 (silmek
  kötüleştiriyor); Grocery'de sıfırı içeriyor.
- **RQ3:** gölge token (C3) hiçbir hücrede C1'den iyi değil; 3/4 hücrede C1'den ve C0'dan
  anlamlı kötü.
- **RQ4 / M2 (birincil SASRec):** Toys yarı ömür 0,66 [0,53–0,82] kendi alımı (≈2,4 hafta),
  Grocery 9,5 [5,6–23,2] (≈86 hafta). BPR'de GA'nın üst ucu uyumun λ sınırına dayanıyor →
  sınırlı yarı ömür yok. M1 Toys BPR'de C0 − C1 +4,9 puan, plaseboya karşı +4,2.

#### Sonuçtan sonra bulunan: valid satırındaki hediye hiçbir koşulda silinmiyor

M2'nin n = 0 kovasında C1'in de hediyenin alt kategorisine %29 slot verdiği görülünce
incelendi. Sebep koddan doğrulandı (`run_experiment.sequential_parts`): koşullar yalnızca
eğitim satırlarını değiştiriyor; SASRec'in test girdisi koşulun eğitim satırları + **valid**
satırı. Valid satırı hediye ise C1/C1b/C3'te de test girdisinin son ürünü olarak kalıyor.
BPR valid satırıyla hiçbir koşulda eğitilmiyor.

Ölçüldü: test kullanıcılarının valid satırı `gift_given` olanı Toys'ta **18.335 (%15,6)**,
C1b kümesiyle 43.407 (%37,0); Grocery'de 6.118 (%2,6) / 15.476 (%6,5). M2'nin n = 0 kovasında
son hediye valid satırı olan kullanıcı Toys'ta 11.486 / 15.434 (%74), Grocery'de 2.422 / 2.688
(%90); n ≥ 1 kovalarında tanım gereği sıfır.

Değerlendirme: bir **kod hatası değil**, bölmenin C0'da donmasının (kural 3) sonucu. Etkisi
önceden yazılmamıştı. SASRec için C1'i kısmi bir temizliğe çeviriyor ve C1 − C4 ile C1 − C0
farklarını **küçültür**; birincil yorum ("alt sınır") yönünde. **Hiçbir tanım, koşul ya da
sayı değiştirilmedi** — önceden kayıtlı tanımları sonuç gördükten sonra değiştirmek yasak.
Yapılan iki şey:
1. SONUCLAR'da sınırlılık olarak yazıldı (madde 3).
2. M2 için **post-hoc duyarlılık**: aynı uyum n = 0 kovası hariç, JSON'daki kova
   ortalamalarına (`fit_half_life`, GA'sız): Toys SASRec 1,08 (birincil 0,66), Grocery SASRec
   4,2 (9,5), Toys BPR 4,4, Grocery BPR 5,6 kendi alımı. SONUCLAR §6.1'de "önceden kayıtlı
   değil" diye işaretli; birincil değerin yerine geçmiyor.

#### Sonuçtan sonra yapılan sunum değişikliği: F22 lejantı

F22 lejantı yarı ömrü GA'sız yazıyordu; BPR için "≈1001,5 hafta" gibi, GA'sı sınırsız bir
nokta tahmini süre ölçümü gibi okunuyordu. Lejant artık GA'yı da yazıyor. GA'nın üst ucu
uyumun λ alt sınırına (`marketing_metrics.LAMBDA_MIN`, eskiden koda gömülü `1e-6`, değeri
aynı) dayanıyorsa "∞" yazıyor ve haftaya çevirmiyor. Dört seriye de aynı kural uygulanıyor.
Hiçbir sayı değişmedi; F20, F21, F23 bit bit aynı çıktı. `marketing_metrics` değişiklikten
sonra yeniden koşturuldu, JSON `code_version` dışında birebir aynı. Test:
`test_m2_legend_shows_the_ci_and_does_not_read_an_unbounded_one_as_a_duration`.

#### M2 onayı

M2 tanımı bu kaydın hemen öncesindeki kayıtla değerler görülmeden onaylandı.

**Commit edilen:** 5 koşu raporu, `gate2.json`, `experiment_stats.json`,
`marketing_metrics.json`, F21–F23, `docs/SONUCLAR.md`.
**Etkilediği bölüm:** `analysis/result_figures.py`, `recsys/marketing_metrics.py` (sabit),
`tests/test_result_figures.py`, SONUCLAR, GENEL_BAKIS §4/§5/§6/§8, CLAUDE.md §7/§13, README
**Kim:** Ekip

---

### 2026-09-20 — Denetim: önceden kayıtlı iki detektör kapısı tutturulamadı (KAYIT)
**Karar:** İki eşik `concept-note` §"Success criteria"da yazılıydı ve **tutturulamadı**.
Bugüne kadar hiçbir belgede uzlaştırılmamıştı; bu kayıt onu kapatıyor. Sonuç **değişmiyor**,
proje bu ölçümle ilerlemiş sayılıyor — ama artık sınırlılık olarak yazılı.

| Kapı (concept-note §5) | Eşik | Ölçülen (`validation_500.json`) |
|---|---|---|
| Detektör makro-F1 (insan etiketine karşı) | **≥ 0,75** | **0,5404** [0,4897–0,5869] · popülasyon ağırlıklı 0,494 |
| `gift_given` kesinliği | **≥ 0,80** | **0,68** · popülasyon ağırlıklı **0,6485** |

**Gerekçe (neden eşiklerin öngördüğü şey yapılmadı):** İkisinin de yazılı karşılığı "prompt'u
revize et (v2) / birincil modeli değiştir" ve "confidence eşiğini yükselt"ti.

1. **Prompt v4 yazılmadı.** Gerekçesi 2026-08-29'da (sonuç görülmeden, `confidence` kaydında)
   yazılmıştı: v4 ~4 saat Kaggle ve **Kapı 1'in yeniden koşulması** demek. Daha önemlisi,
   500 doğrulama satırı prompt ayarı için kullanılamaz — kullanılsaydı elimizdeki tek
   bağımsız referans yok olurdu (deneme setinin F1 olarak raporlanamamasıyla aynı hata).
2. **`confidence` eşiği zaten düşmüştü.** 2026-08-29: `low` ile `unclear` birebir örtüşüyor
   (870/870), alan bağımsız bilgi taşımıyor. Yani kapının öngördüğü kaçış yolu ölçümle
   kapanmıştı; iki karar birbirine bağlanmamıştı.
3. **Eşikler koda hiç girmedi.** `analysis/validation.py` tek kapı ölçütü tanıyor
   (`1_annotator_agreement`, tek etiketleyici olduğu için `passed: null`). Yani bu iki eşik
   fiilen hiçbir zaman kapı olarak kurulmadı — bu kayıt onu düzeltmiyor, **yazıyor**.

**Sonuca etkisi:** Zaten raporlanan etkinin ta kendisi. C1 etiketinin popülasyon kesinliği
0,65 → "hediye" denen satırların ~üçte biri insana göre hediye değil. Önceden kayıtlı yorum
kuralı bu yüzden var: **C1 > C4 gerçek etkinin ALT SINIRI**. Düşük makro-F1'i iki
küçük/bulanık sınıf çekiyor (`received` F1 0,12, `unclear` 0,20); deneyin kullandığı ikili
eksenlerde F1 0,72 (C1) ve 0,79 (C1b).

**Etkilediği bölüm:** `docs/SONUCLAR.md` §7 (madde 1), `concept-note` §5 errata,
`technology-review` §5 errata
**Kim:** Ekip (denetim bulgusu)

### 2026-09-20 — Denetim: birincil metrik NDCG@10 değil Recall@10 (KAYIT)
**Karar:** `concept-note` §"Decision rule" karar kuralını **NDCG@10** üzerinden yazmış;
`configs/base.yaml` → `gate2.metric` ve `SONUCLAR.md` **Recall@10** diyor. Değişimin kaydı
yoktu. **Birincil metrik Recall@10 kalıyor** (deney koşulmadan, 2026-09-14'te config'e öyle
yazıldı ve öyle koşuldu); değişim burada kayda geçiyor.

**Fark nerede sonucu çeviriyor:** Tek hücrede. Grocery × BPR, C1 − C4:

| Metrik | Fark | %95 GA | Önceden kayıtlı yoruma göre |
|---|---|---|---|
| Recall@10 (fiilî birincil) | +0,000201 | [−0,000085, +0,000487] | **saptanamadı** |
| NDCG@10 (concept-note'un kuralı) | +0,000198 | [+0,000055, +0,000352] | **alt sınır** |

Öteki üç hücrede iki metrik aynı yönü veriyor. `experiment_stats.json` altı metriği de
taşıyor ve `SONUCLAR.md` §4.2 ikisini yan yana yazıyor — yani seçim sonucu gizlemiyor,
yalnızca hangisinin "birincil" olduğu kayıtsız değişmişti.

**Etkilediği bölüm:** `docs/SONUCLAR.md` §7, `concept-note` errata
**Kim:** Ekip (denetim bulgusu)

### 2026-09-20 — Denetim: kod ve kayıt düzeltmeleri (sonuç sayısı DEĞİŞMEDİ)
**Karar:** Tam bir uçtan uca denetim yapıldı. Bulgular düzeltildi; **hiçbir sonuç sayısı
değişmedi** — `gate1_*.json`, `prevalence.json`, `validation_500.json` ve
`keyword_precision.json` yeniden üretildi, ölçülen değerlerin hepsi bit düzeyinde aynı çıktı
(tek fark eklenen provenans alanları). Birincil karşıtlık (C1 − C4, Toys × SASRec,
Recall@10 = +0,005449) ham `peruser` parquet'lerinden bağımsız olarak yeniden hesaplandı ve
`experiment_stats.json` ile birebir tuttu.

**Düzeltilen kırıklar:**
- `analysis/precision_check.py`: `score()` **hiç çalışmıyordu** — `relative_to_repo` çağrılıyor
  ama import edilmemiş (`6f5e9b9` ortak yardımcıya geçerken bu dosyayı atlamış, 2026-08-27'den
  beri). T4 yeniden üretilemiyordu. Aynı modülde: etiket sözlüğü hâlâ şema v2'ydi (`received`
  yoktu) ve `sample` elle etiketlenmiş CSV'yi koşulsuz eziyordu (artık `should_skip` +
  `--force`). Yeniden koşuldu: **proxy precision 0,5833 değişmedi**.
- `detection/inference.py`: parça önbelleği modele/backend'e göre anahtarlanmıyordu. Stub ya
  da başka bir modelle üretilmiş parçalar, nihai damga o an yüklü öğrenciden geldiği için
  `label_source: distilled` diye yazılabilirdi. Artık parça klasöründe `_manifest.json` var ve
  uyuşmazlıkta koşu açık hatayla duruyor (desen `llm_annotate._stale_backend`).
- `detection/llm_annotate.py`: kesintiden sonra `stats_w{worker}.json` **eziliyordu** ve kalite
  oranları birleşmiş tam satır sayısına bölünüyordu — Kapı 1'in 3. ölçütü ölçülmemiş bir sayıyı
  okuyabilirdi. Her koşu artık kendi sayaç dosyasını yazıyor, oranlar üretilen satır sayısına
  bölünüyor, rapor `n_rows_generated` ve `counters_cover_all_rows` taşıyor; işçi sayısı dosya
  sayısından değil işçi kimliğinden geliyor. **Dört üretim koşusu tek oturumda bitti**, yani
  mevcut Kapı 1 sayıları bu hatadan etkilenmedi.
- `analysis/gate1.py`: vekil bayrağı join'i denetimsizdi; örnekleme dosyası etiketlemeden sonra
  yeniden üretilseydi 2. ölçüt ölçmediği bir şeyi PASS ederdi. Artık satır/null denetimi var
  (dört kategoride de yeniden koşuldu, PASS 4/4 değişmedi).
- `recsys/atomic.py`: `split.json` yoksa bayat-etiket koruması komple atlanıyordu — o dosyayı
  Kaggle akışı operatöre elle yeniden adlandırtıyor, yani kaybolması gerçek senaryo.

**Provenans:**
- `utils/io.code_version()` artık commit'lenmemiş değişikliği `-dirty` ile işaretliyor.
  `marketing_metrics.json` `f3183df` damgalıydı ama içeriği `589fad8`'deki `LAMBDA_MIN`
  değişikliğiyle üretilmişti (GA üst ucu tam `ln2/1e-6`); damga temiz bir commit'i işaret
  ediyordu. Geçmişe dönük düzelmiyor, bundan sonrası dürüst.
- `gate1_*`, `prevalence`, `validation_500`, `keyword_precision` artık `code_version` taşıyor.
  `validation_500.json` ayrıca `model` ve `prompt_version` yazıyor (ikisi de `null`du: rapor
  hangi modeli doğruladığını söylemiyordu).
- Video_Games duman testi artefaktları `reports/results/smoke/` altına taşındı; adlarından
  gerçek koşulardan ayırt edilemiyorlardı.

**Testler (389 → 398):** Üç değişmez **mutasyonla düşecek** biçimde kilitlendi ve bu
doğrulandı: (1) `test_c4_removes_at_random_not_by_label` C4'ü C1'in kopyası yapan mutasyonda
geçiyordu (`household` C1'de silinmediği için `or` dalı hep doğruydu) — plasebonun tek
tanımlayıcı özelliği test edilmiyordu; (2) eşli bootstrap testleri eşsiz bir uygulamayla da
geçiyordu (GA 0,059 genişliğinde ama hâlâ sıfırı dışlıyor) — artık kullanıcı başı fark sabit
kurulup GA genişliğinin sıfır olması isteniyor; (3) "süreçler arası üretilebilir" testi tek
süreçte koşuyordu ve `hash()` süreç içinde kararlı olduğu için `_seed_for` `hash()`e
çevrilse bile geçiyordu — değer artık çakılı (`993655479`). Yeni: `tests/test_io.py`,
çıkarım manifesti, kesintili koşunun sayaçları.

**Belgeler gerçeğe çekildi:** W&B "verilmiş karar" diye yazılıydı ama kodda tek satır yok
(ölü bağımlılık ve ölü `.env` değişkenleri kaldırıldı) · "her komut idempotent" iddiası beş
modülde doğru değildi · hızlı başlangıçta zorunlu `download --meta` adımı eksikti ·
GENEL_BAKIS diyagramı 12,4M satır etiketlendiğini söylüyordu (gerçek 4.598.612) · depo
kökünde README yoktu.

**Etkilediği bölüm:** `analysis/precision_check.py`, `analysis/gate1.py`,
`analysis/prevalence.py`, `analysis/validation.py`, `detection/inference.py`,
`detection/llm_annotate.py`, `recsys/atomic.py`, `utils/io.py`, altı test dosyası, CLAUDE.md,
README, GENEL_BAKIS, SONUCLAR, kök README
**Kim:** Ekip (denetim)
---

### 2026-09-20 — Bakım borcu kapatıldı: iki gerçek kırık, dört tekilleştirme (sonuç sayısı DEĞİŞMEDİ)
**Karar:** Denetimin "bitmişlik için gerekli değil" diye ayırdığı listenin tamamı yapıldı.
`prevalence`, `keyword_precision`, `gate2`, `experiment_stats` ve `marketing_metrics`
yeniden üretildi; diff **yalnızca** `code_version` satırları.

**Kırık/tehlikeli olan:**
- `llm_annotate --gpus auto` her çağrıda `ValueError` veriyordu, üstelik yardım metni onu
  geçerli gösteriyor ve **Kaggle'da kullanılan değer tam buydu**. Elle verilen sayı da
  `_resolve_gpus`'u atlıyordu: iki kartlı makinede `--gpus 4` sessizce dört süreç açardı.
  İkisi de artık aynı kapıdan geçiyor; düzeltmeyi geri alınca testin düştüğü görülerek
  (mutasyon) doğrulandı.
- `eda`/`deep_eda` üzerine test yazarken **üç gerçek arıza** çıktı: `available_roles`
  config'te tanımsız bir rol görünce bütün EDA'yı düşürüyordu, `table_cross_category` tek
  kategorili koşuda sıfıra bölüyordu, `table_seasonality_summary` eksik bir ayda `KeyError`
  atıyordu. Üçü de artık "ölçülmedi" yazıyor ya da atlıyor.
- `marketing_metrics`'te `explode()` polars 2.0 uyarısı alıyordu: varsayılan değişince
  top-k'sı boş kullanıcı sonuçtan **düşerdi**, yani ortalamanın paydası sessizce değişirdi.
  `empty_as_null=True` artık açık yazılı. `pyproject.toml`'daki toptan
  `ignore::DeprecationWarning` kaldırıldı — uyarıyı susturan oydu.

**Tekilleştirme (davranış değişmeden):** iki `wilson()` kopyası `utils/stats.wilson_interval`
oldu; **iki çağrı noktasının `z` varsayılanı aynı değil** (1,959963985 ve 1,96) ve yayımlanmış
aralıklar o değerlerle üretildi, o yüzden sessizce eşitlemek yerine test kilitledi.
`_seed_for` → `utils/seeding.seed_for`; modül dışından çağrılan `_common_seeds`, `_aligned`,
`_ci` açık adlar aldı. `seed_for(42, "C4") == 993655479` çakılı.

**Kayıt:** Koşu raporu artık `epochs_trained` ve `hit_epoch_cap` yazıyor. Geçmiş 72 koşuyu
kurtarmıyor — SONUCLAR §7 madde 6 duruyor, yanına "bundan sonrası kaydediliyor" eklendi.

**Yeni:** `scripts/demo.py` (veri dosyası ve bağımlılık gerektirmeden commit'li JSON'lardan
dört RQ'yu, iki kapıyı ve provenansı basar), kök `LICENSE` (MIT + veri/model/teslim
carve-out'ları), `data-research/figures/README.md` (o klasör elle tutulmuyor,
`eda.publish_figures_to` her koşuda üzerine yazıyor).

**Testler:** 399 → 441 (`test_eda` 18, `test_demo` 9, `test_stats` 10, `--gpus` 2, epoch 3).

**Etkilediği bölüm:** `detection/llm_annotate.py`, `detection/prompting.py`,
`analysis/eda.py`, `analysis/deep_eda.py`, `analysis/prevalence.py`,
`analysis/precision_check.py`, `analysis/experiment_stats.py`, `recsys/conditions.py`,
`recsys/marketing_metrics.py`, `recsys/run_experiment.py`, `utils/stats.py` (yeni),
`utils/seeding.py` (yeni), `config.py`, `scripts/kaggle_annotate.py`, `pyproject.toml`
**Kim:** Ekip

---

### 2026-09-21 — Capstone raporu ve sunum: kapsam, dil ve "sayılar betikten okunur" kuralı
**Karar:** Yazılmamış son iki teslim üretildi.

- **Rapor** yalnızca **Markdown** ve **İngilizce**: `final-report/final-report.md`.
  `.docx` üretilmedi — diğer iki `.docx` teslimi zaten var, üçüncüsü aynı sayıları üçüncü
  bir yerde daha kopyalamak olurdu. Başlık bloğu Model Refinement / Deployment
  teslimleriyle birebir aynı.
- **Sunum** `.pptx` + PDF + tek sayfalık PDF özet: `presentation/`. Slaytlar **İngilizce**,
  **konuşmacı notları Türkçe** (sunumu Türkçe yapılacak, dinleyiciye İngilizce belge
  kalacak).
- **Her ikisinde de hiçbir sayı elle yazılmadı.** `build_deck.py` ve `build_one_pager.py`
  değerleri `reports/results/*.json`'dan okuyor, figürleri `reports/figures/` altındaki
  commit'li PNG'lerden olduğu gibi gömüyor. Gerekçe: rapor/sunum artefaktlardan sapamasın.
  Tek istisna one-pager'daki yarı ömür grafiği — A4'e sığması için tek panel olarak yeniden
  çiziliyor, ama çizdiği değerler `marketing_metrics.json`'daki kova ortalamalarının ta
  kendisi.
- `python-pptx` **`requirements.txt`'e eklenmedi**: deck'i üretmek için gerekli, pipeline'ı
  koşturmak için değil. `presentation/README.md` kurulumu söylüyor.
- **22 slaytın hepsi PowerPoint COM ile PNG'ye aktarılıp tek tek gözle kontrol edildi**;
  bulunan taşmalar ve çakışmalar betikte düzeltilip yeniden üretildi.

**Bu sırada düzelen iki ifade hatası (sunumda ve ekte):** Kapı 2'nin 3. ölçütü "plasebolar
C0'dan **kötü**" değil "**hiçbiri C0'dan iyi değil**" (kod `ci[0] <= 0` arıyor; Grocery'nin
iki hücresinde GA sıfıra değiyor ama üstüne çıkmıyor). 4. ölçüt "seed yayılımı etkiden küçük"
değil, **C0'ın seed'ler arası değişim katsayısı `gate2.seed_cv_max` eşiğinin altında**.

**F18 İngilizceye çevrildi.** F17 ve F19–F23 İngilizceydi, F18 Türkçeydi; rapor ve sunum
İngilizce olduğu için figür de İngilizce oldu. `validation` yeniden koşuldu,
`validation_500.json`'da **yalnızca `code_version`** değişti. `F20`'nin alt başlığındaki tek
Türkçe parantez düzeltilmedi: o dize `distill_report_base.json`'ın `meta` alanında duruyor ve
düzeltmek ya damıtmayı yeniden koşmayı ya da makine tarafından yazılmış bir artefaktı elle
düzenlemeyi gerektirirdi (`presentation/README.md` → "Bilinen kusur").

**SONUCLAR §7 yeniden numaralandı (1..16).** Listede iki kez "8." vardı ve son madde "15."
diye bitiyordu; madde sayısı hep 16'ydı, numaralar yanlıştı.

**Ekip adları dokunulmadı:** `[Team Member 1/2/3]` ve `*(doldurulacak)*` yer tutucuları
raporda, deck'te ve one-pager'da olduğu gibi duruyor.

**Etkilediği bölüm:** `final-report/` (yeni), `presentation/` (yeni),
`analysis/validation.py` (yalnızca figür metinleri), kök `README.md`, `docs/GENEL_BAKIS.md`,
`docs/SONUCLAR.md`, `implementation-plan/implementation-plan.md`
**Kim:** Ekip

---

### 2026-09-21 — ÖNCEDEN KAYIT: B/C etiketlemesi, seed 42 yeniden koşusu, iki post-hoc analiz (YENİ HİÇBİR SONUÇ GÖRÜLMEDEN)

> Bu kayıt, B ve C'nin sayfaları geri gelmeden ve Kaggle yeniden koşusu başlamadan commit
> edildi. Aşağıdaki kurallar o sonuçlar görüldükten sonra değiştirilmez.

**Karar (kullanıcı, 2026-09-21):** İki açık sınırlılık kapatılmaya çalışılacak. (1) 500
satırlık doğrulama setini iki ekip üyesi (B, C) de etiketleyecek. (2) Seed 42'nin 24 hücresi
Kaggle'da bugünkü kodla yeniden koşacak.

#### κ — Hafta 4 kapısı

- İstatistik **Fleiss κ**, A + B + C, beş sınıf. Eşik `validation.kappa_min` = **0,60**,
  seyrek sınıf kuralı `validation.min_class_n_for_kappa` = 20. Üçü de 2026-08-29'dan beri
  config'te; **değişmiyor**.
- Karar PASS ya da FAIL yazılır. **FAIL çıkarsa** yeniden etiketleme yapılmaz, rehber
  revize edilip ikinci tur açılmaz, eşik gevşetilmez: sonuç olduğu gibi raporlanır.
- B ve C **bağımsız** etiketler: birbirleriyle ve A ile konuşmadan, A'nın sayfasını ve model
  çıktısını görmeden, `docs/ETIKETLEME_REHBERI.md` ile. Sayfalar körlenmiş (LLM'in cevabı
  yok) — 2026-08-26'dan beri öyle üretiliyor.

#### Referans — yayımlanan sayılar A'ya göre kalır

- 2026-09-14 kaydı ölçüm yöntemlerini tek etiketleyiciyle sabitledi ve aşağı akıştaki her sayı
  A referansıyla üretildi: kalibre yaygınlık (RQ1), etiket kalitesi
  (`label_quality_population`), damıtma kapısının 3. ölçütü, deneyin yorum kuralının yanında
  duran K/D değerleri. Referansı şimdi çoğunluk uzlaşısına çevirmek, sonuç görüldükten sonra
  tanım değiştirmek olurdu. **Birincil referans A kalır** (`validation.primary_reference: A`).
- **Çoğunluk uzlaşısı** (üçün en az ikisi; uzlaşısız `tie` satırları dışarıda) **duyarlılık**
  bloğu olarak eklenir: LLM–insan ölçümleri ve kalibre yaygınlık. Birincil sayının yerine
  geçmez.
- Damıtma kapısının 3. ölçütü çoğunluk referansıyla **yeniden hesaplanamaz**: öğrencinin 500
  satırdaki satır bazlı tahminleri saklanmadı, Kaggle raporunda yalnızca toplamlar var. Bu
  açıkça yazılır.

#### Seed 42 yeniden koşusu

- İki amaç: (a) 72 koşunun hiçbirinde kaydedilmemiş **epoch sayısı** (SONUCLAR §7 madde 6);
  (b) bugünkü kodun yayımlanan sayıları ürettiğinin **doğrudan** sınanması.
- Girdi: 2026-09-15'te yüklenen özel dataset. Yereldeki `.inter` dosyalarıyla SHA-256
  düzeyinde aynı (ölçüldü 2026-09-21). Notebook girdilerin özetini oturum özetine yazacak.
- **"Birebir aynı"** için üçü birden gerekir: kullanıcı başı parquet'in bütün metrik kolonları
  ve top-K listeleri eşit · rapordaki `test` metrikleri eşit · `test_pairs_sha256` eşit. Biri
  tutmazsa fark hücre hücre raporlanır ve **orijinal 72 koşu birincil kalır**; yeniden
  koşunun sayıları hiçbir sonuç tablosuna girmez.
- Seed 1337 ve 2024'ün epoch sayısı ölçülmez, **tahmin edilir**: `epochs_42 × fit_s / fit_42`
  (aynı hücrede epoch başına sürenin sabit olduğu varsayımıyla; ikiz koşularda süre %7'ye kadar
  oynuyordu). "Tahmin (±%10)" diye etiketlenir.
- Herhangi bir koşu 300 tavanına **değerse** (ya da tahmini tavanın %90'ını aşarsa) iş durur ve
  kullanıcıyla karar verilir. Daha yüksek tavanla yeniden koşu şimdiden kararlaştırılmıyor.

#### İki post-hoc analiz (sonuçlar görüldükten SONRA tanımlandı)

2026-09-21 denetiminde bulundu. 72 koşunun sonuçları görülmüştü; aşağıdaki iki analizin
**nokta tahminlerine de** denetim sırasında bakıldı. İkisi de `post_hoc: true` damgası
taşır, hiçbir birincil sayının yerine geçmez (`analysis/robustness.py`).

1. **Seed düzeyi.** Önceden kayıtlı her karşıtlığın üç seed'deki farkı, ortalaması, sd'si ve
   işaret uyumu. Sebep: eşli bootstrap kullanıcıları yeniden örnekliyor; eğitimin (seed'in)
   değişkenliğini kapsamıyor.
2. **Maruziyet ayrıştırması.** Test kullanıcıları dört gruba ayrılır: "C1 bu kullanıcının bir
   eğitim satırını sildi mi" × "C4 sildi mi" (C1b × C4b için de aynı). Her grupta eşli
   bootstrap. Sebep: plasebo satır **sayısını** korpus düzeyinde eşliyor, kullanıcı düzeyinde
   eşlemiyor.

**Etkilediği bölüm:** `configs/base.yaml` (`validation`), `analysis/validation.py`,
`analysis/prevalence.py`, `analysis/robustness.py` (yeni), `analysis/reproduction.py` (yeni),
`recsys/run_experiment.py` (yalnızca rapor alanları), `scripts/kaggle_experiment.py`
(yalnızca girdi özeti)
**Kim:** Kullanıcı (ekip adına) · kayıt: ekip

---

### 2026-09-21 — İkinci denetim: bulgular ve düzeltmeler (birincil sayı DEĞİŞMEDİ)

**Karar:** Proje baştan sona yeniden denetlendi. Denetimde ham veriden değişmezler bağımsız
olarak yeniden hesaplandı, kritik kod yolu okundu ve yayımlanan iddialar veriyle
karşılaştırıldı. Bulunan her şey ya düzeltildi ya da sınırlılık olarak yazıldı. Hiçbir
birincil sayı değişmedi; tek istisna makro-F1'in 4. basamağı.

#### Bağımsız yeniden hesapla doğrulananlar (sorun yok)

- **5-core ve bölme:**
  - Toys 2.164.018, Grocery 2.434.594 satır; en düşük kullanıcı ve ürün derecesi 5.
  - Yinelenen (kullanıcı, ürün) çifti 0; bütün satırlar `verified`.
  - Her kullanıcıda test = son satır, valid = sondan ikinci; train ≤ valid ≤ test.
- **Koşullar:**
  - Beş koşulda da valid ve test satırları C0 ile birebir aynı.
  - C1 eğitimdeki bütün `gift_given` satırlarını siliyor; C4 aynı sayıda satır siliyor
    (C1b/C4b için de aynısı).
  - C3'ün gölge satırları yalnızca eğitimde.
- **Etiketler:** öğrenci etiketleri 5-core'un her satırında ve hepsi `distilled`. LLM–insan
  sınıf bazlı P/R/F1 yeniden hesaplandı ve tuttu.
- **Kaggle girdisi:** yüklenen `.inter` ve `split.json` dosyaları yereldekilerle SHA-256
  düzeyinde aynı.
- **Kod farkı:** koşulardan (`f3183df`) bu yana deney kodundaki fark yalnızca rapor
  alanları, bir koruma ve aynı formülle taşınmış `seed_for`.

#### Bulunan ve düzeltilen kusurlar

- **Makro-F1 yuvarlama hatası.** `validation.compare` sınıf F1'lerini 3 basamağa yuvarlayıp
  sonra ortalıyordu. Rapor 0,5405 yerine **0,5404** yazıyordu; aynı dosyadaki bootstrap bloğu
  zaten 0,5405'ti.
  - `validation_500.json` yeniden üretildi; diff'te yalnızca `macro_f1` ve damga değişti.
  - Güncellenen belgeler: rapor, SONUCLAR ve concept-note / technology-review /
    implementation-plan errataları. Eski DECISIONS kayıtlarındaki 0,5404 tarihî olarak kaldı.
- **F20'deki Türkçe parantez figür kodundan geliyordu.** Figür, damıtma raporunun makinenin
  yazdığı `thresholds_fixed` notunu olduğu gibi basıyordu; artık yalnızca tarihi basıyor.
  - 2026-09-21 capstone kaydındaki "düzeltmek damıtmayı yeniden koşmayı gerektirir"
    değerlendirmesi **yanlıştı**.
  - F21–F23 bayt düzeyinde aynı kaldı.
- **`marketing_metrics.json`'ın kirli damgası** (`3f904a6-dirty`). Makinede başka bir şey
  koşmazken tek başına koşuldu; 3 dakika sürdü ve yalnızca damga değişti.
- **Damga, izlenmeyen kod dosyasını saymıyordu.** `code_version` `--untracked-files=no`
  kullanıyordu; commit'lenmemiş yeni bir modülün çıktısı, o modülü içermeyen bir commit'le
  temiz damgalanıyordu. Bu, `robustness_posthoc.json`'ın ilk koşusunda görüldü.
  - Bir test tam tersini kilitliyordu; test tersine çevrildi.
  - `__pycache__` ve `egg-info` gitignore'da olduğu için sayılmıyor.

#### Geri çekilen ya da daraltılan iddialar

- **"Valid satırındaki hediye C1 − C4 ve C1 − C0 farklarını küçültür, büyütmez."** Bu iddia
  2026-09-19 kaydında, SONUCLAR §7 madde 5'te, GENEL_BAKIS'ta, raporda ve deck'te geçiyordu.
  Bir varsayımdı: SASRec'te hediye satırlarını silmek zaten net zararlı (C1 − C0 < 0), yani
  yön sınanmadı. Beş yerde "yön sınanmadı" diye düzeltildi. 2026-09-19 kaydının metnine
  dokunulmadı; bu kayıt onu düzeltiyor.
- **"Alt sınır" gerekçesi.** Önceden kayıtlı kural değişmedi. Ama kuralın gerekçesi kaçırılan
  hediyeler için doğru; yanlış pozitifler için ise o satırların rastgele bir satır kadar
  bilgi taşımasını varsayıyor. Yanlış pozitiflerin çoğu kendi çocuğuna alım, ve C1b sonucu bu
  satırların da ortalamadan az bilgi taşıdığını düşündürüyor.
  - Raporun "gürültülü detektör plasebo kontrollü bir farkı üretemez, yalnızca gizler"
    cümlesi ve deck slayt 7'deki "yön zayıf detektörün eseri olamaz" kutusu düzeltildi.
  - Varsayım SONUCLAR §7 madde 19'a yazıldı.
- **Küçük ifadeler:**
  - "Far cheaper than it is usually assumed" cümlesi kaynaksızdı, çıkarıldı.
  - "Five slots in a hundred" yalnızca BPR için geçerli (SASRec'te 1,3); öyle yazıldı.
  - "Gift rows should be down-weighted" bir öneri değil, çıkarım olarak yazıldı.
  - Kök README'deki "Grocery'de saptanamıyor" → "çok küçük ya da saptanamıyor".
  - Erken durdurmanın hediye içeren valid hedefleri SONUCLAR §7 madde 7'ye eklendi.

#### Post-hoc analizler (ön kayıtta işaretli; `robustness_posthoc.json`)

- **Seed düzeyi:**
  - Toys'ta her karşıtlıkta üç seed de aynı işareti veriyor; C1 − C4, seed ortalamasının
    standart hatasının 13–17 katı.
  - Grocery'de seed yayılımı kullanıcı GA'sıyla aynı mertebede. SASRec C1 − C4'te üç seed de
    pozitif; BPR C1 − C4'te işaret değişiyor (1/3).
- **Maruziyet ayrıştırması:**
  - C1, Toys test kullanıcılarının %44'üne dokunuyor; C4 %73'üne.
  - Toys'ta C1 − C4 dört grubun dördünde pozitif; **iki koşulun da dokunmadığı** kullanıcılarda
    SASRec +0,677 [+0,507, +0,864], BPR +0,515 [+0,389, +0,638].
  - Grocery'de pozitif ortalama tedavinin dokunmadığı kullanıcılardan geliyor.

Yeni sınırlılıklar: SONUCLAR §7 madde 17–19, rapor §9 madde 6–8, deck A4 satır 7–9.

#### Kod (sonuç değiştirmeyen)

- `validation.primary_reference` (A): doğrulama, kalibrasyon ve damıtma paketi aynı birincil
  referansı okuyor. Çok etiketleyicide çoğunluk duyarlılık bloğuna yazılıyor. `validation_500`
  ve `prevalence` yeniden üretildi; yalnızca `primary_reference` alanı ve damga eklendi.
- `analysis/robustness.py` ve `analysis/reproduction.py` eklendi.
- Koşu raporu artık `best_epoch`, `epochs_since_best`, `stopping_step` ve `versions` yazıyor.
  Gerçek RecBole trainer'ıyla yerelde 2 epoch'luk duman koşusunda doğrulandı.
- Kaggle hücresi girdi SHA-256'sını oturum özetine yazıyor.

**Testler:** 441 → 463.

**Etkilediği bölüm:** `analysis/validation.py`, `analysis/prevalence.py`,
`analysis/result_figures.py`, `analysis/robustness.py`, `analysis/reproduction.py`,
`detection/distill.py`, `recsys/run_experiment.py`, `utils/io.py`,
`scripts/kaggle_experiment.py`, `configs/base.yaml`, SONUCLAR, GENEL_BAKIS, CLAUDE.md,
ETIKETLEME_REHBERI, kök README, final raporu, deck, one-pager, üç teslim belgesinin
errataları
**Kim:** Ekip (denetim)
