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
Pilot kategoride iki çerçeve havuzlanırsa oran **%2.13 yerine %15.01** görünüyor: yedi kat
şişme. `tests/test_sampling.py::test_pooling_the_frames_inflates_the_rate` bunu sayıyla
kilitliyor, ki `sample_frame` ayrımını kaldırmaya kalkan biri neyi kaybettiğini görsün.

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
