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
