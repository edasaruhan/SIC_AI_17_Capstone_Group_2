# Hediye Kontaminasyonu Projesi — Detaylı Roadmap

**Çalışma başlığı:** *Bu Benim İçin Değildi: Hediye Alımlarının Öneri Sistemlerindeki Tercih Sinyalini Bozması ve LLM Tabanlı Gürültü Giderme*

---

## 1. Düzeltilmiş Gap Analizi (önce bunu okuyun)

Önceki mesajda "kimsenin çözmediği bir problem" dedim. Bu fazla iddialıydı. Detaylı taramadan sonra gerçek tablo:

### Mevcut olan işler

**A) Wang et al., WSDM 2020 — "Time to Shop for Valentine's Day"** (Texas A&M + Etsy)
En yakın iş. Sequential recommendation modellerinin kullanıcının uzun vadeli tercihini yakaladığını, ama doğum günü, yıldönümü, Sevgililer Günü, Anneler Günü gibi **occasion'ların bu tercihten sapmalara yol açtığını** söylüyorlar ve bu sapmayı modelleyen bir sistem öneriyorlar.
🔗 [PDF](https://people.engr.tamu.edu/caverlee/pubs/wang20wsdm-valentine.pdf) · [Kod](https://github.com/wangjlgz/Occasion-Aware-Recommenation)

**Bizden farkı — üç noktada:**
1. Onlar occasion'ı **tahmin gücü artırmak için modelliyor**; biz onu **temizlenecek gürültü** olarak ele alıyoruz. Amaç zıt: onlar "Sevgililer Günü'nde ne alacağını bilelim", biz "Sevgililer Günü alımından sonra bu insana yanlış şey önermeyi bırakalım".
2. Onlar occasion'ı **takvim/zaman sinyalinden** çıkarıyor; biz **metinden semantik olarak** çıkarıyoruz. Takvim yaklaşımı Kasım'daki hediyeyi yakalar, Mart'ta yeğenin doğum günü için alınan hediyeyi kaçırır.
3. Etsy'nin **özel verisini** kullanmışlar — replike edilemez. Biz kamuya açık veri kullanacağız.

**B) Amazon patentleri** (US 9818145, 10445809, 8352331, 11367117)
Problem sanayide **açıkça tanınmış**. Patentlerden biri şunu diyor: alıcı hediye paketi veya hediye notu isterse ürün davranış analizinden çıkarılabilir, ama **çoğu durumda satıcı bunun hediye olduğunu belirleyemez ve kullanıcı profilinde bozulmalara yol açar** — özellikle az veri noktası varken her hediye alımının bozucu etkisi çok güçlü oluyor.

Bu bizim için **kötü değil, çok iyi haber**: problemin iş değeri patentle kanıtlanmış, ama çözüm hediye paketi/not gibi **açık sinyallere** dayanıyor, metinden çıkarıma değil. Ve akademik literatürde nicelleştirilmemiş.

**C) Denoising implicit feedback literatürü** (büyük ve olgun)
Gürültüyü yanlış tıklama, pozisyon bias'ı, memnuniyetsizlik, iade olarak tanımlıyor. **"Bu alım başka biri için yapıldı"** diye ayrı ve tespit edilebilir bir gürültü sınıfı yok.

**D) Gift recommendation literatürü** — ters problem (hediye alacak kişiye öneri).

**E) Zhao et al., WWW 2020 "Target Customer Distortion"** — isim benziyor ama farklı problem (bir ürünün hangi müşteri kitlesine önerildiğinin kalibrasyonu). Bizimle ilgisi yok.

### Geriye kalan gerçek gap

Üç somut boşluk kaldı, hepsi dar ve savunulabilir:

| # | Boşluk | Neden boş |
|---|---|---|
| **G1** | Hediye alımlarının kamuya açık veride **yaygınlığı ölçülmemiş** | Etsy verisi özel; Amazon verisinde hediye etiketi yok. LLM'ler olmadan metinden çıkarmak pratik değildi. |
| **G2** | Kontaminasyonun öneri kalitesine **etkisi ölçülmemiş** | Kimse "hediyeleri çıkarınca ne kadar iyileşiyor" sorusunu deneysel olarak sormamış |
| **G3** | **Modelleme mi, temizleme mi** karşılaştırılmamış | Wang et al. modelliyor. Temizlemenin ne zaman daha iyi olduğu bilinmiyor. |

**Capstone iddianız bu üçü.** "Yeni bir algoritma icat ettik" değil — *"bilinen ama ölçülmemiş bir problemi kamuya açık veride ölçtük ve müdahalenin etkisini test ettik."* Bu bir capstone için doğru büyüklükte ve dürüst bir iddia.

---

## 2. Araştırma Soruları

**RQ1.** Amazon review'larında hediye/başkası-için alımların oranı nedir, ve bu oran kategori ve mevsime göre nasıl değişir?

**RQ2.** Hediye etkileşimlerini eğitim verisinden çıkarmak (veya ağırlığını düşürmek), kullanıcının **kendi** sonraki alımlarını tahmin etme başarısını artırır mı? Etki büyüklüğü kategorinin hediye yoğunluğuyla orantılı mı?

**RQ3.** Hediye sinyalini **silmek** mi yoksa modele **ayrı bir sinyal olarak vermek** mi daha iyi? (Wang et al. yaklaşımına karşı doğrudan test)

**RQ4.** Bir hediye alımından sonra öneri listesi ne kadar süre "kirli" kalıyor? (Pazarlama açısından en anlamlı çıktı)

---

## 3. Veri: Kesin Seçimler

### Ana kaynak
**Amazon Reviews 2023** (McAuley Lab, UCSD) — 571.54M review, Mayıs 1996 – Eylül 2023, 33 kategori.
🔗 [Ana sayfa](https://amazon-reviews-2023.github.io/) · [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) · [Paper](https://arxiv.org/abs/2403.03952)

**Kritik alanlar (doğrulandı):**

| Alan | Tip | Neden gerekli |
|---|---|---|
| `text` | str | Hediye tespitinin tek kaynağı |
| `title` | str | Kısa ama sinyal yoğun ("Perfect gift!") |
| `rating` | float | Hediye vs kendi alımı rating dağılımı karşılaştırması |
| `timestamp` | int (unix) | **Mevsimsellik doğrulaması bu alana bağlı** |
| `user_id` | str | Sekans kurma |
| `parent_asin` | str | Metadata join anahtarı (`asin` değil!) |
| `verified_purchase` | bool | Kalite filtresi |

Metadata tarafında: `title`, `categories`, `price`, `description`, `main_category`.

### Kategori seçimi — üçlü tasarım

Bu projenin en önemli tasarım kararı. Tek kategori yerine **hediye yoğunluğu spektrumunda üç kategori** seçiyoruz:

| Rol | Kategori | #User | #Item | #Rating | Beklenen hediye oranı |
|---|---|---|---|---|---|
| **Yüksek** | `Toys_and_Games` | 8.1M | 890.7K | 16.3M | Yüksek — oyuncakların çoğu çocuğa alınır |
| **Orta** | `Video_Games` | 2.8M | 137.2K | 4.6M | Orta — yoğun item, mevsimsel hediye |
| **Düşük (kontrol)** | `Grocery_and_Gourmet_Food` | 7.0M | 603.2K | 14.3M | Düşük — insanlar yemeği kendine alır |

**Neden bu tasarım kritik:** Etki büyüklüğü hediye oranıyla orantılı çıkmalı. Grocery'de de Toys kadar iyileşme görürseniz, detektörünüz hediye değil başka bir şey yakalıyor demektir. **Bu, projenin kendi kendini yanlışlayabilme mekanizması** — jüriye anlatacağınız en güçlü metodolojik nokta.

### Pilot için
`All_Beauty` (632K user / 701.5K rating) veya `Handmade_Products` (586.6K user / 664.2K rating). Küçük, hızlı, tüm pipeline'ı 1 günde uçtan uca test edersiniz. **Handmade özellikle ilginç** — Etsy benzeri, hediye yoğun.

### Ön işleme
1. `verified_purchase == True` filtresi
2. Boş/çok kısa metin at (< 5 kelime)
3. **5-core filtreleme** (≥5 etkileşimi olan user ve item) — recsys standardı, McAuley Lab kendi split'lerini sağlıyor
4. Kullanıcı başına kronolojik sekans kur
5. Zaman bazlı split (rastgele değil — sızıntı olur)

---

## 4. Pipeline Mimarisi

```
Amazon Reviews 2023 (kategori bazlı .jsonl.gz)
        │
        ├─► [A] Stratified örnekleme: 40-60K review
        │        ↓
        │   [B] LLM annotator (Qwen3-4B-Instruct / Gemma 4 E4B, vLLM, JSON)
        │        ↓  etiket + gerekçe + alıcı ilişkisi + occasion
        │   [C] İnsan doğrulama seti (500 review, 3 annotator, kappa)
        │        ↓
        │   [D] ModernBERT-base distillation (LLM etiketleri = eğitim verisi)
        │        ↓
        └─► [E] Tüm korpusta inference (milyonlarca review, dakikalar)
                 ↓
             gift_flag ∈ {self, gift_given, household, unclear}
                 ↓
        ┌────────┴────────────────────────────────┐
        │                                          │
   [F] Betimsel analiz                     [G] Recsys deneyi
   - Kategori bazlı oran                   - 5 koşul (C0-C4)
   - Aylık mevsimsellik ★                  - SASRec + BPR-MF + ItemKNN
   - Rating dağılımı farkı                 - Non-gift test item'larda değerlendirme
   - Alıcı/occasion dağılımı                     ↓
                                            [H] Pazarlama metrikleri
                                            - Retargeting israf oranı
                                            - Kontaminasyon yarı ömrü
```

---

## 5. Aşama Detayları

### [A] Örnekleme stratejisi

Rastgele örnekleme yapmayın — hediye sınıfı azınlıkta ve zamana bağlı. **Katmanlı örnekleme:**
- Ay bazında dengeli (12 ay eşit) → mevsimsellik analizini bozmamak için
- Rating bazında dengeli
- Metin uzunluğu bazında dengeli
- Ek olarak: anahtar kelime içeren bir alt küme (`gift`, `present`, `for my`, `bought this for`) **ayrıca** örneklenmeli — ama analizde bu ayrı tutulmalı, yoksa oranı şişirir

**Hedef:** 40-60K review annotation için. Bu miktar ModernBERT'i eğitmeye fazlasıyla yeter.

### [B] LLM Annotator — model seçimi

> ⚠️ **2026-08-26'da yeniden karara bağlandı.** Bu tablo başlangıçta modelleri
> **VRAM**'e göre sıralıyordu. Ekibin donanımı ölçüldükten sonra doğru kriterin VRAM
> değil **GPU kuşağı** olduğu ortaya çıktı: yerel kart GTX 1650 Ti (4 GB, Turing sm75),
> ücretsiz bulut katmanları Kaggle 2× T4 ve Colab T4 — hepsi **pre-Ampere**. Sonuç:
> bfloat16 yok, FlashAttention yok, ve yeni mimariler Turing'i düşürüyor.
> Tam gerekçe: `technology-review/technology-review.md` §4.2.

| Model | Mimari | fp16 VRAM | Lisans | Turing? | Değerlendirme |
|---|---|---|---|---|---|
| **Qwen/Qwen3-4B-Instruct-2507** | 4.0B dense, text-only, GQA, 262K context | ~8 GB | Apache 2.0 | **Evet** | **Birincil.** T4'e kuantizasyonsuz sığıyor, KV cache'e yer kalıyor. Non-thinking — 40-60K sınıflandırmada muhakeme token'ı harcamıyor |
| **google/gemma-4-E4B** | 4.5B efektif, multimodal (PLE) | ~4-5 GB (4-bit) | Apache 2.0 | Alt örneklem | **İkincil.** ~5K uyum örneklemi için; farklı laboratuvar/veri = uyum istatistiği anlamlı olsun diye. Yüklenmezse: Gemma 3 4B-it |
| Qwen3-8B | 8B dense, GQA | ~16 GB | Apache 2.0 | Evet | 4B macro-F1 kapısını geçemezse tırmanma yolu |
| ~~Qwen3.5-9B / 4B~~ | Hibrit Gated-DeltaNet, VL | 9-20.5 GB | Apache 2.0 | **Hayır** (pratikte) | **Donanım nedeniyle elendi, kalite nedeniyle değil.** Ampere erişimi gelirse ilk tercih |
| gpt-oss-20b | 20B MoE | ~16 GB (4-bit) | Apache 2.0 | Sınırda | Muhakeme gücü bu iş için fazla |

**Karar: Qwen3-4B-Instruct-2507 birincil, Gemma 4 E4B ikincil.** İkisini birden koşturup
**model-arası uyum** raporlamak Technology Review'un merkezine oturur. 4B'nin yeterliliği
varsayım değil: kanıt metnin medyan %1-3'ünde duruyor ve naif sözcüksel vekil zaten %58.3
precision veriyor — modelin işi sıfırdan sinyal bulmak değil, bilinen bir hata desenini
(spekülatif hediye dili + alınan hediye) düzeltmek.

**Servis: vLLM, `--dtype float16`.** Bu iş offline batch olduğu için continuous batching şart.
float16 tercih değil zorunluluk: bfloat16 compute capability 8.0 istiyor. Koşu **Kaggle Linux
notebook'unda**; Windows'ta vLLM yok ve 4 GB'a model sığmıyor. Yerelde prompt denemesi için
llama.cpp/GGUF — ama veri setine giren etiketler yalnızca vLLM koşusundan gelir.

### [B2] Etiket şeması — ikili değil, dörtlü

Bu detay projeyi sıradan olmaktan çıkarır:

```json
{
  "purchase_type": "self | gift_given | household | unclear",
  "confidence": "high | medium | low",
  "recipient": "child | spouse | parent | friend | colleague | unknown | null",
  "occasion": "birthday | christmas | wedding | graduation | none | unknown",
  "evidence_span": "kızımın doğum günü için aldım"
}
```

**`household` neden ayrı:** Bebeğine bez almak hediye değil, ama alıcının kendi tercihini de yansıtmıyor. Bu ara kategori hem gerçekçi hem de RQ3 için ilginç — üçünü farklı ağırlıklandırmayı test edebilirsiniz.

**`evidence_span` neden zorunlu:** LLM'i gerekçe göstermeye zorlamak halüsinasyonu azaltır ve insan doğrulamasını 10× hızlandırır — annotator tüm review'u okumak yerine span'e bakar.

**Structured output:** vLLM'in guided decoding / JSON schema desteğini kullanın. Serbest metin parse etmeyin.

### [C] İnsan doğrulaması — pazarlık edilemez

- **500 review**, LLM etiketlerine göre katmanlı (her sınıftan yeterli örnek)
- **3 annotator** (ekip üyeleri), bağımsız etiketleme
- **Fleiss' kappa** raporlayın — insanlar arası uyum düşükse (κ < 0.6) görev tanımınız muğlak demektir, şemayı düzeltin
- LLM'in precision / recall / F1'i her sınıf için
- **Ayrı bir hata analizi:** LLM en çok hangi durumda yanılıyor? (Muhtemelen: "arkadaşım tavsiye etti" ≠ hediye, "eşim için aldım ama ikimiz de kullanıyoruz" = household mu gift mi)

### [D] Distillation — ModernBERT

40-60K LLM etiketiyle bir **ModernBERT-base** sınıflandırıcı eğitin.

**Neden distill etmek gerekli:** Toys_and_Games'te 5-core sonrası 2.16M etkileşim var. Annotator LLM ile hepsini işlemek Kaggle'ın haftalık ~30 saatlik kotasını kat kat aşar. ModernBERT saniyede binlerce örnek işler ve etiket kalitesi neredeyse aynı kalır.

**Neden ModernBERT (DeBERTa/RoBERTa değil):** hızlı inference (asıl bağlayıcı kısıt: milyonlarca satır), 149M parametre — 4 GB'lık yerel karta sığıyor, GLUE'da DeBERTaV3-base'i geçen ilk encoder, `AutoModelForSequenceClassification` ile doğrudan çalışıyor.

> ⚠️ **8192 context gerekçesi ÇÜRÜTÜLDÜ (2026-08-26).** Eski gerekçe "alıcı bilgisi review'un sonunda geçer, 512'de kesilir" diyordu. Ölçtük: kanıt gövdenin medyan %1–3'ünde duruyor, 512 token'da kayıp yalnızca %0.013–0.044. Model seçimi değişmiyor, gerekçe değişiyor. Ayrıntı: `data-research/data-research.md` §4.8, T5.

**Sağlamlık kontrolü:** ModernBERT fidelity kapısını (LLM'e 5 puan yakınlık) kaçırırsa `microsoft/deberta-v3-base` ile tekrar denenir — kontrollü karşılaştırmalar eşit veriyle DeBERTaV3'ün örneklem verimliliğinde önde olabildiğini gösteriyor ve elimizde yalnızca 40–60K etiket var.

**Doğrulama:** ModernBERT'i insan etiketli 500'lük sette de test edin. LLM'e yakın mı? Yakınsa distillation başarılı.

### [E] Tam korpus inference
ModernBERT ile. Çıktı: her etkileşim için `purchase_type` + olasılık skoru. Olasılığı saklayın — soft weighting için lazım.

---

## 6. Deneysel Tasarım — Projenin Kalbi

### Görev tanımı
Sequential recommendation, leave-one-out: kullanıcının kronolojik sekansındaki son etkileşimi tahmin et.

### ⚠️ Kritik kural: test item'ı hediye OLMAMALI

İddia "hediye alımları kullanıcının kendi tercihine dair tahmini bozuyor". Test item'ınız da hediyeyse ölçtüğünüz şey bu değil. **Sadece son etkileşimi `self` olarak sınıflanmış kullanıcılar üzerinde değerlendirin.**

### Beş koşul

| Kod | Koşul | Ne yapıyor |
|---|---|---|
| **C0** | Baseline | Tüm etkileşimlerle eğit (standart uygulama) |
| **C1** | Hard removal | `gift_given` etkileşimlerini eğitimden çıkar |
| **C2** | Soft down-weighting | Hediye etkileşimlerini loss'ta *w* ∈ {0.25, 0.5, 0.75} ile ağırlıklandır |
| **C3** | Modelleme | Hediye bayrağını **feature/token olarak ekle**, silme (Wang et al. yaklaşımı) |
| **C4** | ★ **PLACEBO** | Hediye sayısı kadar **rastgele** etkileşim çıkar |

**C4 olmadan bu proje geçersizdir.** C1 iyileşme gösterirse ilk soru "veri azaltmak zaten iyileştirir mi" olacak. C4 bu soruyu kapatır. C1 > C4 ise etki gerçek; C1 ≈ C4 ise bulgunuz gürültü.

**İkinci placebo (opsiyonel ama güçlü):** Etiketleri kullanıcılar arasında karıştırıp (shuffle) aynı sayıda etkileşim çıkarın. Hediye *dağılımının* mı yoksa *kimliğinin* mi önemli olduğunu ayırır.

### Modeller

| Model | Rol | Neden |
|---|---|---|
| **Popularity** | Alt sınır | Her recsys paper'ının ilk baseline'ı |
| **ItemKNN** | Klasik CF | Basit, yorumlanabilir, sürprizli derecede güçlü |
| **BPR-MF** | Matrix factorization | Sekans dışı klasik referans |
| **SASRec** | **Birincil** | Self-attention sequential; hediye "sekans içi kesinti" olduğu için doğru mimari |
| **GRU4Rec** | Sekans alternatifi | SASRec bulgusunun mimariye özgü olmadığını göstermek için |

**Kütüphane: RecBole.** Beşi de hazır implement, tek konfigürasyon dosyası, aynı değerlendirme protokolü. Kendiniz yazarsanız haftalarca hata ayıklarsınız ve sonuçlarınız literatürle kıyaslanamaz.

**Toplam koşu:** 5 koşul × 5 model × 3 kategori = 75 run. Fazla geliyorsa: SASRec + BPR-MF (2 model) × 5 koşul × 3 kategori = 30 run. Tüketici GPU'sunda makul.

---

## 7. Doğrulama Stratejisi

Bu bölüm projeyi öğrenci ödevinden ayıran şey.

### V1 — İnsan etiketi (iç geçerlilik)
500 review, 3 annotator, kappa + F1. Yukarıda anlatıldı.

### V2 — ★ Mevsimsellik (dış geçerlilik, etiketsiz)
Aylık hediye oranını çizin. Beklenen:
- **Kasım-Aralık patlaması** (Noel/yılbaşı)
- **Şubat tepesi** (Sevgililer Günü)
- **Mayıs tepesi** (Anneler Günü)

Detektörünüz bu takvimi tanımıyor, sadece metin okuyor. Yine de bu deseni üretiyorsa, **hiç insan etiketi olmadan** çalıştığının kanıtıdır. Tek grafik, çok güçlü argüman.

⚠️ **Dikkat:** Review tarihi satın alma tarihi değil — insanlar aldıktan haftalar sonra yorum yazar. Tepeler bir miktar gecikmeli olacak. Bunu önceden söyleyin, sonradan savunma pozisyonuna düşmeyin.

### V3 — Kategori yüz geçerliliği
Toys > Video Games > Grocery sıralaması çıkmalı. Çıkmıyorsa detektör bozuk.

### V4 — Rating dağılımı
Hediye alımlarının rating dağılımı farklı olmalı (muhtemelen daha yüksek ortalama ve daha düşük varyans — ürünü kullanmayan kişi eleştiremez). Fark yoksa şüphelenin.

### V5 — Prompt ve model duyarlılığı
2 prompt × 2 model = 4 kombinasyon. Hediye oranı %12'den %30'a fırlıyorsa bu bir limitasyondur ve **raporlanmalıdır**. Gizlemeyin — dürüst raporlama, sahte kesinlikten iyidir.

---

## 8. Değerlendirme Metrikleri

### Teknik katman
- **Recall@10, NDCG@10, HR@10** — non-gift test item'larında
- Bootstrap güven aralığı (tek sayı değil aralık)
- Farklı random seed'lerle 3-5 tekrar (recsys sonuçları seed'e duyarlıdır)

### ★ Pazarlama katmanı (capstone'un asıl istediği)

**M1 — Retargeting israf oranı**
Kullanıcının sadece hediye alımı üzerinden girdiği kategorilerden gelen öneri slotlarının yüzdesi. Doğrudan "boşa giden gösterim" demek.

**M2 — Kontaminasyon yarı ömrü** *(en iyi grafik)*
Bir hediye alımından sonra, sonraki önerilerin yüzde kaçı hediyenin kategorisinden geliyor — zamana karşı çizin. "Bir hediye alımı öneri listenizi ortalama X hafta kirletiyor" cümlesi jüriye tek slaytta geçer.

**M3 — Kullanıcı segmenti analizi**
Az etkileşimli kullanıcılarda etki daha büyük olmalı — Amazon patentinin de dediği gibi, veri azken tek bir hediye alımı profili orantısız bozar. Bunu doğrulamak hem bulgu hem de patentle örtüşme kanıtı.

---

## 9. Teknoloji Yığını — Seçimler ve Gerekçeler

| Katman | Seçim | Alternatifler | Gerekçe |
|---|---|---|---|
| Veri yükleme | `datasets` (HF) + `polars` | pandas | Polars 10M+ satırda pandas'tan çok hızlı ve bellek dostu |
| LLM servis | **vLLM** | Ollama, llama.cpp, TGI | Offline batch için continuous batching şart; 5-10× hız |
| Annotator LLM | **Qwen3-4B-Instruct-2507** (+ Gemma 4 E4B) | Qwen3.5 (Turing'de koşmuyor), gpt-oss-20b | Apache 2.0, structured output, **pre-Ampere GPU'da çalışan** dense/GQA mimari |
| Structured output | vLLM guided decoding / `outlines` | regex parse | JSON şeması garantili; parse hatası sıfır |
| Distillation hedefi | **ModernBERT-base** | DeBERTa-v3 (sağlamlık kontrolü), RoBERTa | Hızlı inference + 149M ayak izi. **8192 context gerekçesi değil** — ölçümle çürütüldü |
| Eğitim | HF `transformers` + `accelerate` | PyTorch Lightning | Standart, tüm ekip biliyor |
| Recsys | **RecBole** | Cornac, implicit, RecPack | 100+ algoritma, tek protokol, literatürle kıyaslanabilir |
| Deney takibi | **Weights & Biases** (free tier) | MLflow, TensorBoard | 75 run'ı takip etmenin tek makul yolu |
| Annotation UI | **Argilla** veya basit Streamlit | Label Studio | 500 review için Streamlit yeter |
| İstatistik | `scipy`, `statsmodels` | — | Bootstrap CI, anlamlılık testleri |
| Görselleştirme | `matplotlib` + `seaborn` | plotly | Yayın kalitesi grafik |
| Versiyon | Git + GitHub + DVC (opsiyonel) | — | Submission zaten GitHub'a |

**Compute:** RTX 3060/4060 (8-12GB) yeter. Kaggle free tier (2× T4, haftada ~30 saat) yedek. Colab free (T4) pilot için.

**Toplam maliyet: $0.**

---

## 10. Zaman Planı (8 hafta)

| Hafta | İş | Çıktı |
|---|---|---|
| **1** | Literatür taraması (Wang et al., denoising, patentler). Pilot kategori (`All_Beauty`) indir, EDA. | Lit review taslağı, veri anlaşıldı |
| **2** | Etiket şeması tasarımı. Prompt geliştirme. 200 review'da manuel deneme. Şemayı revize et. | Kararlı annotation şeması |
| **3** | Kaggle'da vLLM kurulum. Qwen3-4B-Instruct ile pilot kategoride 10K annotation. İlk mevsimsellik grafiği. | **Karar noktası: detektör çalışıyor mu?** |
| **4** | 500 review insan etiketleme (3 kişi). Kappa + F1. Prompt/model duyarlılık analizi. | Doğrulanmış detektör + hata analizi |
| **5** | 3 kategoride tam annotation (40-60K). ModernBERT distillation. Tam korpus inference. | Etiketli tam veri seti |
| **6** | RecBole kurulumu. C0 + C4 (baseline + placebo) tüm kategorilerde. | **Karar noktası: etki var mı?** |
| **7** | C1, C2, C3 koşulları. Bootstrap CI, seed tekrarları. | Tam sonuç tablosu |
| **8** | Pazarlama metrikleri (M1-M3). Üç bölümün yazımı. Grafikler. | Teslim |

**İki karar noktası kritik:** Hafta 3 sonunda detektör mevsimsellik göstermiyorsa, prompt/model değiştirin — devam etmeyin. Hafta 6 sonunda C0 ile C4 arasında anlamlı fark varsa (olmamalı), deney kurulumunuzda hata var demektir.

---

## 11. Riskler ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| **Hediye oranı çok düşük çıkar (%2-3), sinyal yok** | Orta | Yüksek | Toys/Handmade gibi hediye yoğun kategoriye kay. Ayrıca: düşük oran da bir bulgu — "review veren kişiler ağırlıkla kendi alımlarını yorumluyor" |
| **C1 ile C0 arasında fark yok** | **Yüksek** | Orta | **Bu bir başarısızlık değil.** "Kontaminasyon %X ama modern sequential modeller buna dayanıklı" tamamen yayınlanabilir bir negatif bulgudur. Baştan bu çerçeveyle yazın. |
| Seçim yanlılığı (sadece review'lu alımlar) | Kesin | Orta | Limitations'da açıkça yazın. İddianızı "review'lu etkileşimler arasında" diye sınırlayın. Bu Data Research bölümünüzü **güçlendirir**. |
| LLM etiketleri tutarsız | Orta | Yüksek | V5 duyarlılık analizi zaten planda. `evidence_span` zorunluluğu halüsinasyonu kısar. |
| Review tarihi ≠ alım tarihi (mevsimsellik bulanır) | Kesin | Düşük | Önceden söyleyin. Tepe konumu değil, tepenin *varlığı* önemli. |
| Kapsam şişmesi (3 kategori × 5 model × 5 koşul) | Yüksek | Orta | Öncelik: Toys + Grocery, SASRec + BPR-MF, C0/C1/C4. Gerisi "nice to have". |
| `household` sınıfı belirsiz, kappa düşük | Orta | Orta | Kappa < 0.6 ise `household`'ı `unclear` ile birleştirip üçlü şemaya inin |

---

## 12. Capstone Bölümleriyle Eşleme

### Literature Review — beş tema
1. **Occasion/context-aware recommendation** — Wang et al. WSDM 2020 (merkez), Hypergraph temporal intent modeling WWW 2025
2. **Denoising implicit feedback** — Wang et al. "Learning Robust Recommender from Noisy Implicit Feedback" (TKDE), LLM-enhanced hard sample identification (2024), natural noise literatürü
3. **Sequential recommendation** — SASRec, BERT4Rec, GRU4Rec, Mamba4Rec
4. **LLM as annotator / weak supervision** — LLM tabanlı veri etiketleme, distillation
5. **Endüstri kaynakları** — Amazon patentleri (US 9818145, 10445809), Etsy vakası (Wang et al. yazarları Etsy'den)

**Gap cümlesi:** *"Occasion literatürü sapmayı modelliyor, denoising literatürü gürültüyü siliyor; ama hiçbiri hediye alımını ayrı ve metinden tespit edilebilir bir gürültü sınıfı olarak ele almıyor ve kamuya açık veride nicelleştirmiyor."*

### Data Research
Amazon Reviews 2023 üzerinde: source, access, format, size, time period (1996-2023), granularity (etkileşim seviyesi), variables (yukarıdaki tablo), quality (verified_purchase filtresi, boş metin), **imbalance** (hediye sınıfı azınlık), **bias** (seçim yanlılığı — review'lu alımlar), **privacy** (anonim user_id ama review metni serbest metin, isim geçebilir → anonimleştirme tartışması), limitations. Artı: 3 kategorinin karşılaştırmalı EDA'sı, mevsimsellik grafiği.

### Technology Review — dört karşılaştırma ekseni
1. **Annotator LLM'leri:** Qwen3-4B-Instruct-2507 vs Gemma 4 E4B — uyum, hız, VRAM, lisans, GPU kuşağı uyumluluğu
2. **LLM vs distilled encoder:** Qwen3-4B-Instruct-2507 vs ModernBERT — doğruluk/hız/maliyet dengesi
3. **Recsys mimarileri:** ItemKNN vs BPR-MF vs SASRec vs GRU4Rec — gürültüye dayanıklılık farklı mı?
4. **Müdahale stratejileri:** hard removal vs soft weighting vs feature-as-signal
5. **Servis altyapısı:** vLLM vs Ollama throughput karşılaştırması

Limitations bölümü için hazır malzeme: LLM halüsinasyonu, prompt duyarlılığı, distillation kaybı, privacy (review metninde kişisel bilgi), açık modellerin çok dilli performans farkı.

---

## 13. İlk Hafta — Somut Başlangıç

```bash
pip install datasets polars vllm transformers accelerate recbole wandb
```

```python
# Pilot: en küçük kategori, uçtan uca boru hattını test et
from datasets import load_dataset

ds = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_All_Beauty",
    split="full",
    trust_remote_code=True,
)

# İlk sağlık kontrolü: naif anahtar kelime taraması
import re
GIFT_HINTS = r"\b(gift|present|bought (this )?for (my|her|his)|for my (son|daughter|mom|dad|wife|husband|friend|niece|nephew))\b"

hits = [r for r in ds.select(range(20000))
        if re.search(GIFT_HINTS, (r["text"] or "").lower())]
print(f"Naif tarama oranı: {len(hits)/20000:.1%}")
for r in hits[:15]:
    print("—", r["text"][:180])
```

**Bu 10 satırın amacı LLM değil, karar.** Naif tarama %1'in altında sonuç veriyorsa ve örnekler alakasızsa, kategori seçimini gözden geçirin. %5-15 arası bir şey görüyorsanız ve örnekler gerçekten hediye ise, LLM'e geçin — çünkü LLM naif taramanın kaçırdıklarını ("aldım ve kızım bayıldı") yakalayacak ve yanlış pozitifleri ("harika bir hediye olur" — ürün açıklamasından etkilenmiş yorum) eleyecek.

Ardından ay bazlı oranı çizin. Aralık'ta tepe görüyorsanız, proje çalışıyor demektir.
