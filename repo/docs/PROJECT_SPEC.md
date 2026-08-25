# Proje Dokümanı — Hediye Kontaminasyonu

**Başlık:** *Bu Benim İçin Değildi: Hediye Alımlarının Öneri Sistemlerindeki Tercih Sinyalini Bozması ve LLM Tabanlı Gürültü Giderme*

**Durum:** Öneri aşaması — ekip onayı bekliyor
**Kapsam:** AI in Marketing Capstone — Literature Review + Data Research + Technology Review

---

## 0. Bir Sayfada Proje

| | |
|---|---|
| **Problem** | Öneri sistemleri "satın alma = tercih" varsayar. Hediye alımlarında bu varsayım yanlıştır ve kullanıcı profilini kirletir. |
| **İddia** | Bu kontaminasyon bilinen ama kamuya açık veride hiç ölçülmemiş bir problemdir. Ölçeceğiz ve müdahalenin etkisini test edeceğiz. |
| **Veri** | Amazon Reviews 2023 (571.5M review, açık, doğrulandı) — 3 kategori |
| **Yöntem** | Açık LLM ile review metninden hediye tespiti → küçük modele distillation → öneri modelini hediyeli/hediyesiz eğitip karşılaştırma |
| **Ana çıktı** | Kategori bazlı kontaminasyon oranı + "kontaminasyon yarı ömrü" + retargeting israf oranı |
| **Donanım** | RTX 3060/4060 veya Kaggle free tier |
| **Maliyet** | $0 |
| **Süre** | 8 hafta |
| **Ekip** | 3 kulvar (bkz. §11) |

---

## 1. Problem Tanımı

### 1.1 Teknik tanım

Öneri sistemleri **implicit feedback** ile eğitilir: bir kullanıcının bir ürünle etkileşimi (tıklama, satın alma) o ürüne yönelik pozitif tercih sinyali sayılır. Model bu sinyallerden kullanıcı temsili öğrenir.

Bu varsayım bir alım sınıfında **sistematik olarak yanlıştır**: kullanıcı ürünü kendisi için değil, başkası için almıştır. Hediye alımı, alıcının tercih uzayında hiç bulunmayan bir noktayı pozitif etiketle işaretler. Model bunu tercih olarak öğrenir ve sonraki önerileri buna göre kaydırır.

Kontaminasyonun iki özelliği onu diğer gürültü türlerinden ayırıyor:

1. **Rastgele değil, yapısal.** Belirli kategorilerde (oyuncak, takı, kitap) ve belirli zamanlarda (Kasım-Aralık, Şubat, Mayıs) yoğunlaşıyor. Rastgele gürültü ortalamada sönümlenir; yapısal gürültü sönümlenmez.
2. **Metinden tespit edilebilir.** Kullanıcı review'da genellikle açıkça söylüyor. Bu, diğer gürültü sınıflarından (yanlış tıklama, pozisyon bias'ı) farkı — onlar gözlemlenemez, bu gözlemlenebilir.

### 1.2 Pazarlama ekonomisi

Kontaminasyon rahatsızlık değil, ölçülebilir kayıp:

**Slot maliyeti.** Öneri yüzeyinde sabit sayıda slot var (ana sayfa ~20, e-posta ~6, retargeting reklamı 1-3). Kirli bir slot, doldurulabilecek başka bir üründen çalınmış demektir. Fırsat maliyeti gösterim başına.

**Retargeting israfı.** Reklamverenin ödediği, platformun tahsil ettiği gösterim, hiç var olmamış bir tercihi hedefliyor. Bu doğrudan nakit kaybıdır ve iki tarafta birden.

**Az veri = orantısız bozulma.** Beş etkileşimli bir profilde tek hediye, profilin %20'sini yanlış yönlendirir. Amazon patenti bu noktayı özellikle vurguluyor.

### 1.3 Neden şimdi yapılabiliyor

Bilgi review metninde her zaman vardı. Çıkarmak mümkün değildi.

Anahtar kelime araması yetersiz — iki yönde birden hata yapıyor:
- **Yanlış pozitif:** "Great gift idea!" yazan kişi ürünü kendine almış, sadece hediyeye de uygun olduğunu söylüyor.
- **Yanlış negatif:** "Aldım, kızım bayıldı" cümlesinde `gift` kelimesi geçmiyor ama net bir hediye.

Ayrım **anlam** gerektiriyor, kelime eşleşmesi değil. Milyonlarca review'da bunu yapmak için ya binlerce insan-saat gerekiyordu ya da hiç. Tüketici GPU'sunda çalışan açık ağırlıklı modeller bu boşluğu yeni kapattı — gap'in bugüne kadar açık kalmasının nedeni budur.

---

## 2. Gap Analizi

Bu bölüm dürüstlük gerektiriyor. Yakın işler var ve literature review'da bunları saklamak mümkün değil, saklamamalıyız da.

### 2.1 Mevcut işler

**A) Wang et al., WSDM 2020 — "Time to Shop for Valentine's Day"** (Texas A&M + Etsy)
🔗 [PDF](https://people.engr.tamu.edu/caverlee/pubs/wang20wsdm-valentine.pdf) · [Kod](https://github.com/wangjlgz/Occasion-Aware-Recommenation)

En yakın iş. Sequential modellerin uzun vadeli tercihi yakaladığını, ama doğum günü / yıldönümü / Sevgililer Günü / Anneler Günü gibi occasion'ların bu tercihten sapmalara yol açtığını söylüyor ve sapmayı modelleyen bir sistem öneriyorlar.

| Boyut | Wang et al. | Bizim proje |
|---|---|---|
| **Amaç** | Occasion'ı modelleyerek tahmini iyileştir | Occasion'ı gürültü sayıp temizle |
| **Sinyal kaynağı** | Takvim / zaman | Review metni (semantik) |
| **Kaçırdığı** | Mart'taki yeğen doğum günü hediyesi | — |
| **Veri** | Etsy özel verisi (replike edilemez) | Amazon Reviews 2023 (açık) |

**B) Amazon patentleri** — US 9818145, 10445809, 8352331, 11367117

Problem sanayide açıkça tanınmış. Patentlerden biri: alıcı hediye paketi veya hediye notu talep ederse ürün davranış analizinden çıkarılabilir, ancak çoğu durumda satıcı bunun hediye olduğunu belirleyemez ve kullanıcı profilinde bozulmalar oluşur; az veri noktası varken her hediye alımının bozucu etkisi orantısız güçlüdür.

**Bu bizim lehimize:**
- Problemin iş değeri patentle kanıtlanmış → "bu önemli mi ki" sorusu kapanıyor
- Patentteki çözüm **açık sinyale** (hediye paketi kutucuğu) dayanıyor → kullanıcı işaretlemezse çaresiz
- Akademik literatürde **hiç nicelleştirilmemiş** → sayı veren ilk çalışma biz olacağız

**C) Denoising implicit feedback literatürü** — büyük ve olgun.
Gürültüyü yanlış tıklama, pozisyon bias'ı, memnuniyetsizlik, iade olarak tanımlıyor. **"Bu alım başka biri içindi"** diye ayrı bir gürültü sınıfı yok.

**D) Gift recommendation literatürü** — ters problem (hediye alacak kişiye öneri üretmek).

**E) Zhao et al., WWW 2020 "Target Customer Distortion"** — isim benziyor, problem farklı (bir ürünün hangi müşteri kitlesine önerildiğinin kalibrasyonu). İlgisiz.

### 2.2 Geriye kalan gap

| # | Boşluk | Neden boş kalmış |
|---|---|---|
| **G1** | Hediye alımlarının kamuya açık veride yaygınlığı ölçülmemiş | Etsy verisi özel; Amazon verisinde hediye etiketi yok; LLM öncesi metinden çıkarmak pratik değildi |
| **G2** | Kontaminasyonun öneri kalitesine etkisi ölçülmemiş | Kimse "hediyeleri çıkarınca ne oluyor" deneyini yapmamış |
| **G3** | Silmek mi modellemek mi belirsiz | Wang et al. modelliyor; temizlemenin ne zaman üstün olduğu bilinmiyor |

**Capstone iddiamız:** *"Yeni algoritma icat ettik"* değil. *"Bilinen ama hiç ölçülmemiş bir problemi açık veride ölçtük ve müdahalenin etkisini test ettik."* Bir capstone için doğru büyüklükte ve savunulabilir bir iddia.

---

## 3. Araştırma Soruları

**RQ1.** Amazon review'larında hediye/başkası-için alımların oranı nedir; kategoriye ve mevsime göre nasıl değişir?

**RQ2.** Hediye etkileşimlerini eğitimden çıkarmak (veya ağırlığını düşürmek), kullanıcının **kendi** sonraki alımını tahmin etme başarısını artırır mı? Etki büyüklüğü kategorinin hediye yoğunluğuyla orantılı mı?

**RQ3.** Hediye sinyalini **silmek** mi, modele **ayrı sinyal olarak vermek** mi daha iyi? (Wang et al. yaklaşımına doğrudan test)

**RQ4.** Bir hediye alımından sonra öneri listesi ne kadar süre kirli kalıyor?

> **Çerçeveleme notu:** RQ2'yi "ne kadar iyileşir" diye değil "duyarlı mı" diye soruyoruz. Bunun nedeni §13'te açıklanıyor ve kritik.

---

## 4. Veri

### 4.1 Kaynak

**Amazon Reviews 2023** (McAuley Lab, UCSD) — 571.54M review, Mayıs 1996 – Eylül 2023, 33 kategori.
🔗 [Ana sayfa](https://amazon-reviews-2023.github.io/) · [HuggingFace](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) · [Paper](https://arxiv.org/abs/2403.03952)

Kategori bazlı `.jsonl.gz` dosyaları olarak ayrı ayrı indirilebiliyor — tüm 571M'i indirmek gerekmiyor.

### 4.2 Kullanacağımız alanlar

| Alan | Tip | Neden gerekli |
|---|---|---|
| `text` | str | Hediye tespitinin tek kaynağı |
| `title` | str | Kısa ama sinyal yoğun |
| `rating` | float | Hediye vs kendi alımı rating dağılımı karşılaştırması (V4) |
| `timestamp` | int (unix) | **Mevsimsellik doğrulaması buna bağlı** |
| `user_id` | str | Sekans kurma |
| `parent_asin` | str | Metadata join anahtarı — `asin` değil! |
| `verified_purchase` | bool | Kalite filtresi |

Metadata tarafında: `title`, `categories`, `price`, `main_category`.

### 4.3 Kategori seçimi — üçlü tasarım

Projenin en önemli tasarım kararı. Tek kategori yerine **hediye yoğunluğu spektrumunda üç kategori**:

| Rol | Kategori | #User | #Item | #Rating | Beklenen hediye oranı |
|---|---|---|---|---|---|
| **Yüksek** | `Toys_and_Games` | 8.1M | 890.7K | 16.3M | Yüksek — oyuncakların çoğu çocuğa alınır |
| **Orta** | `Video_Games` | 2.8M | 137.2K | 4.6M | Orta — yoğun item, mevsimsel hediye |
| **Düşük (kontrol)** | `Grocery_and_Gourmet_Food` | 7.0M | 603.2K | 14.3M | Düşük — insanlar yiyeceği kendine alır |

**Neden kritik:** Etki büyüklüğü hediye oranıyla orantılı çıkmalı — oyuncakta büyük, markette ≈ sıfır. Markette de oyuncaktaki kadar iyileşme görürsek, detektörümüz hediye değil başka bir şey yakalıyor demektir.

**Bu, projenin kendi kendini yanlışlama mekanizmasıdır.** Bilimsel bir çalışmayı öğrenci ödevinden ayıran şey budur: hatasını kendi yakalayabilecek şekilde kurulmuş olmak.

**Pilot kategori:** `All_Beauty` (632K user / 701.5K rating) veya `Handmade_Products` (586.6K user / 664.2K rating). Küçük, tüm pipeline'ı 1 günde uçtan uca test eder. Handmade özellikle ilginç — Etsy benzeri, hediye yoğun.

### 4.4 Ön işleme

1. `verified_purchase == True` filtresi
2. Boş / çok kısa metin at (< 5 kelime)
3. **5-core filtreleme** (≥5 etkileşimli user ve item) — recsys standardı; McAuley Lab hazır split sağlıyor
4. Kullanıcı başına kronolojik sekans kur
5. **Zaman bazlı split** — rastgele değil, sızıntı olur

---

## 5. Pipeline Mimarisi

```
Amazon Reviews 2023 (kategori bazlı .jsonl.gz)
        │
        ├─► [A] Katmanlı örnekleme: 40-60K review
        │        ↓
        │   [B] LLM annotator (Qwen3.5-9B, vLLM, JSON çıktı)
        │        ↓  etiket + gerekçe + alıcı ilişkisi + occasion
        │   [C] İnsan doğrulama seti (500 review, 3 annotator, kappa)
        │        ↓
        │   [D] ModernBERT-base distillation
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

## 6. Aşama Detayları

### [A] Örnekleme stratejisi

Rastgele örnekleme yapmayın — hediye sınıfı azınlıkta ve zamana bağlı.

**Katmanlı örnekleme:**
- Ay bazında dengeli (12 ay eşit) → mevsimsellik analizini bozmamak için
- Rating bazında dengeli
- Metin uzunluğu bazında dengeli
- Ek olarak anahtar kelime içeren bir alt küme ayrıca örneklenmeli — **ama analizde ayrı tutulmalı**, yoksa oranı şişirir

**Hedef:** 40-60K review. ModernBERT'i eğitmeye fazlasıyla yeter.

### [B] LLM annotator — model seçimi

Ağustos 2026 itibarıyla tüketici GPU'sunda çalışan seçenekler:

| Model | Q4 VRAM | Lisans | Değerlendirme |
|---|---|---|---|
| **Qwen3.5-9B** | ~6-7 GB | Apache 2.0 | **Birincil.** Structured output için eğitilmiş, temiz lisans |
| **Gemma 4 12B** | ~8 GB | Apache 2.0 | **İkincil.** 12GB kartta rahat; model-arası uyum için |
| Qwen3.5-4B | ~3 GB | Apache 2.0 | Hız gerekiyorsa; sınıflandırma için muhtemelen yeterli |
| gpt-oss-20b | ~16 GB | Apache 2.0 | 16GB+ kart varsa; muhakeme gücü bu iş için fazla |

**Karar: Qwen3.5-9B birincil, Gemma 4 12B ikincil.** İkisini birden koşturup **model-arası uyum** raporlamak Technology Review'un merkezine oturur.

**Servis: vLLM.** Bu iş offline batch olduğu için continuous batching şart — Ollama'ya göre 5-10× hız.

### [B2] Etiket şeması — ikili değil, dörtlü

```json
{
  "purchase_type": "self | gift_given | household | unclear",
  "confidence": "high | medium | low",
  "recipient": "child | spouse | parent | friend | colleague | unknown | null",
  "occasion": "birthday | christmas | wedding | graduation | none | unknown",
  "evidence_span": "kızımın doğum günü için aldım"
}
```

**`household` neden ayrı:** Bebeğine bez almak hediye değil, ama alıcının kendi tercihini de yansıtmıyor. Bu ara kategori hem gerçekçi hem de RQ3 için ilginç — üçünü farklı ağırlıklandırmayı test edebiliriz.

**`evidence_span` neden zorunlu:** LLM'i gerekçe göstermeye zorlamak halüsinasyonu azaltır ve insan doğrulamasını ~10× hızlandırır — annotator tüm review'u okumak yerine span'e bakar.

**Structured output:** vLLM guided decoding / JSON schema. Serbest metin parse etmeyin.

### [C] İnsan doğrulaması — pazarlık edilemez

- **500 review**, LLM etiketlerine göre katmanlı (her sınıftan yeterli örnek)
- **3 annotator** (ekip üyeleri), bağımsız etiketleme
- **Fleiss' kappa** raporla — κ < 0.6 ise görev tanımı muğlaktır, şemayı düzelt
- LLM'in her sınıf için precision / recall / F1
- **Hata analizi:** LLM en çok nerede yanılıyor? (Muhtemelen: "arkadaşım tavsiye etti" ≠ hediye; "eşim için aldım ama ikimiz de kullanıyoruz" = household mu gift mi)

### [D] Distillation — ModernBERT

40-60K LLM etiketiyle bir **ModernBERT-base** sınıflandırıcı eğit.

**Neden gerekli:** Toys_and_Games'te 5-core sonrası milyonlarca review var. 9B modelle hepsini işlemek günler sürer.

**Neden ModernBERT (DeBERTa/RoBERTa değil):** 8192 token context (uzun review kesilmiyor), modern eğitim, `AutoModelForSequenceClassification` ile doğrudan çalışıyor, tüketici GPU'sunda dakikalar içinde fine-tune ediliyor.

**Doğrulama:** ModernBERT'i insan etiketli 500'lük sette de test et. LLM'e yakınsa distillation başarılı.

### [E] Tam korpus inference

ModernBERT ile. Çıktı: her etkileşim için `purchase_type` + olasılık skoru. **Olasılığı sakla** — soft weighting için lazım.

---

## 7. Deneysel Tasarım

### 7.1 Görev

Sequential recommendation, leave-one-out: kullanıcının kronolojik sekansındaki son etkileşimi tahmin et.

### 7.2 ⚠️ Kritik kural: test item'ı hediye OLMAMALI

İddia "hediye alımları kullanıcının kendi tercihine dair tahmini bozuyor". Test item'ı da hediyeyse ölçtüğünüz şey bu değil.

**Sadece son etkileşimi `self` olarak sınıflanmış kullanıcılar üzerinde değerlendirin.**

### 7.3 Beş koşul

| Kod | Koşul | Ne yapıyor |
|---|---|---|
| **C0** | Baseline | Tüm etkileşimlerle eğit (standart uygulama) |
| **C1** | Hard removal | `gift_given` etkileşimlerini eğitimden çıkar |
| **C2** | Soft down-weighting | Hediye etkileşimlerini loss'ta *w* ∈ {0.25, 0.5, 0.75} ile ağırlıklandır |
| **C3** | Modelleme | Hediye bayrağını feature/token olarak **ekle**, silme (Wang et al. yaklaşımı) |
| **C4** | ★ **PLASEBO** | Hediye sayısı kadar **rastgele** etkileşim çıkar |

**C4 olmadan bu proje geçersizdir.** C1 iyileşme gösterirse jürideki ilk akıllı kişi soracak: *"Rastgele bir grup çıkarsaydınız da artmaz mıydı?"* C4 bu soruyu kapatır. C1 > C4 ise etki gerçek; C1 ≈ C4 ise bulgu gürültüdür ve bunu dürüstçe yazarız.

**İkinci plasebo (opsiyonel, güçlü):** Etiketleri kullanıcılar arasında shuffle edip aynı sayıda etkileşim çıkar. Hediye *dağılımının* mı yoksa *kimliğinin* mi önemli olduğunu ayırır.

### 7.4 Modeller

| Model | Rol | Neden |
|---|---|---|
| **Popularity** | Alt sınır | Her recsys paper'ının ilk baseline'ı |
| **ItemKNN** | Klasik CF | Basit, yorumlanabilir, sürpriz derecede güçlü |
| **BPR-MF** | Matrix factorization | Sekans dışı klasik referans |
| **SASRec** | **Birincil** | Self-attention sequential; hediye "sekans içi kesinti" olduğu için doğru mimari |
| **GRU4Rec** | Sekans alternatifi | Bulgunun mimariye özgü olmadığını göstermek için |

**Kütüphane: RecBole.** Beşi de hazır implement, tek konfigürasyon, aynı değerlendirme protokolü. Kendimiz yazarsak haftalarca hata ayıklarız ve sonuçlarımız literatürle kıyaslanamaz.

**Koşu sayısı:** 5 koşul × 5 model × 3 kategori = 75 run. Fazla gelirse: SASRec + BPR-MF × 5 koşul × 3 kategori = 30 run.

---

## 8. Doğrulama Stratejisi

Bu bölüm projeyi öğrenci ödevinden ayıran şey.

**V1 — İnsan etiketi (iç geçerlilik).** 500 review, 3 annotator, kappa + F1. Detay §6[C].

**V2 — ★ Mevsimsellik (dış geçerlilik, etiketsiz).**
Aylık hediye oranını çiz. Beklenen: **Kasım-Aralık patlaması**, **Şubat tepesi** (Sevgililer), **Mayıs tepesi** (Anneler Günü).

Detektör takvimi tanımıyor, sadece metin okuyor. Yine de bu deseni üretiyorsa, **hiç insan etiketi olmadan** çalıştığının kanıtıdır. Tek grafik, çok güçlü argüman.

⚠️ Review tarihi ≠ satın alma tarihi. İnsanlar aldıktan haftalar sonra yorum yazar; tepeler gecikmeli olacak. **Bunu önceden söyleyin**, sonradan savunma pozisyonuna düşmeyin.

**V3 — Kategori yüz geçerliliği.** Toys > Video Games > Grocery sıralaması çıkmalı.

**V4 — Rating dağılımı.** Hediye alımlarının rating dağılımı farklı olmalı (muhtemelen daha yüksek ortalama, daha düşük varyans — ürünü kullanmayan kişi eleştiremez). Fark yoksa şüphelen.

**V5 — Prompt ve model duyarlılığı.** 2 prompt × 2 model = 4 kombinasyon. Hediye oranı %12'den %30'a fırlıyorsa bu bir limitasyondur ve **raporlanır**. Gizlemeyin — dürüst raporlama sahte kesinlikten iyidir.

---

## 9. Değerlendirme Metrikleri

### 9.1 Teknik katman

- **Recall@10, NDCG@10, HR@10** — non-gift test item'larında
- Bootstrap güven aralığı (tek sayı değil aralık)
- 3-5 farklı random seed (recsys sonuçları seed'e duyarlıdır)

### 9.2 ★ Pazarlama katmanı

Capstone bizden pazarlamacının kullanabileceği bir karar istiyor. Teknik metrikler iç katman; bunlar dış katman.

**M1 — Retargeting israf oranı.** Kullanıcının yalnızca hediye alımı üzerinden girdiği kategorilerden gelen öneri slotlarının yüzdesi. Doğrudan "boşa giden gösterim".

**M2 — Kontaminasyon yarı ömrü.** *(en iyi grafik)* Bir hediye alımından sonra sonraki önerilerin yüzde kaçı hediyenin kategorisinden geliyor — zamana karşı çiz. *"Tek bir hediye alımı öneri listenizi ortalama X hafta bozuyor"* cümlesi tek slaytta geçer.

**M3 — Segment analizi.** Az etkileşimli kullanıcılarda etki daha büyük olmalı — Amazon patenti de bunu söylüyor. Doğrularsak hem bulgu hem de sanayi kaynağıyla örtüşme kanıtı.

---

## 10. Teknoloji Yığını

| Katman | Seçim | Alternatifler | Gerekçe |
|---|---|---|---|
| Veri yükleme | `datasets` (HF) + `polars` | pandas | Polars 10M+ satırda çok daha hızlı ve bellek dostu |
| LLM servis | **vLLM** | Ollama, llama.cpp, TGI | Offline batch için continuous batching şart |
| Annotator LLM | **Qwen3.5-9B** (+ Gemma 4 12B) | Phi-4, Mistral Small | Apache 2.0, structured output, tüketici GPU |
| Structured output | vLLM guided decoding / `outlines` | regex parse | JSON şeması garantili, parse hatası sıfır |
| Distillation hedefi | **ModernBERT-base** | DeBERTa-v3, RoBERTa | 8192 context, hızlı fine-tune |
| Eğitim | HF `transformers` + `accelerate` | Lightning | Standart, ekip biliyor |
| Recsys | **RecBole** | Cornac, implicit, RecPack | 100+ algoritma, tek protokol, literatürle kıyaslanabilir |
| Deney takibi | **Weights & Biases** (free) | MLflow, TensorBoard | 75 run'ı takip etmenin tek makul yolu |
| Annotation UI | **Argilla** veya Streamlit | Label Studio | 500 review için Streamlit yeter |
| İstatistik | `scipy`, `statsmodels` | — | Bootstrap CI, anlamlılık testleri |
| Görselleştirme | `matplotlib` + `seaborn` | plotly | Yayın kalitesi grafik |
| Versiyon | Git + GitHub | DVC | Submission zaten GitHub'a |

**Compute:** RTX 3060/4060 (8-12GB) yeter. Kaggle free tier (2× T4, haftada ~30 saat) yedek. Colab free (T4) pilot için.
**Toplam maliyet: $0.**

---

## 11. İş Bölümü

Üç kulvar, dengeli yük. Kim hangisini alacak konuşulacak.

### Kulvar A — Veri ve Tespit
En çok mühendislik burada.
- Amazon Reviews 2023 indirme, ön işleme, 5-core filtreleme
- Katmanlı örnekleme kodu
- Etiket şeması tasarımı ve prompt geliştirme
- vLLM kurulumu, LLM annotation koşusu
- ModernBERT distillation ve tam korpus inference
- **Sahiplendiği bölüm:** Technology Review'un LLM/NLP kısmı

### Kulvar B — Doğrulama ve Analiz
- İnsan etiketleme sürecinin yürütülmesi ve koordinasyonu
- Fleiss' kappa, precision/recall/F1 hesapları
- Mevsimsellik analizi (V2) ve grafikleri
- Kategori karşılaştırması (V3), rating dağılımı (V4)
- Prompt/model duyarlılık analizi (V5)
- **Sahiplendiği bölüm:** Data Research'ün ana yazarı

### Kulvar C — Öneri Sistemi ve Deney
- RecBole kurulumu ve konfigürasyonu
- 5 koşulun kurulması ve koşturulması
- Bootstrap CI, seed tekrarları, anlamlılık testleri
- Pazarlama metrikleri (M1-M3) hesabı
- **Sahiplendiği bölüm:** Technology Review'un recsys kısmı + sonuç tabloları

### Ortak iş
- **500 review'luk insan etiketleme** — üçümüzün de yapması gereken tek şey. Uyum istatistiği için en az 3 bağımsız etiketleyici lazım. Bir öğleden sonra sürer.
- Literature review — beş temaya bölünüp paylaşılır (§14)

---

## 12. Zaman Planı (8 hafta)

| Hafta | İş | Çıktı |
|---|---|---|
| **1** | Literatür taraması. Pilot kategori indir, EDA, naif keyword taraması. | Lit review taslağı + **karar: kategori doğru mu?** |
| **2** | Etiket şeması. Prompt geliştirme. 200 review'da manuel deneme, şemayı revize et. | Kararlı annotation şeması |
| **3** | vLLM kurulum. Pilot kategoride 10K annotation. İlk mevsimsellik grafiği. | **Karar noktası: detektör çalışıyor mu?** |
| **4** | 500 review insan etiketleme (3 kişi). Kappa + F1. Duyarlılık analizi. | Doğrulanmış detektör + hata analizi |
| **5** | 3 kategoride tam annotation (40-60K). ModernBERT distillation. Tam inference. | Etiketli tam veri seti |
| **6** | RecBole kurulumu. C0 + C4 (baseline + plasebo) tüm kategorilerde. | **Karar noktası: etki var mı?** |
| **7** | C1, C2, C3. Bootstrap CI, seed tekrarları. | Tam sonuç tablosu |
| **8** | Pazarlama metrikleri. Üç bölümün yazımı. Grafikler. | Teslim |

**İki karar noktası kritik:**
- **Hafta 3 sonu:** Detektör mevsimsellik göstermiyorsa prompt/model değiştir — devam etme.
- **Hafta 6 sonu:** C0 ile C4 arasında anlamlı fark varsa (olmamalı), deney kurulumunda hata var demektir.

---

## 13. Riskler

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| **C1 ile C0 arasında fark yok** | **Yüksek** | Orta | **Başarısızlık değil.** Bkz. aşağıdaki not. |
| Hediye oranı çok düşük (%2-3) | Orta | Yüksek | Hediye yoğun kategoriye kay. Düşük oran da bulgudur: "review yazanlar ağırlıkla kendi alımlarını yorumluyor" |
| Seçim yanlılığı (sadece review'lu alımlar) | **Kesin** | Orta | Limitations'da açıkça yaz. İddiayı "review'lu etkileşimler arasında" diye sınırla. Data Research'ü **güçlendirir**. |
| LLM etiketleri tutarsız | Orta | Yüksek | V5 duyarlılık analizi planda; `evidence_span` zorunluluğu halüsinasyonu kısar |
| Review tarihi ≠ alım tarihi | Kesin | Düşük | Önceden söyle. Tepe konumu değil, tepenin *varlığı* önemli. |
| Kapsam şişmesi (3×5×5) | Yüksek | Orta | Öncelik: Toys + Grocery, SASRec + BPR-MF, C0/C1/C4. Gerisi "nice to have". |
| `household` sınıfı muğlak, kappa düşük | Orta | Orta | κ < 0.6 ise `household`'ı `unclear` ile birleştir, üçlü şemaya in |

### ⚠️ En olası senaryo ve neden sorun değil

**Hediyeleri çıkarmanın anlamlı fark yaratmaması yüksek ihtimal.** Modern sequential modeller gürültüye dayanıklı olabilir.

Bu bir başarısızlık değil. *"Kontaminasyon oranı %18, ama SASRec bu gürültüye dayanıklı çıktı"* tamamen geçerli ve raporlanabilir bir bulgudur — hatta ilginçtir, çünkü Amazon patenti aksini varsayıyor.

**Bu yüzden RQ2'yi "ne kadar iyileşir" diye değil "duyarlı mı" diye sorduk.** İkinci soru hangi cevap gelirse gelsin yanıtlanmış olur; birincisi sadece tek bir cevapla yanıtlanır. Bu çerçeveleme farkı, altıncı haftadaki moral çöküşünü baştan engelliyor.

---

## 14. Capstone Bölümleriyle Eşleme

### Literature Review — beş tema
1. **Occasion/context-aware recommendation** — Wang et al. WSDM 2020 (merkez), hypergraph temporal intent modeling (WWW 2025)
2. **Denoising implicit feedback** — "Learning Robust Recommender from Noisy Implicit Feedback" (TKDE), LLM-enhanced hard sample identification (2024), natural noise literatürü
3. **Sequential recommendation** — SASRec, BERT4Rec, GRU4Rec, Mamba4Rec
4. **LLM as annotator / weak supervision** — LLM tabanlı veri etiketleme, distillation
5. **Endüstri kaynakları** — Amazon patentleri (US 9818145, 10445809), Etsy vakası

**Gap cümlesi:** *"Occasion literatürü sapmayı modelliyor, denoising literatürü gürültüyü siliyor; ama hiçbiri hediye alımını ayrı ve metinden tespit edilebilir bir gürültü sınıfı olarak ele almıyor ve kamuya açık veride nicelleştirmiyor."*

### Data Research
Amazon Reviews 2023 üzerinde: source, access, format, size, time period (1996-2023), granularity (etkileşim seviyesi), variables (§4.2), quality (verified_purchase, boş metin), **imbalance** (hediye sınıfı azınlık), **bias** (seçim yanlılığı), **privacy** (anonim user_id ama serbest metinde isim geçebilir → anonimleştirme tartışması), limitations. Artı: 3 kategorinin karşılaştırmalı EDA'sı ve mevsimsellik grafiği.

### Technology Review — beş karşılaştırma ekseni
1. **Annotator LLM'leri:** Qwen3.5-9B vs Gemma 4 12B — uyum, hız, VRAM, lisans
2. **LLM vs distilled encoder:** Qwen3.5-9B vs ModernBERT — doğruluk/hız/maliyet
3. **Recsys mimarileri:** ItemKNN vs BPR-MF vs SASRec vs GRU4Rec — gürültü dayanıklılığı farklı mı?
4. **Müdahale stratejileri:** hard removal vs soft weighting vs feature-as-signal
5. **Servis altyapısı:** vLLM vs Ollama throughput

Limitations için hazır malzeme: LLM halüsinasyonu, prompt duyarlılığı, distillation kaybı, privacy, açık modellerin çok dilli performans farkı.

---

## 15. Ön Gereksinimler

**Şart:** Python, pandas (veya polars öğrenmeye açıklık), temel git.

**Faydalı ama şart değil:** PyTorch, HuggingFace ekosistemi, temel öneri sistemi bilgisi.

**Kimsenin bilmesi gerekmiyor:** RecBole, vLLM, ModernBERT. Üçünün de dokümantasyonu iyi ve yol boyunca öğrenilir. Hiçbiri araştırma seviyesi zorluk içermiyor.

**Donanım:** En az bir kişide RTX 3060 sınıfı GPU olması işi çok hızlandırır. Yoksa Kaggle free tier'la da yürür, sadece koşular gece bırakılır.

---

## 16. İlk Hafta — Somut Başlangıç

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

**Bu 10 satırın amacı LLM değil, karar.**

- Oran %1'in altındaysa ve örnekler alakasızsa → kategori seçimini gözden geçir
- Oran %5-15 arasıysa ve örnekler gerçekten hediyeyse → LLM'e geç

LLM'e geçmenin nedeni: naif taramanın kaçırdıklarını ("aldım ve kızım bayıldı") yakalayacak ve yanlış pozitifleri ("harika bir hediye olur" — ürün açıklamasından etkilenmiş yorum) eleyecek.

Ardından ay bazlı oranı çiz. Aralık'ta tepe görüyorsan proje çalışıyor demektir.

---

## Ekip için iki soru

1. Hangi kulvarda çalışmak istersiniz? (§11)
2. Bu tasarımda gözden kaçırdığım bir delik var mı? Özellikle §7.3'teki koşul setine ve §8'deki doğrulama zincirine bakın.
