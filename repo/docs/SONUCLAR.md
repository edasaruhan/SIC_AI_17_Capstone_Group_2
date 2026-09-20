# Sonuçlar — Hediye Kontaminasyonu

> **Tarih:** 2026-09-19 · **Deney:** 72/72 koşu (Kaggle, 2× T4), kod `e1c7f89` (67 koşu) ve
> `f3183df` (5 koşu; `src/`, `scripts/`, `configs/` iki sürümde aynı) · **Analiz:** `f3183df`.
>
> Buradaki her sayı `reports/results/` altındaki bir JSON'dan geliyor; hangisinden geldiği
> her bölümde yazıyor. Eşikler, karşıtlıklar, yorum kuralları ve pazarlama metriklerinin
> tanımları **deney koşulmadan** yazıldı (DECISIONS 2026-09-14, commit `8c697a8` ve
> sonrası). Sonuç görüldükten sonra eklenenler ikidir, ikisi de yerinde öyle işaretli:
> valid satırındaki hediyelerin sayımı (§6.1, §7 madde 5) ve M2'nin n = 0 kovası hariç
> duyarlılığı (§6.1). Hiçbiri birincil bir sayının yerine geçmiyor.

---

## Kısa cevap

| Soru | Cevap |
|---|---|
| **RQ1** — hediye ne kadar yaygın? | Toys'ta review'ların **%19,0**'u [15,8–22,3] hediye (insan kalibrasyonlu; LLM'in ham oranı %23,25). Grocery, All Beauty ve Video Games'te %5–8. Dört kategoride de Aralık–Ocak tepesi var (1,58–2,08×). |
| **RQ2** — hediyeyi çıkarmak kendi sonraki alımın tahminini iyileştirir mi? | **Hediye satırları, aynı sayıda rastgele satırdan belirgin daha az işe yarıyor** (C1 > C4): Toys'ta SASRec +%16, BPR +%40 (Recall@10). Grocery'de etki çok küçük ya da saptanamıyor → **doz–yanıt tutuyor**. Ama hediyeyi silmenin **hiç silmemeye** göre net kazancı modele bağlı: Toys'ta BPR **+%4** iyileşiyor, SASRec **−%7** kötüleşiyor. |
| **RQ3** — silmek mi, "hediyeydi" diye işaretlemek mi? | Bu çalışmadaki işaretleme yöntemi (gölge token) **her iki seçenekten de kötü** ya da onlardan ayırt edilemiyor: C3, C1'den 4 hücrenin 3'ünde, C0'dan 3'ünde anlamlı düşük. |
| **RQ4** — öneri listesi ne kadar süre kirli kalıyor? | Sıralı modelde (SASRec) Toys'ta kısa: fazla pay **2,0 puan**la başlıyor, **~1 kendi alımında yarıya iniyor** (≈2–4 hafta, yaklaşık). Grocery'de küçük (1,5 puan) ama yavaş sönüyor. Sırayı görmeyen BPR'de önceden kayıtlı ölçüyle sınırlı bir yarı ömür çıkmıyor; fazla pay yeni alımlarla azalıyor gibi görünüyor. n = 0 kovası yapısal olarak farklı (§6.1) — duyarlılık analiziyle sayılar değişiyor, Toys SASRec'in "~1 kendi alımı" cevabı değişmiyor. |

Deneyin kurulum kapısı (Kapı 2) dört kategori × model hücresinin **dördünde de PASS** —
sonuçlar yorumlanabilir.

---

## 1. Ne ölçüldü, nasıl

**Veri.** Amazon Reviews 2023. Yaygınlık (RQ1) dört kategorinin `clean` korpusundan
(kategori başına 10.000 satırlık orantılı `main` örneklemi); deney (RQ2–RQ4) Toys_and_Games
("yüksek hediye") ve Grocery_and_Gourmet_Food ("düşük hediye", kontrol) 5-core
korpuslarında. All Beauty'nin 5-core'u boş; Video Games deneye alınmadı.

**Etiket zinciri.** Qwen3-4B (vLLM, prompt v3) 47.200 satırı etiketledi → tek bir insan
etiketleyici (A) 500 satırı körleme etiketledi (Hafta 4) → ModernBERT-base öğrencisi LLM
etiketleriyle eğitildi ve sadakat kapısını geçti (öğretmene karşı C1 F1 0,910, C1b F1 0,931;
insana karşı C1 F1 0,742, öğretmen 0,731) → öğrenci deney korpusunun **4.598.612** satırının
tamamını etiketledi.

**İki hediye tanımı.** **C1** = `gift_given` (birincil). **C1b** = `gift_given` +
`household` + `received` ("alıcı ürünü kendisi seçmedi", sağlamlık). Hafta 4'ün insan
referansına göre popülasyondaki etiket kalitesi: C1 kesinlik **0,65** · duyarlılık 0,80;
C1b kesinlik 0,84 · duyarlılık 0,74 (`experiment_stats.json` → `label_quality_population`).

**Deney.** Zaman bazlı leave-one-out; test ürünü her zaman `self` (kullanıcının **kendi**
sonraki alımı). Bölme ve kullanıcı/ürün evreni C0'da donduruldu; koşullar yalnızca
**eğitim** satırlarını değiştiriyor.

| Koşul | Eğitimde ne değişiyor | Toys | Grocery |
|---|---|---:|---:|
| C0 | hiçbir şey | 1.626.714 satır | 1.896.612 satır |
| C1 | `gift_given` satırları silinir | −401.286 (%24,7) | −58.209 (%3,1) |
| C4 | C1 kadar **rastgele** satır silinir (plasebo) | −401.286 | −58.209 |
| C1b | C1b etiketli satırlar silinir | −859.194 (%52,8) | −143.789 (%7,6) |
| C4b | C1b kadar rastgele satır silinir (plasebo) | −859.194 | −143.789 |
| C3 | `gift_given` satırının ürünü ayrı bir "gölge" kimliğe (`<id>::gift`) taşınır; gölge ürün önerilemez | 73.683 gölge ürün | 25.031 gölge ürün |

(Satır sayıları BPR'ninki; `condition_*.json`.) Modeller SASRec (sıralı, birincil) ve BPR;
RecBole 1.2.0 varsayılan hiperparametreleri, erken durdurma valid NDCG@10, üst sınır 300
epoch. Her hücre üç seed (42, 1337, 2024) → **72 koşu**. Değerlendirme tam sıralama üzerinden
Recall@10 (birincil), NDCG@10, HR@10; @20 de raporda. Test kullanıcısı: Toys **117.386**,
Grocery **238.756**.

**İstatistik.** Kullanıcı başı metrik önce üç seed üzerinden ortalanır; iki koşul aynı
kullanıcılar üzerinde **eşli bootstrap**la karşılaştırılır (1.000 tekrar, %95 yüzdelik GA;
Kapı 2'de 2.000). Önceden kayıtlı yorum kuralı (plasebo karşıtlıkları): GA sıfırın üstündeyse
**"alt sınır"** (etiket gürültüsü farkı plaseboya doğru çeker), sıfırı içeriyorsa
**"saptanamadı"** — "etki yok" yazılmaz.

---

## 2. RQ1 — Hediye alımları ne kadar yaygın?

Kaynak: `prevalence.json`, F19. `clean` korpus, `main` örneklemi (n = 10.000 / kategori).
"Kalibre" = LLM'in her sınıfının insan etiketinde gerçekte neye karşılık geldiğine göre
düzeltilmiş oran (500 satır, tek etiketleyici).

| Kategori | C1 · LLM ham | **C1 · kalibre** [%95 GA] | C1b · ham | **C1b · kalibre** [%95 GA] |
|---|---:|---|---:|---|
| Toys and Games | %23,25 | **%19,0** [15,8–22,3] | %44,0 | **%44,2** [40,4–48,4] |
| Video Games | %7,86 | **%7,9** [5,8–10,6] | %15,2 | **%22,7** [17,8–28,1] |
| Grocery and Gourmet Food | %4,28 | **%5,4** [3,4–8,3] | %9,6 | **%18,3** [13,4–24,1] |
| All Beauty | %4,70 | **%5,5** [3,4–8,4] | %8,8 | **%18,0** [12,8–23,5] |

- **Toys açık ara önde.** Ham LLM oranı (%23,25) kalibre aralığın dışında: model kendi
  çocuğuna alınanı hediye sayıyor. Kalibrasyondan sonra Video Games düşük hediyeli gruptan
  **ayrılamıyor** (GA'lar örtüşüyor).
- **Mevsimsellik** (Aralık+Ocak ÷ Haziran–Eylül, LLM etiketi, `gate1_*.json`): Toys **1,58**
  [1,43–1,73], Video Games **2,08** [1,74–2,49], Grocery **1,71** [1,33–2,23], All Beauty
  **2,06** [1,61–2,63]. Tepe Kasım değil Aralık–Ocak: review tarihi alımın gerisinde kalıyor.
- **Deney korpusunda** (5-core, öğrenci etiketi, ham) hediye payı Toys **%24,8**, Grocery
  **%3,1**; C1b tanımıyla %52,4 / %7,6 (`inference_*.json`).
- C1b'nin düşük hediyeli kategorilerdeki kalibre oranı muhtemelen **yukarı yanlı**:
  kalibrasyon insan–LLM uyumunu kategoriler arasında havuzluyor, bu varsayım Toys'ta tutmuyor
  (§7).

---

## 3. Kapı 2 — Deney kurulumu geçerli mi?

Kaynak: `gate2.json`. Dört ölçüt, eşikler koşulardan önce (`configs/base.yaml` → `gate2`).

| Hücre | 1 · test çiftleri aynı | 2 · evren ⊆ C0 | 3 · C4 − C0 (Recall@10) | 3 · C4b − C0 | 4 · C0 seed DK | **Karar** |
|---|---|---|---|---|---|---|
| Toys · SASRec | ✅ 18 koşu, tek özet | ✅ | −0,851 puan [−0,916, −0,789] | −2,246 [−2,336, −2,163] | 0,012 | **PASS** |
| Toys · BPR | ✅ | ✅ | −0,411 [−0,457, −0,365] | −1,085 [−1,137, −1,032] | 0,025 | **PASS** |
| Grocery · SASRec | ✅ | ✅ | −0,020 [−0,051, +0,010] | −0,164 [−0,196, −0,131] | 0,005 | **PASS** |
| Grocery · BPR | ✅ | ✅ | −0,023 [−0,054, +0,005] | −0,136 [−0,165, −0,107] | 0,002 | **PASS** |

Ölçüt 3'ün koşulu "plasebo C0'ı **geçmiyor**" (GA'nın alt ucu ≤ 0): rastgele veri silmek
doğruluğu düşürmeli, iyileştirirse kurulum hatalıdır. Dört hücrede de plasebo C0'ın altında
ya da ondan ayırt edilemiyor. Seed değişim katsayısı eşiğin (0,10) çok altında.

> **Belgelerdeki eski ifade.** ROADMAP, PROJECT_SPEC, `concept-note` §5 ve
> `implementation-plan` §"Gate 2" bu kapıyı "C0 ile C4 arasında anlamlı fark
> **olmamalı**" diye anlatıyor (dördü de; ilk ikisi 2026-09-18'de, kalan ikisi
> 2026-09-20 denetiminde bulundu). Bağlayıcı olan, koşulardan iki gün önce commit'lenen
> ölçüt yukarıdaki. Toys'ta C4, C0'dan anlamlı düşük — **eski ifadeyle Kapı 2 FAIL olurdu.**
> Ölçüt sonuç görüldükten sonra değiştirilmedi; çelişki DECISIONS 2026-09-18'de ayrıca yazılı.

---

## 4. RQ2 — Hediye satırlarını çıkarmak kendi sonraki alımın tahminini iyileştirir mi?

Kaynak: `experiment_stats.json`, F21. Bütün karşıtlıklar üç seed'li; kullanıcı sayısı Toys
117.386, Grocery 238.756.

### 4.1 Koşulların Recall@10 düzeyi (seed ortalaması, %)

| Hücre | C0 | C1 | C4 | C1b | C4b | C3 |
|---|---:|---:|---:|---:|---:|---:|
| Toys · SASRec | 4,313 | 4,007 | 3,462 | 3,292 | 2,067 | 3,883 |
| Toys · BPR | 1,616 | 1,687 | 1,205 | 1,578 | 0,530 | 1,522 |
| Grocery · SASRec | 3,011 | 3,027 | 2,991 | 2,939 | 2,847 | 2,941 |
| Grocery · BPR | 1,446 | 1,442 | 1,422 | 1,457 | 1,310 | 1,432 |

Bu tablo betimseldir; aşağıdaki karşıtlıklar GA'lıdır. Tabloda karşıtlığı önceden kayıtlı
olmayan bir çift de görünüyor: Toys BPR'de eğitim verisinin **yarısından fazlasını** (C1b)
silmek Recall@10'u 1,616'dan 1,578'e indiriyor; aynı miktarda rastgele silmek (C4b) 0,530'a.

### 4.2 Birincil: C1 − C4 (hediye satırlarını silmek vs aynı sayıda rastgele satırı silmek)

| Hücre | Recall@10 farkı (puan) [%95 GA] | göreli | NDCG@10 farkı [%95 GA] | Yorum (önceden kayıtlı) |
|---|---|---:|---|---|
| Toys · SASRec | **+0,545** [+0,482, +0,610] | +%15,7 | +0,248 [+0,219, +0,278] | alt sınır |
| Toys · BPR | **+0,482** [+0,436, +0,529] | +%40,0 | +0,253 [+0,225, +0,283] | alt sınır |
| Grocery · SASRec | **+0,036** [+0,005, +0,068] | +%1,2 | +0,018 [+0,004, +0,032] | alt sınır |
| Grocery · BPR | **+0,020** [−0,009, +0,049] | +%1,4 | +0,020 [+0,005, +0,035] | Recall: saptanamadı · NDCG: alt sınır |

**Okuma.** Aynı miktarda veri kaybında, kaybedilen satırlar hediye olduğunda model kullanıcının
kendi sonraki alımını belirgin daha iyi tahmin ediyor. Yani hediye satırları, kullanıcının
kendi tercihi hakkında ortalama bir satırdan **daha az** bilgi taşıyor — kontaminasyonun
doğrudan ölçüsü bu. C1 etiketinin kesinliği 0,65 olduğundan silinen satırların bir kısmı
aslında hediye değil; bu da farkı plaseboya doğru çeker. Sayılar bu yüzden gerçek etkinin
**alt sınırı**.

### 4.3 C1 − C0 (hediyeyi silmek vs hiçbir şey silmemek)

| Hücre | Recall@10 farkı [%95 GA] | göreli | yön |
|---|---|---:|---|
| Toys · SASRec | −0,306 [−0,360, −0,247] | −%7,1 | **negatif** |
| Toys · BPR | +0,071 [+0,027, +0,114] | +%4,4 | **pozitif** |
| Grocery · SASRec | +0,016 [−0,012, +0,046] | +%0,5 | sıfırı içeriyor |
| Grocery · BPR | −0,003 [−0,032, +0,026] | −%0,2 | sıfırı içeriyor |

**Okuma.** Pratikteki soru bu: hediyeleri bulup silmek, bugünkü (hepsini kullanan) modelden
daha iyi mi? **Modele bağlı.** BPR'de evet, küçük ama anlamlı. SASRec'te hayır: Toys'ta eğitim
satırlarının dörtte birini silmek, o satırların taşıdığı kirlilikten daha pahalıya geliyor.
Olası sebepler (sınanmadı): sıralı model hediye satırlarından da bir şey öğreniyor (sekans
sürekliliği, ürün birlikteliği, etiket hatası yüzünden aslında `self` olan satırlar), ve C1'de
hediyeler test anındaki girdi sekansından da çıktığı için SASRec daha kısa bir geçmişle tahmin
yapıyor. §4.2 ile birlikte: hediye satırları her iki modelde de ortalama bir satırdan **düşük
değerli**. BPR'de **net zararlı**, SASRec'te hâlâ **net yararlı**.

### 4.4 Sağlamlık: C1b − C4b (geniş tanım)

| Hücre | Recall@10 farkı [%95 GA] | göreli | Yorum |
|---|---|---:|---|
| Toys · SASRec | +1,225 [+1,144, +1,297] | +%59,3 | alt sınır |
| Toys · BPR | +1,047 [+0,995, +1,102] | +%197 | alt sınır |
| Grocery · SASRec | +0,092 [+0,061, +0,123] | +%3,2 | alt sınır |
| Grocery · BPR | +0,147 [+0,118, +0,178] | +%11,2 | alt sınır |

Geniş tanımda yön aynı ve dört hücrenin dördünde saptanabiliyor; Grocery BPR'de de. Göreli
farklar büyük, çünkü C4b Toys'ta eğitimin yarısından fazlasını rastgele siliyor ve taban
(C4b) çok düşüyor. Mutlak farka bakın.

### 4.5 Doz–yanıt: Toys'un etkisi Grocery'ninkinden büyük mü?

Kaynak: `experiment_stats.json` → `dose_response`. İki kategorinin kullanıcı başı farkları
bağımsız bootstrap'la karşılaştırılır.

| Model · karşıtlık | Toys | Grocery | Fark [%95 GA] |
|---|---:|---:|---|
| SASRec · C1 − C4 | +0,545 | +0,036 | **+0,509** [+0,436, +0,581] |
| BPR · C1 − C4 | +0,482 | +0,020 | **+0,462** [+0,409, +0,518] |
| SASRec · C1b − C4b | +1,225 | +0,092 | **+1,133** [+1,060, +1,214] |
| BPR · C1b − C4b | +1,047 | +0,147 | **+0,900** [+0,837, +0,962] |

Hediye payı yüksek kategoride (Toys, C1 %24,7) etki, düşük kategoridekinden (Grocery, %3,1)
dört karşılaştırmada da anlamlı büyük. Bu **iki noktalı** bir doz–yanıt: yön doğru, eğrinin
biçimi hakkında bir şey söylemiyor.

---

## 5. RQ3 — Silmek mi, işaretlemek mi?

Kaynak: `experiment_stats.json`, F21.

| Hücre | C3 − C1 (Recall@10) [%95 GA] | göreli | C3 − C0 [%95 GA] | göreli |
|---|---|---:|---|---:|
| Toys · SASRec | −0,124 [−0,177, −0,078] | −%3,1 | −0,430 [−0,481, −0,373] | −%10,0 |
| Toys · BPR | −0,165 [−0,205, −0,122] | −%9,8 | −0,093 [−0,137, −0,054] | −%5,8 |
| Grocery · SASRec | −0,085 [−0,115, −0,054] | −%2,8 | −0,069 [−0,098, −0,038] | −%2,3 |
| Grocery · BPR | −0,010 [−0,040, +0,018] | −%0,7 | −0,014 [−0,042, +0,014] | −%0,9 |

**Okuma.** Hediyeyi modele "gölge token" olarak vermek hiçbir hücrede silmekten iyi değil;
dört hücrenin üçünde anlamlı kötü, hepsini kullanmaktan da kötü. Bu, **bu işaretleme
yönteminin** sonucu, "sinyal vermek" fikrinin genel sonucu değil. Olası sebep (sınanmadı):
her hediye ürünü için yeni ve seyrek bir kimlik açılıyor (Toys'ta 73.683), bu kimlikler az
veriyle kötü öğreniliyor ve değerlendirmede hiç önerilemiyor. BPR'de kullanıcı vektörünün
bir kısmı önerilemeyecek ürünleri açıklamaya harcanıyor. C2 (hediye satırını kayıpta düşük
ağırlıkla tutmak) uygulanmadı; kapsam dışı.

---

## 6. RQ4 — Pazarlama metrikleri

Kaynak: `marketing_metrics.json`, F22 (M2), F23 (M1). Tanımlar 2026-09-14'te deney
koşulmadan yazıldı. **M2'nin tanımını kullanıcı 2026-09-19'da değerleri görmeden onayladı.**
"Alt kategori" metadata'daki ikinci seviye kategori (Toys'ta 223, Grocery'de 336 değer).
"Hediye-yalnız alt kategori" = kullanıcının geçmişinde yalnızca hediye olarak alışveriş
yaptığı alt kategori.

### 6.1 M2 — Kontaminasyon yarı ömrü (RQ4'ün cevabı)

Son hediyenin alt kategorisi hediye-yalnız olan test kullanıcıları (Toys 29.577, Grocery
11.614). n = o hediyeden sonra kullanıcının kendi (`self`) alım sayısı. Fazla pay = C0'ın
top-10'unun o alt kategoriye verdiği pay − C1'inki. Uyum `A·exp(−λn)`, yarı ömür = ln 2 / λ.
Haftaya çeviri = yarı ömür × test kullanıcılarının ardışık review aralığı medyanı (Toys 25,5
gün, Grocery 63,8 gün) — **yaklaşık**, çünkü review tarihi alım tarihi değil.

| Hücre | Başlangıç fazlası A | **Yarı ömür** (kendi alımı) [%95 GA] | ≈ hafta [GA] |
|---|---:|---|---|
| **Toys · SASRec** (birincil) | 2,0 puan | **0,66** [0,53–0,82] | **≈2,4** [1,9–3,0] |
| **Grocery · SASRec** (birincil) | 1,5 puan | **9,5** [5,6–23,2] | ≈86 [51–212] |
| Toys · BPR (sırayı görmez) | 3,1 puan | 43 [19–∞] | tanımlanamaz |
| Grocery · BPR (sırayı görmez) | 2,4 puan | 110 [20–∞] | tanımlanamaz |

"∞" = bootstrap GA'sının üst ucu uyumun λ alt sınırına dayanıyor; önceden kayıtlı ölçüyle
sınırlı bir yarı ömür yok.

**Okuma.**
- **Toys, sıralı model:** hediye, önerilerde hediyenin alt kategorisine 2 puan fazla slot
  açıyor. Kullanıcının **bir** kendi alımından sonra bu fazla yarıdan aşağı iniyor, dört
  alımdan sonra sıfırda. Kabaca 2–4 hafta.
- **Grocery, sıralı model:** fazla küçük (1,5 puan) ama yavaş sönüyor. Grocery kullanıcıları
  seyrek review yazdığı için hafta karşılığı uzun ve belirsiz (51–212 hafta).
- **BPR:** fazla pay n arttıkça azalıyor (F22), ama önceden kayıtlı uyum sınırlı bir yarı
  ömür vermiyor. BPR sırayı görmüyor. Azalma muhtemelen hediyenin kullanıcı profilinde yeni
  alımlarla **seyrelmesi**, unutulması değil. Pazarlama dilinde: klasik iş birlikçi
  filtrelemede kirlilik zamanla değil, yeni alımlarla azalıyor.

**Sonuç görüldükten sonra bulunan bir durum ve duyarlılık analizi (post-hoc).** n = 0
kovasındaki kullanıcıların çoğunda son hediye, kullanıcının **valid satırı** (testten önceki
son etkileşim): Toys'ta 15.434 kullanıcının 11.486'sında (%74), Grocery'de 2.688'in 2.422'sinde
(%90). Koşullar yalnızca eğitim satırlarını değiştirdiği için bu hediye C1'de silinmiyor.
SASRec onu hem C0'da hem C1'de test girdisinin son ürünü olarak görüyor. BPR ise hiçbir
koşulda valid satırıyla eğitilmiyor. Dolayısıyla n = 0 kovası öteki kovalardan yapısal olarak
farklı. n = 0 dışarıda bırakılıp aynı uyum kova ortalamalarına yapılınca (**önceden kayıtlı
değil, GA'sız nokta tahmini**):

| Hücre | Birincil yarı ömür | n ≥ 1 kovalarıyla (post-hoc) |
|---|---:|---:|
| Toys · SASRec | 0,66 (≈2,4 hafta) | 1,08 (≈3,9 hafta) |
| Grocery · SASRec | 9,5 (≈86 hafta) | 4,2 (≈38 hafta) |
| Toys · BPR | 43 [19–∞] | 4,4 |
| Grocery · BPR | 110 [20–∞] | 5,6 |

Sıralı modelin Toys'taki cevabı iki yolla da aynı mertebede: **bir kendi alımı civarı**.
Grocery ve BPR'deki sayılar n = 0'ın nasıl ele alındığına duyarlı. Birincil değer önceden
kayıtlı olan; bu tablo yalnızca duyarlılık.

### 6.2 M1 — Retargeting israf payı

Top-10 öneri slotlarının kullanıcının **yalnızca hediye olarak** girdiği alt kategorilerden
gelen payı; bu tür alt kategorisi olan test kullanıcılarında (Toys 37.398 / 117.386 = %32;
Grocery 13.307 / 238.756 = %6). Pozitif = soldaki koşul daha çok slot harcıyor.

| Hücre | C0 payı | C1 payı | **C0 − C1** [%95 GA] | **C4 − C1** (plaseboya karşı) | C0 − C3 |
|---|---:|---:|---|---|---|
| Toys · SASRec | %19,8 | %18,5 | +1,3 puan [+1,2, +1,4] | +0,4 [+0,3, +0,5] | +1,0 [+0,9, +1,1] |
| Toys · BPR | %16,3 | %11,4 | **+4,9** [+4,8, +5,0] | **+4,2** [+4,0, +4,3] | +3,0 [+2,9, +3,1] |
| Grocery · SASRec | %13,4 | %12,0 | +1,4 [+1,2, +1,5] | +1,2 [+1,1, +1,3] | +0,8 [+0,7, +0,9] |
| Grocery · BPR | %13,1 | %10,7 | +2,4 [+2,2, +2,6] | +2,2 [+2,1, +2,4] | +1,5 [+1,3, +1,7] |

Geniş eksen (C1b): C0 − C1b Toys SASRec +3,3 · BPR +6,3 · Grocery SASRec +1,3 · BPR +2,3
puan; C4b − C1b sırasıyla −0,14 [−0,25, −0,05] · +3,8 · +1,3 · +2,4 (hepsinin GA'sı sıfırı dışlıyor).

**Okuma.** Toys'ta test kullanıcılarının üçte birinin yalnızca hediye olarak alışveriş yaptığı
bir alt kategorisi var. BPR bu kullanıcıların top-10'unun %16'sını oraya ayırıyor; hediyeleri
görmemiş model %11'ini. Kabaca **her 100 slottan 5'i** hediyenin açtığı ilgiye gidiyor.
Veri kaybı kontrol edildiğinde (C4 − C1) bunun 4,2 puanı kalıyor. SASRec'te fark küçük
(+1,3) ve büyük kısmı veri kaybıyla açıklanıyor (C4 − C1 yalnızca +0,4). "İsraf" bir üst
sınır varsayımıdır: bu slotların hiç dönüşmediği ölçülmedi — kullanıcı o alt kategoride
sonradan kendisi için de alışveriş yapabilir.

### 6.3 M3 — Geçmiş uzunluğuna göre C1 − C4

Test kullanıcıları geçmiş uzunluğunun üçte birlik dilimlerine ayrıldı (kesim noktaları 5 ve
7 etkileşim; Toys 61.471 / 24.718 / 31.197, Grocery 115.003 / 48.950 / 74.803).

| Hücre · C1 − C4, Recall@10 | kısa (≤5) | orta (6–7) | uzun (≥8) |
|---|---|---|---|
| Toys · SASRec | +0,54 [+0,44, +0,62] | +0,55 [+0,41, +0,69] | +0,55 [+0,42, +0,67] |
| Toys · BPR | +0,54 [+0,48, +0,61] | +0,52 [+0,42, +0,62] | +0,33 [+0,25, +0,41] |
| Grocery · SASRec | +0,03 [−0,02, +0,08] | +0,04 [−0,03, +0,12] | +0,04 [−0,01, +0,10] |
| Grocery · BPR | +0,06 [+0,02, +0,10] | +0,00 [−0,06, +0,07] | −0,02 [−0,07, +0,03] |

Toys'ta etki her dilimde var. SASRec'te geçmiş uzunluğuyla değişmiyor; BPR'de uzun geçmişli
kullanıcıda daha küçük (muhtemelen hediye uzun profilde seyreliyor; sınanmadı). Grocery'de dilim düzeyinde
saptanamıyor (BPR kısa dilim hariç). C1b − C4b iki kategoride de her dilimde pozitif
(`marketing_metrics.json` → `M3_segments`).

---

## 7. Sınırlılıklar

1. **Önceden kayıtlı iki detektör eşiği tutturulamadı.** `concept-note` §5 iki eşik
   yazmıştı: makro-F1 **≥ 0,75** ve `gift_given` kesinliği **≥ 0,80**. Ölçülen: **0,5404**
   [0,4897–0,5869] (popülasyonda 0,494) ve **0,68** (popülasyonda 0,6485). Eşiklerin yazılı
   karşılığı "prompt v4 yaz / modeli değiştir" ve "confidence eşiğini yükselt"ti; ikisi de
   yapılmadı — v4, Kapı 1'in yeniden koşulmasını gerektiriyordu ve 500 doğrulama satırı
   prompt ayarı için kullanılamazdı (tek bağımsız referans yok olurdu), `confidence` alanı
   ise zaten 2026-08-29'da ölçümle düşürülmüştü (`low` ≡ `unclear`). Eşikler config'e hiç
   girmedi, yani fiilen hiçbir zaman kapı olarak kurulmadılar. Proje bu ölçümle ilerledi;
   sonuçlar bu yüzden alt sınır olarak okunuyor (madde 4). Gerekçe: DECISIONS 2026-09-20.
2. **Birincil metrik, concept-note'un yazdığı NDCG@10 değil Recall@10.** Değişiklik deney
   koşulmadan config'e yazıldı (2026-09-14) ama kaydedilmemişti. Tek bir hücrede sonucu
   çeviriyor: Grocery × BPR C1 − C4, Recall'da "saptanamadı" [−0,000085, +0,000487],
   NDCG'de "alt sınır" [+0,000055, +0,000352]. İki metrik de §4.2'de yan yana duruyor.
   Gerekçe: DECISIONS 2026-09-20.
3. **İnsan referansı tek kişi; etiket güvenilirliği ölçülmedi.** Hafta 4 kapısı bu yüzden
   INCOMPLETE (asla PASS yazılmadı). Bütün kalibrasyon ve etiket kalitesi sayıları tek
   etiketleyiciye göre.
4. **C1 etiketi gürültülü:** popülasyonda kesinlik 0,65 (öğrenci 0,66) — "hediye" denen
   satırların yaklaşık üçte biri insana göre hediye değil, çoğu kendi çocuğuna alınan. RQ2
   etkileri bu yüzden alt sınır; gerçek etki muhtemelen daha büyük.
5. **Valid satırındaki hediye silinmiyor** (sonuçtan sonra fark edildi, §6.1). Koşullar
   yalnızca eğitim satırlarını değiştiriyor; testten önceki son etkileşim (valid) her koşulda
   yerinde. Toys'ta test kullanıcılarının **%15,6**'sında bu satır `gift_given` (C1b kümesiyle
   %37,0); Grocery'de %2,6 / %6,5. SASRec onu C1/C1b/C3'te de test girdisinde görüyor. Yani
   SASRec için C1 kısmi bir temizlik ve C1 − C4 ile C1 − C0 farklarını **küçültür**, büyütmez.
   BPR valid satırıyla hiçbir koşulda eğitilmediği için etkilenmiyor. Tasarım bilinçliydi
   (bölme C0'da donar), sonuca etkisi önceden yazılmamıştı.
6. **Kaç epoch eğitildiği bilinmiyor.** RecBole'un epoch satırları loglara düşmedi; 300
   tavanına değen koşu olup olmadığı söylenemez. Protokol her koşulda aynı, ama mutlak
   metriklerin tam yakınsadığı iddia edilemez. Hiperparametre araması yapılmadı (RecBole
   varsayılanları, bütün koşullarda aynı).
7. **SASRec'in erken durdurma kümesi koşula göre küçülüyor:** eğitim geçmişi tamamen silinen
   kullanıcının valid satırı düşüyor (Toys C1b'de %17). Test kümesi her koşulda aynı.
8. **Deneyde iki kategori var.** Doz–yanıt iki noktalı; Video Games deneye girmedi, All
   Beauty'nin 5-core'u boş.
9. **Kalibrasyonun varsayımı Toys'ta tutmuyor** (insan–LLM uyumu kategoriden bağımsız
   değil): Toys'un kendi uyumuyla C1b oranı %54,4, havuzlanmış uyumla %44,2. Düşük hediyeli
   kategorilerde C1b'nin kalibre oranı muhtemelen yukarı yanlı.
8. `received` sınıfı pratikte güvenilmez: etiketleyici 11 kez kullandı, LLM'in bu sınıftaki
   kesinliği popülasyonda 0,09. C1b'ye girer, C1'e girmez.
10. **Öğrenci `clean` dağılımında eğitildi, 5-core'u etiketledi.** LLM etiketlerinin yalnızca
   küçük bir kısmı (Toys'ta %17,6) 5-core'da. Kayma ölçüldü ve raporlandı (DECISIONS
   2026-09-15), düzeltilmedi.
11. **Review ≠ alım:** yalnızca review yazılmış alımlar görülüyor; review tarihi alımın
    gerisinde (mevsim tepesi Aralık–Ocak'a kayıyor; M2'nin haftaya çevirisi yaklaşık).
12. **RQ3 tek bir işaretleme yöntemini sınıyor** (gölge token). C2 uygulanmadı.
13. **M1 "israfı" dönüşüm ölçmüyor**, yalnızca slot payı. **M2 gözlemsel:** n'si büyük
    kullanıcılar başka açılardan da farklı (daha uzun geçmiş); fark C0 − C1 eşli olduğu için
    kullanıcı sabit, ama n'nin kendisi rastgele atanmadı.
14. **Çoklu karşılaştırma düzeltmesi yok.** Birincil karşıtlık (C1 − C4, Recall@10) önceden
    belirlendi; öteki karşıtlıklar ve metrikler ikincil ve GA'larıyla birlikte okunmalı.
15. **Kapı 2'nin eski ifadesi** belgelerde başka bir ölçütü anlatıyordu (§3).

---

## 8. Yeniden üretim

```bash
cd repo
# koşu raporları reports/results/experiment_*.json (72), kullanıcı başı dosyalar
# data/processed/recbole/<kategori>/peruser/ (git'e girmez; Kaggle çıktısından)
python -m gift_contamination.analysis.experiment_stats  --config configs/base.yaml  # gate2.json, experiment_stats.json (~14 dk)
python -m gift_contamination.recsys.marketing_metrics   --config configs/base.yaml  # marketing_metrics.json (~4 dk)
python -m gift_contamination.analysis.result_figures    --config configs/base.yaml  # F20–F23
```

Koşular: `scripts/kaggle_experiment.py` (DECISIONS 2026-09-16, 2026-09-18, 2026-09-19).
Kaggle oturumları arasında iki kez koşan 7 hücrenin çıktısı **bit bit aynı** — deney
belirlenimci. Bootstrap'lar config'teki seed'den `zlib.crc32` ile türetilen seed'lerle; aynı
girdiyle aynı GA'lar çıkar (`marketing_metrics` ikinci kez koşturuldu, çıktı aynı).

## Figürler

| Figür | Ne gösteriyor |
|---|---|
| F17 | LLM ve sözcüksel vekil — aylık hediye oranı (dört kategori) |
| F18 | LLM vs insan (500 satır) |
| F19 | Kategori bazlı yaygınlık, ham ve kalibre |
| F20 | Damıtma sadakati |
| **F21** | Koşul karşıtlıkları (RQ2/RQ3), Recall@10, %95 GA |
| **F22** | M2 — hediyeden sonraki kendi alımlarına göre fazla pay ve uyum |
| **F23** | M1 — hediye-yalnız alt kategorilere giden slot payı |
