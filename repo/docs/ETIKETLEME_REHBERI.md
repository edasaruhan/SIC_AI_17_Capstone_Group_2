# Etiketleme Rehberi

> Bu dosya iki yerde kullanılıyor: **Hafta 2**'de prompt geliştirme denemesinde (200 review,
> tek kişi) ve **Hafta 4**'te doğrulama setinde (500 review, 2–3 kişi). Hafta 4'te herkes
> aynı kuralları uygulamak zorunda — yoksa ölçtüğümüz κ, gerçek belirsizliği değil
> rehbersizliği ölçer.
>
> Aşağıdaki örnekler **uydurmadır**. Gerçek review metni bu depoya girmez.

---

## 0. Nasıl çalışılır

```bash
cd repo

# 1. Excel sayfasını üret (zaten üretildiyse atlar)
.venv/Scripts/python.exe -m gift_contamination.data.labelsheet --export

# 2. data/annotations/human/prompt_trial_200.xlsx dosyasını aç, `label` kolonunu doldur

# 3. Bitince (ya da ara verirken) geri al ve doğrula
.venv/Scripts/python.exe -m gift_contamination.data.labelsheet --ingest
```

**CSV'yi değil xlsx'i doldurun.** CSV'yi Excel'de açıp kaydetmek Türkçe Windows'ta ayracı
`;` yapar ve metni cp1254'e düşürür; ikisi de dosyayı sessizce bozar ve bu ancak saatler
sonra fark edilir. xlsx'te bu kavramlar yok.

Sayfada `label` ve `notes` kolonları sarı. `label` açılır menülü — dördü dışında bir şey
yazamazsınız. `notes` çoğu satırda boş kalır; ne zaman doldurulacağı §4'te.

**Satır silmeyin, sıralamayı değiştirmeyin, filtrelemek serbest.** `--ingest` satır
kaybını yakalar ve hata verir.

`--ingest` her an çalıştırılabilir: yarım dosya hata değil, ilerleme raporu verir.
İş bölerek yapılabilir (örn. günde 50 satır).

---

## 1. Soru şu: bu ürünü kim kullanacak?

Ürünü **değerlendiren kişinin kendi zevkini** yansıtıyor mu, yoksa **başkası için mi**
alınmış? Projenin tamamı bu ayrımın üzerine kurulu: öneri sistemi her satın almayı
"bu kişi bunu sevdi" diye okuyor, hediyelerde bu yanlış.

| Etiket | Anlamı |
|---|---|
| `self` | Kendisi için almış (ya da hediye **almış**, veren değil) |
| `gift_given` | Başkasına hediye olarak vermiş |
| `household` | Ev halkı için / ortak kullanım. Hediye değil ama kendi zevki de değil |
| `unclear` | Metinde karar vermeye yetecek kanıt yok |

---

## 2. Karar akışı

Sırayla sorun, ilk "evet"te durun:

**1. Metin, ürünün başka birine verildiğini/alındığını söylüyor mu?**
   Örn. "torunuma aldım", "kızımın doğum günü için", "arkadaşıma hediye ettim"
   → **`gift_given`**

**2. Ürün, değerlendirenin evindeki biri için veya ortak kullanım için mi?**
   Örn. "bebeğimizin bezleri", "mutfağımız için aldık", "eşimle ikimiz kullanıyoruz"
   → **`household`**

**3. Metinde ürünün kullanımı/deneyimi hakkında birinci ağızdan bir şey var mı?**
   Örn. "üç haftadır kullanıyorum", "cildime iyi geldi", "kurulumu kolaydı"
   → **`self`**

**4. Hiçbiri yoksa** → **`unclear`**

---

## 3. Tuzaklar — asıl iş burada

Kolay satırlar zaten kolay. Bu 200 satır kasıtlı olarak zor vakalarla dolduruldu
(%50'si anahtar kelime ile işaretlenmiş, %26'sı "hediye olur" gibi spekülatif ifade
içeriyor). Aşağıdakiler prompt'un kırıldığı yerler.

### 3.1 "Harika bir hediye olur" → `self`

> *"Ürün çok kaliteli, kesinlikle güzel bir hediye olur."*

Bu kişi **hediye almamış**, hediye olabileceğini söylüyor. Bu bir **tavsiye**, bir işlem
değil. Bu tuzak korpusta çok yaygın ve sözcüksel vekilin en büyük hata kaynağı.

Ayırt edici soru: **fiil geçmiş zamanda ve gerçekleşmiş mi?**
- "hediye ettim", "aldım ona" → gerçekleşmiş → `gift_given`
- "hediye olur", "hediye alınabilir", "düşünüyorum" → gerçekleşmemiş → `self`

### 3.2 Hediye **almak** ≠ hediye **vermek** → `self`

> *"Bunu doğum günümde hediye aldım ve bayıldım."*

Değerlendiren kişi **alıcı**. Ürün onda, o kullanıyor, yorum onun deneyimi.
Öneri sistemi açısından bu bir kirlenme değil → **`self`**.

> Not: teknik olarak bu da kişinin kendi seçimi değil. Ama biz **verenin** hesabındaki
> kirlenmeyi ölçüyoruz; alanın hesabında ürün gerçekten kullanılıyor.

### 3.3 Torun ≠ ev halkı → `gift_given`

> *"Torunum için aldım, çok sevdi."*

**Bu şemadaki en kritik ayrım.** Torun ayrı hanede yaşar; ona alınan şey hediyedir.
Toys_and_Games'te adı geçen alıcıların **%26'sı torun** — yani birincil deney
kategorisinin dörtte biri bu tek karara bağlı.

`household` **değil**. Yeğen, kuzen, "arkadaşımın çocuğu" da aynı şekilde `gift_given`.

### 3.4 Kendi çocuğu — en zor vaka

**Önce hane sınırını çizin.** `household` ile `gift_given` arasındaki fark akrabalık
derecesi değil, **aynı evde yaşanıp yaşanmadığı**:

| Aynı evde (→ `household` adayı) | Ayrı evde (→ her zaman `gift_given`) |
|---|---|
| "my son", "my daughter", "my kids" | "my grandson", "my granddaughter" |
| "my 5 year old", "my little one", "my toddler" | "my niece", "my nephew", "my cousin" |
| "my baby", "our baby" | "my friend's daughter" |
| "my wife/husband" (ortak kullanım) | "my mom", "my dad", "my sister" (ayrı yaşıyorsa) |

Yaşla ifade edilen çocuk ("my 2 and 5 year old") kendi çocuğudur — torun neredeyse her
zaman açıkça "grand-" ile yazılır.

**Sonra kuralı uygulayın.** Burada net bir doğru yok, o yüzden **tutarlılık** doğruluktan
önemli:

- **Vesile adı geçiyorsa** ("doğum günü", "yılbaşı", "karne hediyesi") → **`gift_given`**
- **Günlük/ortak kullanımsa** ("okul çantası", "bebeğin maması", "çocuk odasına aldık")
  → **`household`**
- İkisi de yoksa → **`household`**

**Güçlü bir ipucu: değerlendiren ürünü kendi eline almış mı?** Kurulumu anlatıyorsa, pil
taktıysa, çalışmadığını kendi test ettiyse, "biz/we" diye yazıyorsa — ürün evden çıkmamış
demektir ve puan gerçekten onun yargısıdır. Bu `household` lehine güçlü kanıttır.

> **Örnek.** *"2 yaşındaki ve 5 yaşındaki çocuğum için iki tane aldım... pilleri taktım,
> ikisi de çalışmadı... paranın sayılmasını istiyorduk ama SAYMIYOR."*
> → `household`. Kendi çocukları (hane içi), vesile yok, ve ürünü kendisi test etmiş.

Bu vakaya `notes`'a `KENDI_COCUGU` yazın. Sayısı yüksek çıkarsa kuralı Hafta 4'te
κ sonucuna göre yeniden tanımlayacağız — ama ancak işaretlerseniz sayabiliriz.

### 3.5 "Arkadaşım tavsiye etti" → `self`

> *"Arkadaşım önerdi, ben de aldım."*

Başkası cümlede geçiyor ama **ürünü alan ve kullanan değerlendiren kişi**.

### 3.6 Eşe alınan — ikisi de kullanıyorsa `household`

- "Eşime doğum günü hediyesi aldım" → `gift_given`
- "Eşim için aldım ama ikimiz de kullanıyoruz" → `household`
- "Mutfağımıza aldık" → `household`

### 3.7 Bilgi vermeyen kısa review → `unclear`

> *"Harika."* · *"Beş yıldız."* · *"Beklediğim gibi."*

Hediye ifadesinin **yokluğu, `self` kanıtı değildir.** Metin hiçbir şey söylemiyorsa
`unclear` doğru cevaptır; `self` demek uydurmaktır.

Ama: *"Üç haftadır kullanıyorum, memnunum"* kısa olsa da `self` — birinci ağızdan
kullanım var.

### 3.8 İş yeri / komşu → `gift_given`

Meslektaş, komşu, öğretmen — hepsi hane dışı → `gift_given`.

---

## 4. `notes` kolonuna ne yazılır

**Satırların çoğunda boş kalacak, bu normal.** 200 satırın 30-40'ında bir şey yazarsınız.

Ama yazdıklarınız prompt v2'nin yazıldığı yer. Mekanizma şu: etiketleyenin kafasındaki
tereddüdün LLM'e talimat olarak geçmesinin tek kanalı bu kolon. Bir tuzak 15 satırda
tekrarlanıyor ama not düşülmediyse, prompt'a girmez ve LLM aynı hatayı **40.000 satırda**
yapar.

| Ne zaman | Ne yazılır |
|---|---|
| §3.4 kendi çocuğu vakası | `KENDI_COCUGU` |
| Etiket verildi ama içe sinmedi | `EMIN_DEGIL` |
| Rehberde olmayan bir durum | Bir cümle, serbest |

Uzun yazmayın; etiket başına birkaç kelime yeterli. `KENDI_COCUGU` ve `EMIN_DEGIL`
etiketleri birebir bu şekilde yazılmalı - sayılabilmeleri için.

---

## 5. Sık sorulanlar

**Ürünün ne olduğunu bilmiyorum, karar veremiyorum.**
`category` kolonu var. Yetmiyorsa `unclear` + `notes`'a nedenini yazın.

**Hem kendine hem hediye almış.**
> *"İki tane aldım, biri bana biri anneme."*

`gift_given`. Kirlenme var — öneri sistemi açısından önemli olan bu.

**Emin değilim, %60 hediye gibi.**
Etiketi verin, `notes`'a `EMIN_DEGIL` yazın. `unclear`'ı **kanıt yokluğu** için saklayın,
kararsızlık için değil. İkisi farklı şeyler ve karıştırılırsa `unclear` oranı
detektörün performansı hakkında yanlış bilgi verir.

**Yorum ürünle ilgisiz (kargo şikâyeti vb.).**
Kim için alındığına dair bir şey yoksa → `unclear`.

**Ne kadar sürer?**
200 satır ≈ 2–3 saat. Satır başına 30–60 saniye. Daha hızlı gidiyorsanız muhtemelen
metni okumuyorsunuz; daha yavaşsanız fazla düşünüyorsunuz — kararsız kalınca
`notes`'a yazıp geçin.

**Kolonların hepsini görmüyorum.**
Kasıtlı. Hangi satırın anahtar kelimeyle işaretlendiği sayfada **yok**, çünkü görseydiniz
etiketleriniz o karara yaslanırdı ve insan–algoritma karşılaştırması kendi kendini
doğrulayan bir ölçüme dönerdi. `--ingest` o kolonları geri ekliyor.

---

## 6. Bittikten sonra ne oluyor

`--ingest` şunu yazdırır:

```
etiketlenen: 200 / 200
  self          128
  gift_given     44
  household      19
  unclear         9
sozcuksel vekile karsi: {'proxy_dogru': 38, 'proxy_yanlis_pozitif': 62,
                         'proxy_kacirdi': 6, 'ikisi_de_hayir': 94}
```

İkinci satır prompt v2'nin asıl girdisi:

- **`proxy_yanlis_pozitif`** — anahtar kelime "hediye" dedi, siz demediniz.
  Çoğu §3.1 (*"hediye olur"*) olacak. Prompt bu ayrımı zaten yazıyor; sayı yüksekse
  few-shot örneği eklenir.
- **`proxy_kacirdi`** — siz "hediye" dediniz, anahtar kelime kaçırdı. **En değerli
  grup**: LLM'in sözcüksel yöntemden fazlasını yapması gereken yer tam olarak burası.
  Bu satırlar prompt v2'ye few-shot örneği olarak girer.

Sonra `prompts/gift_detection_v2.md` açılır — **v1 üzerine yazılmaz**, prompt dosyası
versiyonlamayı şart koşuyor (hangi annotation hangi prompt'la üretildi izlenebilsin).

⚠️ Bu 200 satırın `row_id`'leri `prompt_trial_ids.json`'a yazıldı ve Hafta 4'ün 500'lük
doğrulama setinden **dışlanacak**. Prompt bu satırlara bakarak yazıldığı için aynı
satırlarla doğrulamak, modeli kendi test setine fit etmek olur.
