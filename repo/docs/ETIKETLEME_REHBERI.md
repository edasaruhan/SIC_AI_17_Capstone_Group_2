# Etiketleme Rehberi

**Bu rehber, elinde `validation_500_A.xlsx` (ya da `_B` / `_C`) dosyası olan kişi için
yazıldı.** Projeyi hiç bilmiyor olabilirsiniz — gerekmiyor, buradaki her şey sıfırdan
anlatılıyor. Okuması yaklaşık 15 dakika sürüyor; başlamadan önce **§1–§5 arasını
mutlaka okuyun**, zor vakaları (§6) iş sırasında da açıp bakabilirsiniz.

Sorunuz kalırsa §8'e bakın; orada da yoksa sorun — cevabı buraya ekleyeceğiz.

> Rehberdeki bütün örnek cümleler **uydurmadır**. Gerçek review metinleri gizlilik
> gereği bu depoya girmiyor.

---

## 1. Ne yapıyoruz, siz neden buradasınız

### Problem

Amazon, Netflix, Spotify — hepsinin öneri motoru aynı sessiz varsayımla çalışır:
**satın aldıysan beğenmişsindir.** Bu varsayım çoğu zaman işe yarar. Bir yerde
tamamen çöker: **hediyeler.**

Torununa oyuncak tren alan bir kadın, o siparişten sonra aylarca oyuncak tren
önerisi alır. Ürünü hiç açmamıştır, zevkini hiç yansıtmaz. Ama sistem için o satın
alma, kendisi için aldığı her şeyle aynı ağırlıktadır. Biz buna **kontaminasyon**
(kirlenme) diyoruz.

Literatür bu problemi biliyor ama **ölçmüyor**. Kaç yüzde olduğu, hangi kategoride
yoğunlaştığı, temizlendiğinde önerinin gerçekten düzelip düzelmediği ölçülmemiş.
Projemiz tam olarak bunu ölçüyor.

### Sizin işiniz neden kritik

Bir yapay zekâ modeline 47.200 Amazon yorumunu okuttuk ve her birine "bu kişi
kendisi için mi aldı, başkasına mı?" diye etiket koydurduk. Model bir cevap üretti:
örneğin oyuncak kategorisinde alımların **%23'ü hediye**.

Peki modelin doğru söylediğini nereden bileceğiz?

**Modele soramayız.** Kendi ödevini kendisi kontrol etmiş olur. Tek yol, birkaç
insanın aynı yorumları **modelden bağımsız** olarak etiketlemesi ve iki kümeyi
karşılaştırmak.

İşte o insanlar sizsiniz. Elinizdeki 500 satır **projenin tek gerçek referansı**.
Raporda yazacak her sayı — hediye oranı, modelin doğruluğu, deneyin sonucu —
sizin bu 500 satırda verdiğiniz kararların üzerine kurulacak.

Yani: bu bir "veri girişi" işi değil. Ölçümün kendisi.

---

## 2. Başlamadan önce

### Dosyanız

Size bir Excel dosyası gönderildi: **`validation_500_A.xlsx`**, `_B` veya `_C`.
Harf sizin kimliğiniz — üçünüz **aynı 500 satırı** ayrı ayrı etiketliyorsunuz.

Kurulum yok, program yok, komut yok. Excel (ya da LibreOffice, Google Sheets)
yeter. **Dosya adını değiştirmeyin**, sonunda aynı adla geri gönderin.

### Sayfada ne var

| Kolon | Ne işe yarar |
|---|---|
| `val_id` | Satır numarası. Dokunmayın. |
| `category` | Ürünün kategorisi — oyuncak mı, kozmetik mi, oyun mu, gıda mı |
| `title` | Yorumun başlığı |
| `text` | Yorumun kendisi. **Asıl okuyacağınız yer burası.** |
| `label` | **Sizin dolduracağınız kolon.** Açılır menü var. |
| `notes` | İsteğe bağlı not. Çoğu satırda boş kalacak (§7). |

`label` ve `notes` sarı renkli — doldurulacak yerler onlar.

### Yorumlar İNGİLİZCE

Veri seti Amazon'un ABD mağazasından geliyor; bütün yorumlar İngilizce.
Anlamadığınız bir cümleyi çevirmekte hiçbir sakınca yok (§8'de detay).

Yorumların çoğu kısa: **%43'ü 20 kelimeden az**, medyan 24 kelime. Uzun olanlar
(60+ kelime) 500 satırın sadece 76'sı — yani işin büyük kısmı hızlı geçecek.

### Kategoriler

| Kategori | Kaç satır |
|---|---|
| Toys and Games (oyuncak) | 200 |
| Video Games (oyun) | 100 |
| All Beauty (kozmetik/bakım) | 100 |
| Grocery and Gourmet Food (gıda) | 100 |

Oyuncak iki katı çünkü hediye oranının en yüksek olduğu kategori orası.

### Ne kadar sürer

**Satır başına 30–60 saniye, toplam 3–4 saat.** Tek oturuşta yapmayın — bölerek
çalışın (örneğin günde 100 satır). Yorgunken verilen etiketler ölçümü bozar.

Kaydedip kapatın, sonra kaldığınız yerden devam edin. Yarım dosya sorun değil.

---

## 3. Üç kural — bunlar ölçümün kendisi

Bu üçü "iyi olur" değil, **zorunlu**. İhlal edilirse 500 satırlık emek ölçüm
değeri taşımaz.

**1 · Kendi dosyanızdan başkasını açmayın.**
Üçünüz aynı satırlara bakıyorsunuz. Ölçtüğümüz şey *"iki bağımsız insan aynı
yoruma aynı etiketi veriyor mu?"* Birinin cevabını görürseniz o soru anlamsızlaşır.

**2 · Etiketlerken birbirinizle tartışmayın.**
Zor vakaları merak ediyorsanız **bittikten sonra** konuşun. Anlaşmazlığın kendisi
veridir — hangi durumların gerçekten belirsiz olduğunu bize o gösteriyor. Baştan
uzlaşırsanız o bilgiyi silmiş olursunuz.

**3 · ChatGPT'ye, Claude'a, herhangi bir yapay zekâya sormayın.**
Bütün projenin amacı *modelin ne dediğini insanla karşılaştırmak*. Etiketleriniz
bir modelden gelirse ölçtüğümüz şey iki modelin birbirine benzerliği olur —
doğruluk değil. Bir cümleyi **çevirmek** için kullanmak serbest (§8), etiketi
**sormak** yasak.

> **Modelin cevabı sayfada yok — bilerek.** Modelin verdiği etiket, ne kadar emin
> olduğu, hangi cümleye dayandığı: hiçbiri sayfaya yazılmadı. Görseydiniz işiniz
> hızlanırdı ama kararınız ona yaslanırdı. Bağımsızlık hıza tercih edildi.

---

## 4. Tek soru: bu ürünü kim kullanacak?

Her satırda cevaplamanız gereken tek soru bu:

> Yorumu yazan kişi ürünü **kendi zevkine göre kendisi için mi** aldı,
> yoksa **başkası için mi**?

Beş cevap var:

| Etiket | Ne demek | Kısa örnek |
|---|---|---|
| `self` | Kendisi için almış, kendisi kullanıyor | *"Üç haftadır kullanıyorum, memnunum."* |
| `gift_given` | Başkasına hediye vermiş | *"Torunuma aldım, bayıldı."* |
| `household` | Ev halkı için / ortak kullanım. Hediye değil ama tam olarak kendi zevki de değil | *"Mutfağımız için aldık."* |
| `received` | Kendisi hediye **almış** — kullanıyor ama seçen o değil | *"Doğum günümde hediye geldi."* |
| `unclear` | Metinde karar vermeye yetecek bilgi yok | *"Harika. Beş yıldız."* |

`household` neden ayrı bir sınıf? Çünkü öneri sistemi açısından *"eşimle ortak
kullandığımız blender"* ile *"kendime aldığım koşu ayakkabısı"* aynı şey değil.
İkisi de hediye değil ama biri kişinin kendi zevkini yansıtmıyor.

---

## 5. Karar akışı

Sırayla sorun, **ilk "evet"te durun**:

**1 · Ürün başka birine verilmiş mi?**
*"torunuma aldım"*, *"kızımın doğum günü için"*, *"arkadaşıma hediye ettim"*
→ **`gift_given`**

**2 · Ürün evdeki biri için veya ortak kullanım için mi?**
*"bebeğimizin bezleri"*, *"mutfağımız için"*, *"eşimle ikimiz kullanıyoruz"*
→ **`household`**

**3 · Yorumu yazan kişi ürünü hediye mi almış?**
*"doğum günümde hediye geldi"*, *"kardeşim yolladı"*
→ **`received`**

**4 · Kişinin ürünü kendi kullandığına dair bir şey var mı?**
*"üç haftadır kullanıyorum"*, *"cildime iyi geldi"*, *"kurulumu kolaydı"*
→ **`self`**

**5 · Hiçbiri yoksa** → **`unclear`**

---

## 6. Zor vakalar — asıl iş burada

Kolay satırlar zaten kolay. Bu 500 satır **bilerek zor vakalarla dolduruldu**:
sınırda duran, modeli yanıltan örnekler özellikle seçildi. Aşağıdakiler modelin
en çok yanıldığı yerler.

### 6.1 "Hediye olur" — üç ayrı durum, üç ayrı etiket

"Gift" kelimesi üç tamamen farklı bağlamda geçiyor ve üçü farklı etiket alıyor.
Bu, denemelerimizin en önemli bulgusuydu.

**(1) Gerçekleşmiş hediye.** Ürün el değiştirmiş. Genelde geçmiş zaman.
> *"Torunuma harika bir hediye oldu."* · *"Yeğenlerime yolladım."*

→ **`gift_given`**. Dikkat: **"aldım" fiilini aramayın** — *"harika bir hediye
oldu"* cümlesinde satın alma geçmese de gerçekleşmiş bir hediyedir.

**(2) Tavsiye + kendi kullanımı da anlatılıyor.**
> *"Yirmi yıldır alıyorum, hiç pişman olmadım. Hediye olarak da çok iyi gider."*

→ **`self`**. Kişi ürünü kendisi kullanıyor; ayrıca hediye fikri veriyor.

**(3) Tavsiye + başka hiçbir kanıt yok.**
> *"Bu ürün harika bir hediye olur."* · *"Üniversite öğrencilerine ideal hediye."*

→ **`unclear`**. Kimin aldığı, kullandığı, verdiği belli değil.

> **Neden (3) `self` değil?** Hediye kanıtının yokluğu, kendine aldığının kanıtı
> değildir. `self` demek metinde olmayan bir şeyi uydurmaktır. §6.7 ile birlikte
> okuyun — ikisi aynı kuralın iki yüzü.

Ayırt edici sıra: **(a)** ürün el değiştirmiş mi? → `gift_given` · **(b)**
değilse kendi kullanımına dair bir şey var mı? → `self` · **(c)** ikisi de yoksa
→ `unclear`.

### 6.2 Hediye **almak** ≠ hediye **vermek** → `received`

> *"Bunu doğum günümde hediye aldım ve bayıldım."*

Yorumu yazan kişi **alıcı**. Ürün onda, o kullanıyor, yorum onun deneyimi — ama
ürünü **o seçmedi**. → **`received`**

Bu sınıf özellikle önemli, çünkü öneri sistemi açısından bu da bir kirlenme:
kişi ürünü beğenmiş olabilir ama o tercihi kendisi yapmadı.

### 6.3 Torun ≠ ev halkı → `gift_given`

> *"Torunum için aldım, çok sevdi."*

**Şemadaki en kritik ayrım.** Torun ayrı hanede yaşar; ona alınan şey hediyedir.
Oyuncak kategorisinde hediye olarak verilen ürünlerin **%44'ü toruna** gidiyor
(alıcısı belli olanların yarısı) — yani en büyük kategorimizdeki hediyelerin
neredeyse yarısı bu tek karara bağlı.

`household` **değil**. Yeğen, kuzen, "arkadaşımın çocuğu" da aynı şekilde
**`gift_given`**.

### 6.4 Kendi çocuğu — en zor vaka

**Önce hane sınırını çizin.** `household` ile `gift_given` arasındaki fark
akrabalık derecesi değil, **aynı evde yaşanıp yaşanmadığı**:

| Aynı evde → `household` adayı | Ayrı evde → her zaman `gift_given` |
|---|---|
| "my son", "my daughter", "my kids" (oğlum, kızım, çocuklarım) | "my grandson", "my granddaughter" (torunum) |
| "my 5 year old", "my little one", "my toddler" (5 yaşındaki, küçüğüm) | "my niece", "my nephew", "my cousin" (yeğenim, kuzenim) |
| "my baby", "our baby" (bebeğim/bebeğimiz) | "my friend's daughter" (arkadaşımın kızı) |
| "my wife/husband" — ortak kullanımsa | "my mom", "my sister" — ayrı yaşıyorsa |

Yaşla anlatılan çocuk ("my 3 and 6 year old") **kendi çocuğudur** — torun
neredeyse her zaman açıkça "grand-" ile yazılır.

**Sonra kuralı uygulayın.** Burada tek bir "doğru" cevap yok, o yüzden
**tutarlılık doğruluktan önemli**. Şu sırayla:

- **Özel bir gün geçiyorsa** ("birthday", "Christmas", "karne hediyesi")
  → **`gift_given`**
- **Günlük / ortak kullanımsa** ("okul çantası", "bebek maması", "çocuk odasına")
  → **`household`**
- **İkisi de yoksa** → **`household`**

> **Güçlü ipucu: kişi ürünü kendi eline almış mı?** Kurulumu anlatıyorsa, pil
> taktıysa, çalışmadığını kendi test ettiyse, "we/biz" diye yazıyorsa — ürün
> evden çıkmamış demektir. Bu `household` lehine güçlü kanıttır.

> **Örnek.** *"2 ve 5 yaşındaki çocuklarım için iki tane aldım... pilleri taktım,
> ikisi de çalışmadı..."*
> → **`household`**. Kendi çocukları (hane içi), özel gün yok, ürünü kendisi test
> etmiş.

Bu vakalarda `notes` kolonuna **`KENDI_COCUGU`** yazın. Neden önemli olduğu §7'de.

### 6.5 "Arkadaşım tavsiye etti" → `self`

> *"Arkadaşım önerdi, ben de aldım."*

Cümlede başkası geçiyor ama **ürünü alan ve kullanan yorumu yazan kişi**.

### 6.6 Eşe alınan — ikisi de kullanıyorsa `household`

- *"Eşime doğum günü hediyesi aldım"* → **`gift_given`** (özel gün + ona ait)
- *"Eşim için aldım ama ikimiz de kullanıyoruz"* → **`household`**
- *"Mutfağımıza aldık"* → **`household`**

### 6.7 Bilgi vermeyen kısa yorum → `unclear`

> *"Harika."* · *"Beş yıldız."* · *"Beklediğim gibi."*

Hediye ifadesinin **yokluğu, `self` kanıtı değildir.** Metin hiçbir şey
söylemiyorsa doğru cevap `unclear`; `self` demek uydurmaktır.

Ama: *"Üç haftadır kullanıyorum, memnunum"* kısa olsa da **`self`** — birinci
ağızdan kullanım var. Uzunluk değil **kanıt** belirliyor.

### 6.8 İş yeri / komşu → `gift_given`

Meslektaş, komşu, öğretmen, "sınıf hediyesi" — hepsi hane dışı → **`gift_given`**.

---

## 7. `notes` kolonuna ne yazılır

**Satırların çoğunda boş kalacak — bu normal.** 500 satırın belki 50'sinde bir
şey yazarsınız.

| Ne zaman | Ne yazılır |
|---|---|
| §6.4'teki "kendi çocuğu" vakası | `KENDI_COCUGU` |
| Etiketi verdiniz ama içinize sinmedi | `EMIN_DEGIL` |
| Rehberde hiç geçmeyen bir durum | Bir cümle, serbest |

Bu iki etiketi **birebir böyle** yazın (Türkçe karakter yok, alt çizgi var) —
sayılabilmeleri için.

**`KENDI_COCUGU` neden önemli?** Ekibin vermesi gereken açık bir karar var:
`household` da kirlenme sayılacak mı? Bu karar hediye oranını oyuncak
kategorisinde %23 ile %43 arasında değiştiriyor. Sizin işaretlediğiniz satırlar
o kararın dayanağı olacak — işaretlemezseniz sayamayız.

---

## 8. Sık sorulanlar

**İngilizce bilmiyorum / bir cümleyi anlamadım. Çevirebilir miyim?**
Evet. Google Translate, DeepL, hatta ChatGPT'ye *"bu cümle ne diyor"* diye
sormak serbest. Yasak olan **etiketi sormak**: *"bu hediye mi?"*, *"hangi etiketi
vermeliyim?"* Kararı siz vereceksiniz, çeviri sadece metni anlamanız için.

**Ürünün ne olduğunu anlamadım.**
`category` kolonuna bakın. Yine de anlamadıysanız `unclear` verin ve `notes`'a
nedenini yazın.

**Hem kendine hem hediye almış.**
> *"İki tane aldım, biri bana biri anneme."*

→ **`gift_given`**. Kirlenme var; öneri sistemi açısından önemli olan bu.

**Emin değilim — %60 hediye gibi geldi.**
Etiketi yine de verin, `notes`'a `EMIN_DEGIL` yazın. `unclear`'ı **kanıt yokluğu**
için saklayın, kararsızlık için değil. İkisi farklı şeyler: biri "metinde bilgi
yok", diğeri "bilgi var ama ben emin olamadım".

**Yorum ürünle ilgisiz (kargo şikâyeti, satıcıya küfür vb.).**
Kim için alındığına dair bir şey yoksa → **`unclear`**.

**Yanlışlıkla satırları sıraladım / filtreledim. Bozuldu mu?**
**Hayır.** Etiketler satır numarasına değil `val_id`'ye bağlanıyor; sıralasanız
bile doğru satıra gider. Filtrelemek de serbest.
**Ama satır SİLMEYİN** — o yakalanır ve hata verir.

**Bir satırı boş bıraktım / yarıda kaldım.**
Sorun değil. Yarım dosya hata değil, ilerleme raporu üretiyor. Kaydedip devam
edin.

**Etiketi yanlış yazarsam?**
`label` hücresine tıklayınca açılır menü çıkıyor; oradan seçerseniz yanlış
yazmanız mümkün değil. Elle yazıp menüyü atlarsanız (ör. kopyala-yapıştır) beş
etiket dışında bir şey girebilirsiniz — ama toplama adımı bunu yakalıyor ve
hangi satır olduğunu söylüyor. Yine de menüden seçmek en güvenlisi.

**Excel yerine başka bir program kullanabilir miyim?**
LibreOffice sorunsuz. Google Sheets'te de açılır ama sonunda **`.xlsx` olarak
indirmeniz** şart (Sheets'in kendi formatı okunmuyor). Hangi programı
kullanırsanız kullanın: **kolon adlarını ve sırasını değiştirmeyin** — dosya o
başlıklara göre okunuyor.

**Kolonların hepsini görmüyorum, eksik mi?**
Kasıtlı. Modelin cevabı ve hangi satırın anahtar kelimeyle işaretlendiği sayfaya
**yazılmadı** (§3). Görseydiniz kararınız ona yaslanırdı.

**Ne kadar sürer, hızlı gidiyorum?**
Satır başına 30–60 saniye normal. Çok daha hızlıysanız muhtemelen metni
okumuyorsunuz; çok daha yavaşsanız fazla düşünüyorsunuz — kararsız kalınca
`notes`'a `EMIN_DEGIL` yazıp geçin.

**Bitince ne yapayım?**
Dosyayı **adını değiştirmeden** kaydedip geri gönderin.

---

## 9. Bittikten sonra ne oluyor

> **Güncelleme (2026-09-14).** Doğrulama **tek etiketleyiciyle (A)** tamamlandı; B ve C
> teslim etmedi. Bu yüzden aşağıdaki **uyum (Fleiss κ) hesaplanmadı** ve kapı PASS
> değil **INCOMPLETE** olarak kayıtlı. Modelin doğruluğu ve anahtar kelime yöntemi A'nın
> etiketlerine karşı ölçüldü. A'nın "kendi çocuğu" notlarının 131'i `KENDI_COCUGU`
> koduna çevrildi. Sonuçlar ve sınırlılıklar: `DECISIONS.md`, 2026-09-14.
> Aşağısı özgün üç kişilik tasarımı anlatıyor.

Üç dosya toplandığında tek bir komut çalışıyor ve üç sayı çıkıyor:

| Ölçüm | Ne soruyor | Eşik |
|---|---|---|
| **Uyum (Fleiss κ)** | Üçünüz birbirinizle ne kadar uyuşuyorsunuz? | **≥ 0,60** |
| Modelin doğruluğu | Yapay zekâ sizinle ne kadar örtüşüyor? | eşik yok — ölçüm |
| Anahtar kelime yöntemi | Basit kelime araması sizinle ne kadar örtüşüyor? | eşik yok — ölçüm |

**Birincisi bir kapı, diğer ikisi sadece ölçüm.**

"Fleiss κ" korkutucu bir isim ama basit bir şey: *üç kişi aynı satırlara aynı
etiketi ne sıklıkla verdi* — rastgele uyuşma payı düşülmüş hâli. 1,0 tam uyum
demek, 0 ise "yazı tura atmışsınız kadar" demek.

**κ düşük çıkarsa suç sizde değil.** O durumda sorun görev tanımındadır: demek ki
kurallar yeterince net değil ve modelden tutarlılık beklemek anlamsız. Şemayı
sadeleştirir (muhtemelen `household` ile `gift_given` birleşir), gerekçesini
yazar, devam ederiz.

Diğer iki sayıya **bilerek eşik konmadı**: sonucu gördükten sonra eşik uydurmak,
kapıyı sonradan kurmak olurdu.

> **Küçük not.** Eşikler siz etiketlemeye başlamadan **önce** yazıldı ve
> `configs/base.yaml` dosyasına kaydedildi. Sonucu görüp eşik değiştirmek bu
> projede yasak — bulguyu geçersiz kılar.

---

### Teşekkürler

Bu 4 saat, projenin en değerli 4 saati. Geri kalan her şey — modelin doğruluk
iddiası, hediye oranı, öneri deneyinin sonucu — sizin verdiğiniz 500 karara
dayanacak.

---

<sub>**Ekip için not.** Bu rehber bilerek etiketleyene göre yazıldı; sayfaları
üreten ve toplayan komutlar burada değil, `README.md` hızlı başlangıcında ve
`GENEL_BAKIS.md` §8'de. Etiketleme kurallarının kendisi ölçülerek belirlendi ve
gerekçeleri `DECISIONS.md`'de — kural değiştirmeden önce oraya bakın.</sub>
