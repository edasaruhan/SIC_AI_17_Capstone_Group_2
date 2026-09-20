# Presentation

Capstone sunumu ve tek sayfalık özet. **İkisi de üretilmiş dosyalar** —
elle düzenlenmez, betikler yeniden çalıştırılır.

| Dosya | Nedir |
|---|---|
| `SIC_AI_17_Group_2.pptx` | 22 slayt (17 ana + 5 ek), 16:9. Slaytlar **İngilizce**, konuşmacı notları **Türkçe**. |
| `SIC_AI_17_Group_2.pdf` | Aynı deck'in PDF kopyası (PowerPoint ile dışa aktarıldı). |
| `one_pager.pdf` | A4 portre tek sayfalık özet (İngilizce). Jüriye dosya olarak verilebilir. |
| `one_pager.png` | Aynı sayfanın PNG'si — yalnızca gözle kontrol için. |
| `build_deck.py` | Deck'i üreten betik. |
| `build_one_pager.py` | One-pager'ı üreten betik. |

## Neden betik?

Slaytlardaki ve özetteki **her sayı** `repo/reports/results/*.json` dosyalarından
okunur; hiçbiri betiğe elle yazılmamıştır. Bir artefakt yeniden üretilirse deck
ve one-pager da onunla birlikte değişir, yani sunum sonuçlardan sapamaz.

Figürler `repo/reports/figures/` altındaki **commit'li PNG'lerden olduğu gibi**
gömülür — yeniden çizilmez. Tek istisna one-pager'daki yarı ömür grafiği: A4'e
sığması için tek panel olarak yeniden çiziliyor, ama çizdiği değerler
`marketing_metrics.json`'daki kova ortalamalarının ta kendisi.

## Yeniden üretmek

`python-pptx` **proje bağımlılığı değildir** (`repo/requirements.txt`'te yoktur);
yalnızca deck'i üretmek için gerekir:

```bash
pip install python-pptx
cd presentation
python build_deck.py           # -> SIC_AI_17_Group_2.pptx
python build_one_pager.py      # -> one_pager.pdf + one_pager.png
```

One-pager yalnızca `matplotlib` kullanır, o da zaten projenin bağımlılığıdır.

PDF kopyası ve slayt PNG'leri PowerPoint COM ile alındı (Windows):

```powershell
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open("$PWD\SIC_AI_17_Group_2.pptx", $true, $false, $false)
$pres.Export("$PWD\png", "PNG", 1700, 956)     # slayt slayt gözle kontrol için
$pres.SaveAs("$PWD\SIC_AI_17_Group_2.pdf", 32)
$pres.Close(); $pp.Quit()
```

LibreOffice kurulu bir makinede aynı iş `soffice --headless --convert-to pdf`
ile yapılabilir.

## Slayt haritası

| # | Slayt |
|---|---|
| 1 | Başlık |
| 2 | Problem — hediye ≠ tercih |
| 3 | Pazarlama maliyeti: yanlış sinyal harcamaya dönüşüyor |
| 4 | RQ1–RQ4 ve önceden kayıtlı yorum kuralı |
| 5 | Boru hattı tek bakışta + veri hunisi |
| 6 | Tespit ve **Kapı 1** (PASS 4/4, dört kategori) |
| 7 | **Dürüstlük slaytı** — tutturulamayan iki kapı, tek etiketleyici |
| 8 | Damıtma ve sadakat kapısı (PASS 3/3) |
| 9 | RQ1 — yaygınlık ve Aralık–Ocak tepesi |
| 10 | Deneyin fikri: plasebo neden gerekli + altı koşul |
| 11 | **Kapı 2** — dört hücrede de geçerli |
| 12 | RQ2 — ana bulgu ve doz–yanıt |
| 13 | RQ2b — "plasebodan iyi", "hiçbir şey yapmamaktan iyi" demek değil |
| 14 | RQ3 — silmek mi, işaretlemek mi |
| 15 | RQ4 — yarı ömür (F22) |
| 16 | RQ4b — israf payı (F23) |
| 17 | Kapanış: ne iddia ediyoruz / etmiyoruz / sırada ne var |
| A1 | Altı koşulun tam tanımı |
| A2 | Kapı 1 ve Kapı 2 ölçütleri, tam sayılarla |
| A3 | Bütün karşıtlıklar tek tabloda |
| A4 | Sınırlılıklar |
| A5 | Her sayı nasıl kontrol edilir (provenans, demo komutu) |

Planlanan 16 ana slayt yerine 17 var: RQ2'nin "plaseboya karşı" ve "hiçbir şey
yapmamaya karşı" sonuçları tek slayta sığmıyordu ve ikincisi bulguyu
sınırlandıran taraf — ayrı slayt oldu.

## Bilinen kusur

`F20_distill_fidelity_base.png`'in alt başlığında İngilizce metnin içinde tek bir
Türkçe parantez var ("egitimden once"). O dize `distill_report_base.json`'ın
`meta` alanında duruyor ve figüre oradan basılıyor; düzeltmek ya damıtmayı
yeniden koşmayı ya da makine tarafından yazılmış bir artefaktı elle düzenlemeyi
gerektirirdi. İkincisi projenin kuralına aykırı, o yüzden olduğu gibi bırakıldı.

## Ekip adları

Başlık slaydındaki ve one-pager'daki `[Team Member 1/2/3]` **yer tutucudur**;
`build_deck.py` içindeki `TEAM` sabiti düzenlenip iki betik yeniden
çalıştırılarak doldurulur.
