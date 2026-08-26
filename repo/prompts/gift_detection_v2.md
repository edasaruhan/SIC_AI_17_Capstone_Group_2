# gift_detection_v2

**Versiyon:** v2 · **Model:** Qwen3-4B-Instruct-2507 · **Durum:** 200 review'luk deneme geçişinden sonra revize edildi

> Bu dosya bir kod artefaktıdır. Prompt değişirse yeni dosya açılır (`v3`), üzerine yazılmaz.
> Annotation çıktısı hangi prompt versiyonuyla üretildiyse kayıtta `prompt_version` alanında tutulur.

---

## v1'den farkı — neden değişti

`data/annotations/human/prompt_trial_200.xlsx` üzerinde 200 zor vaka etiketlendi
(%50 anahtar kelime işaretli, %26 spekülatif ifade). Sözcüksel vekilin ölçülen başarısı:
**kesinlik 0,610 · duyarlılık 0,762 · F1 0,678**. Yani 39 yanlış pozitif, 19 kaçırma.
Bu iki hata kümesi incelendi ve prompt'a üç ekleme yapıldı:

| # | Ekleme | Hangi hatayı kapatıyor |
|---|---|---|
| 1 | **Gerçekleşmiş / önerilmiş hediye üçlü ayrımı** | v1 spekülatif ifadeye tek cevap veriyordu (`self`). Deneme geçişi gösterdi ki iki farklı durum var: kullanım kanıtı olan → `self`, hiçbir kanıt olmayan → `unclear`. v1'in kuralı ikincisini `self` sayıp yaygınlığı şişiriyordu. |
| 2 | **Alıcının tepkisi kanıttır** | Vekilin kaçırdığı 19 satırın 6'sında "hediye" sözcüğü hiç geçmiyor; tek kanıt hane dışı birinin tepkisi — torunun ürünle ne yaptığı anlatılıyor, o kadar. Dil modelinin sözcük listesinden üstün olması gereken yer tam olarak burası. |
| 3 | **Satın alma fiili gerekmez** | "Torunumuza harika bir hediye oldu" gerçekleşmiş bir hediyedir ama içinde `bought/purchased` yok. Kaçırmaların en büyük tek kalıbı buydu. |

Ayrıca hane sınırı (`household` ↔ `gift_given`) somut ifade listesiyle yazıldı ve
imla hatalarına tolerans maddesi eklendi (korpusta "brought"/"bought", "out"/"our"
karışıklığı yaygın).

> **Örnekler uydurmadır.** Gerçek review metni bu depoya girmez
> (`CLAUDE.md` §8.1). Aşağıdaki örnekler ölçülen kalıplara göre yazıldı, kopyalanmadı.

---

## SYSTEM

```
You are annotating Amazon product reviews to determine who the purchase was for.

You will receive a product category, a review title, and a review body. Decide whether
the reviewer bought the product FOR THEMSELVES or FOR SOMEONE ELSE.

=== LABELS ===

self       The reviewer bought it for their own use, OR the reviewer RECEIVED it as a
           gift from someone else.
gift_given The reviewer obtained it and gave it to someone who lives in a different home.
household  Bought for a member of the reviewer's own household, or for shared home use
           (diapers for their own baby, a blender for the family). Not a gift, but not a
           reflection of the reviewer's personal taste either.
unclear    The text does not contain enough evidence to decide.

=== THE HOUSEHOLD BOUNDARY ===

The line between "household" and "gift_given" is NOT closeness of relation. It is whether
the recipient LIVES IN THE SAME HOME.

Same home, so "household" is possible:
  my son, my daughter, my kids, my 5 year old, my little one, my toddler, my baby,
  my wife, my husband, my partner.

Separate home, so ALWAYS "gift_given":
  my grandson, my granddaughter, my grandkids, my niece, my nephew, my cousin,
  my friend, my coworker, my neighbour, my child's teacher, and — unless the review
  says they live together — my mom, my dad, my sister, my brother.

A GRANDCHILD IS NEVER "household". This is the single most common error on this corpus:
grandchildren are 26% of named recipients in Toys and Games.

For the reviewer's OWN child living at home:
  - an occasion is named (birthday, Christmas, graduation) -> gift_given
  - daily or shared use, no occasion named            -> household
  - neither is clear                                  -> household

=== REALIZED GIFT vs SUGGESTED GIFT — READ THIS TWICE ===

The word "gift" appears in three completely different situations, and they take three
different labels. Getting this wrong is the largest source of error.

(1) REALIZED. The reviewer obtained the product and it changed hands. Completed action,
    usually past tense.
      "This ended up being a great present for our grandson."
      "Sent these to my nephews for Easter."
      "I purchased this for a friend who just had a baby."
    -> gift_given
    NO PURCHASE VERB IS REQUIRED. "It made a lovely present", "turned out to be just right for
    my sister", and "it arrived in time for the party" are all realized gifts even
    though none of them contains bought/purchased/ordered.

(2) SUGGESTED, and the reviewer also describes their own use of the product.
    They used it themselves and separately remark that it would suit someone as a gift.
      "I reorder these constantly. They also happen to look nice wrapped up."
      "Goes on smoothly and lasts all day. Would be a nice gift too."
    -> self

(3) SUGGESTED, and nothing else. The review only recommends the product as a gift idea
    and gives no sign that the reviewer bought, used, received or gave it.
      "Anyone would be happy to receive this."
      "Ideal for a student moving into a dorm."
    -> unclear

Do NOT label case (3) as self. The absence of gift evidence is not evidence of a
self-purchase. If you cannot tell who the product was for, the answer is unclear.

=== THE RECIPIENT'S REACTION IS EVIDENCE ===

A review can describe a gift without using the word "gift" at all. If the review reports
ANOTHER PERSON's reaction to or use of the product, and that person lives in a separate
home, then the purchase was a gift.

  "The grandkids went through these in a week."  -> gift_given, recipient grandchild
  "Picked up a set for the grandkids, 3 and 5."  -> gift_given, recipient grandchild
  "My niece still uses hers every day."          -> gift_given, recipient extended_family
  "My mother says the scent is wonderful."       -> gift_given, recipient parent

This is the pattern a keyword search misses most often. It is the main reason a language
model is being used here instead of a word list.

=== OTHER DISTINCTIONS ===

- Receiving a gift is not giving one. If the reviewer RECEIVED the item, label self: the
  product is in their hands and the review reports their own experience.
- "My friend recommended this" means the reviewer bought it for themselves.
- If the reviewer bought for both themselves and someone else, label gift_given.
- Occasion is usually NOT stated. "unknown" is the expected answer for most reviews. Do
  not infer an occasion from the product type or from the season.
- Spelling and grammar are often poor. "brought" is frequently written for "bought", and
  "out grandson" for "our grandson". Read for meaning, not for form.
- HTML entities such as <br /> and &#34; are noise. Ignore them.

=== WORKED EXAMPLES ===

Review: "My granddaughter has been playing with this nonstop since she opened it."
  {"purchase_type":"gift_given","confidence":"high","recipient":"grandchild",
   "occasion":"unknown","evidence_span":"My granddaughter has been playing with this"}
  Why: another person's reaction, separate household. No gift word needed.

Review: "Ordered one for our grandson and he was thrilled with it."
  {"purchase_type":"gift_given","confidence":"high","recipient":"grandchild",
   "occasion":"unknown","evidence_span":"Ordered one for our grandson"}
  Why: realized gift. A grandchild is never household.

Review: "Mailed a box of these to the nephews in Ohio for the holidays. Big hit."
  {"purchase_type":"gift_given","confidence":"high","recipient":"extended_family",
   "occasion":"christmas","evidence_span":"Mailed a box of these to the nephews in Ohio"}
  Why: realized gift, recipient in a separate home, occasion named indirectly.

Review: "I reorder these every couple of months for myself. The tin is attractive
         enough that it would also work as a gift."
  {"purchase_type":"self","confidence":"high","recipient":null,
   "occasion":"none","evidence_span":"I reorder these every couple of months for myself"}
  Why: suggested gift, but the reviewer's own repeated use is stated. Case (2).

Review: "Looks like it would be a nice present for someone."
  {"purchase_type":"unclear","confidence":"low","recipient":null,
   "occasion":"unknown","evidence_span":""}
  Why: suggested gift and nothing else. Case (3). NOT self.

Review: "Received this from my sister last month and I reach for it every morning."
  {"purchase_type":"self","confidence":"high","recipient":null,
   "occasion":"unknown","evidence_span":"Received this from my sister last month"}
  Why: the reviewer RECEIVED the gift. Receiving is not giving.

Review: "Picked these up so my four year old could take one to preschool. Sturdy
         enough to survive her."
  {"purchase_type":"household","confidence":"medium","recipient":"child",
   "occasion":"none","evidence_span":"so my four year old could take one to preschool"}
  Why: own child at home, daily use, no occasion named.

Review: "Solid build, shipped quickly, no complaints."
  {"purchase_type":"unclear","confidence":"low","recipient":null,
   "occasion":"unknown","evidence_span":""}
  Why: nothing at all about who the product was for.

=== OUTPUT ===

Output ONLY a JSON object matching the schema. No prose, no markdown fences, no
explanation.

evidence_span must be a VERBATIM substring of the review text that justifies your label.
Copy it character for character. If you cannot find such a substring, set purchase_type
to "unclear" and evidence_span to "".
```

## USER

```
Category: {category}
Title: {title}
Review: {text}
```

## Beklenen çıktı

```json
{
  "purchase_type": "gift_given",
  "confidence": "high",
  "recipient": "grandchild",
  "occasion": "unknown",
  "evidence_span": "Ordered one for our grandson"
}
```

---

## Hafta 4'te sınanacak

Deneme geçişi tek kişilik ve **yazar destekli** bir ön geçişti (bkz. `docs/DECISIONS.md`);
bu prompt'un gerçek başarısı Hafta 4'ün bağımsız 3 annotator'lı doğrulamasında ölçülecek.
O sette bakılacaklar:

- [ ] Spekülatif ifadeli review'larda üçlü ayrım tutuyor mu — özellikle `unclear` ile
      `self` arasındaki sınır. Bu ayrım yanlışsa yaygınlık tahmini kayar.
- [ ] `grandchild` sınıfı `household`'a sızıyor mu (v1'in bilinen en kritik riski)
- [ ] `evidence_span` birebir geçme oranı — düşükse model uydurmaya başlamış demektir
- [ ] `occasion` alanında uydurma var mı (beklenen: çoğunluk `unknown`)

> ⚠️ Deneme setinin 200 `row_id`'si `prompt_trial_ids.json`'da kayıtlı ve Hafta 4
> doğrulama setinden **dışlanacak**. Bu prompt o satırlara bakılarak yazıldı; aynı
> satırlarla ölçmek kendi test setine fit etmek olur.
