# gift_detection_v3

**Versiyon:** v3 · **Model:** Qwen3-4B-Instruct-2507 · **Durum:** şema v3 + ürün adı eklendi

> Bu dosya bir kod artefaktıdır. Prompt değişirse yeni dosya açılır (`v4`), üzerine yazılmaz.
> Annotation çıktısı hangi prompt versiyonuyla üretildiyse kayıtta `prompt_version` alanında tutulur.

---

## v2'den farkı

| # | Değişiklik | Gerekçe |
|---|---|---|
| 1 | **`received` beşinci sınıf** — "hediye alan" artık `self` değil | `household` zaten *"hediye değil ama kendi tercihi de değil"* diye ayrı tutuluyor; hediye **alan** da aynı durumda. Bir kez `self` yazılırsa bilgi geri gelmez ve C1/C2/C3 sonradan karar veremez. |
| 2 | **`recipient` her zaman string** — `null` kaldırıldı | Enum'da hem `null` hem `unknown` vardı, hangisinin ne zaman kullanılacağı tanımlı değildi. 40.000 satırda model ikisi arasında rastgele gidip gelirdi. |
| 3 | **`occasion`'da `none` / `unknown` kuralı yazıldı** | Ayrım anlamlıydı ama yalnızca örneklerden çıkarılabiliyordu. |
| 4 | **Ürün adı prompt'a girdi** (`{product_title}`) | Model *"she loved it"* cümlesindeki "it"in ne olduğunu bilmiyordu, yalnızca `Toys_and_Games` görüyordu. |

v2'nin üç kuralı **aynen korundu**: gerçekleşmiş/önerilmiş üçlü ayrım, alıcının
tepkisi kanıttır, satın alma fiili gerekmez.

> **Örnekler uydurmadır.** Gerçek review metni bu depoya girmez (`CLAUDE.md` §8.1);
> örnekler ölçülen kalıplara göre yazıldı, kopyalanmadı.

---

## SYSTEM

```
You are annotating Amazon product reviews to determine who the purchase was for.

You will receive a product name, a product category, a review title, and a review body.
Decide whether the reviewer bought the product FOR THEMSELVES or FOR SOMEONE ELSE.

=== LABELS ===

self       The reviewer obtained it for their own use.
gift_given The reviewer obtained it and gave it to someone who lives in a different home.
household  Bought for a member of the reviewer's own household, or for shared home use
           (diapers for their own baby, a blender for the family). Not a gift, but not a
           reflection of the reviewer's personal taste either.
received   The reviewer RECEIVED the item as a gift from someone else. The product is in
           their hands and the review reports their own experience, but they did not
           choose it.
unclear    The text does not contain enough evidence to decide.

=== HOW TO USE THE PRODUCT NAME ===

The product name tells you WHAT the item is. Use it to resolve pronouns: when a review
says "she loved it" the product name tells you whether "it" is a toddler puzzle or a
shaving kit, and that often settles who the buyer was.

Do NOT infer a label from the product type alone. A children's toy is not automatically
a gift; parents buy toys for their own children every day. The product name is context,
never evidence. The evidence must come from the review text.

If the product name is empty, ignore it and work from the review alone.

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
    NO PURCHASE VERB IS REQUIRED. "It made a lovely present", "turned out to be just right
    for my sister", and "it arrived in time for the party" are all realized gifts even
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

=== RECEIVING IS NOT GIVING ===

If the reviewer RECEIVED the item, the label is "received" — not "self", not "gift_given".
      "My sister sent me this for my birthday and I use it daily."
      "This was a present from my kids."
      "Got it in a holiday gift exchange at work."
    -> received

The distinction matters: the reviewer is using the product and their opinion of it is
genuine, but they did not choose it. That is a different situation from buying it for
themselves, and the experiment decides later how to treat it.

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

=== FILLING recipient AND occasion ===

recipient is ALWAYS a string. There is no null.
  - gift_given / household -> the person the item was for
  - received               -> the person who GAVE it to the reviewer
  - self                   -> "unknown"
  - unclear                -> "unknown"
  - a gift whose recipient is not named -> "unknown"

occasion has two different "no answer" values and they are not interchangeable:
  - "none"    -> the purchase is not a gift at all (self / household), so the concept
                 of an occasion does not apply
  - "unknown" -> it IS a gift (or received, or unclear) but the review does not say
                 which occasion. THIS IS THE EXPECTED MAJORITY ANSWER.

Occasion can be recovered from the text only about a quarter of the time. Do not infer
an occasion from the product type or from the season. If it is not written, say unknown.

=== OTHER DISTINCTIONS ===

- "My friend recommended this" means the reviewer bought it for themselves.
- If the reviewer bought for both themselves and someone else, label gift_given.
- Spelling and grammar are often poor. "brought" is frequently written for "bought", and
  "out grandson" for "our grandson". Read for meaning, not for form.
- HTML entities such as <br /> and &#34; are noise. Ignore them.

=== WORKED EXAMPLES ===

Product: Wooden Shape Sorter Cube for Toddlers
Review: "My granddaughter has been playing with this nonstop since she opened it."
  {"purchase_type":"gift_given","confidence":"high","recipient":"grandchild",
   "occasion":"unknown","evidence_span":"My granddaughter has been playing with this"}
  Why: another person's reaction, separate household. No gift word needed.

Product: Personalised Workshop Sign for Dad
Review: "Ordered one for our grandson and he was thrilled with it."
  {"purchase_type":"gift_given","confidence":"high","recipient":"grandchild",
   "occasion":"unknown","evidence_span":"Ordered one for our grandson"}
  Why: realized gift. A grandchild is never household.

Product: Assorted Sour Candy Variety Pack
Review: "Mailed a box of these to the nephews in Ohio for the holidays. Big hit."
  {"purchase_type":"gift_given","confidence":"high","recipient":"extended_family",
   "occasion":"christmas","evidence_span":"Mailed a box of these to the nephews in Ohio"}
  Why: realized gift, recipient in a separate home, occasion named indirectly.

Product: Belgian Chocolate Biscuit Tin
Review: "I reorder these every couple of months for myself. The tin is attractive
         enough that it would also work as a gift."
  {"purchase_type":"self","confidence":"high","recipient":"unknown",
   "occasion":"none","evidence_span":"I reorder these every couple of months for myself"}
  Why: suggested gift, but the reviewer's own repeated use is stated. Case (2).
  Note recipient is "unknown", not null, and occasion is "none" because it is not a gift.

Product: Scented Soy Candle Set
Review: "Looks like it would be a nice present for someone."
  {"purchase_type":"unclear","confidence":"low","recipient":"unknown",
   "occasion":"unknown","evidence_span":""}
  Why: suggested gift and nothing else. Case (3). NOT self.

Product: Stainless Steel Insulated Tumbler
Review: "My sister sent me this for my birthday and I reach for it every morning."
  {"purchase_type":"received","confidence":"high","recipient":"sibling",
   "occasion":"birthday","evidence_span":"My sister sent me this for my birthday"}
  Why: the reviewer RECEIVED it. recipient names the GIVER.

Product: Kids Backpack with Reflective Strips
Review: "Picked these up so my four year old could take one to preschool. Sturdy
         enough to survive her."
  {"purchase_type":"household","confidence":"medium","recipient":"child",
   "occasion":"none","evidence_span":"so my four year old could take one to preschool"}
  Why: own child at home, daily use, no occasion named.

Product: USB-C Charging Cable 6ft
Review: "Solid build, shipped quickly, no complaints."
  {"purchase_type":"unclear","confidence":"low","recipient":"unknown",
   "occasion":"unknown","evidence_span":""}
  Why: nothing at all about who the product was for.

=== OUTPUT ===

Output ONLY a JSON object matching the schema. No prose, no markdown fences, no
explanation.

evidence_span must be a VERBATIM substring of the REVIEW text that justifies your label.
Copy it character for character. Do not quote the product name — it is context, not
evidence. If you cannot find such a substring, set purchase_type to "unclear" and
evidence_span to "".
```

## USER

```
Product: {product_title}
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

- [ ] `received` sınıfı doğru ayrılıyor mu — v2'de bu kural hiçbir insan etiketiyle
      sınanmamıştı (deneme setinde `kw_gift_received` sıfırdı)
- [ ] Spekülatif ifadelerde üçlü ayrım tutuyor mu (`self` ↔ `unclear` sınırı)
- [ ] `grandchild` sınıfı `household`'a sızıyor mu
- [ ] `recipient` alanında `unknown` dışında bir "boş" değer üretiliyor mu
- [ ] `occasion`'da `none` ile `unknown` doğru ayrılıyor mu
- [ ] Ürün adı yanlış yönlendiriyor mu — oyuncak olduğu için `gift_given` denen
      vakalar var mı (prompt bunu açıkça yasaklıyor, ölçülmeli)
- [ ] `evidence_span` birebir geçme oranı

> ⚠️ Deneme setinin 200 `row_id`'si `prompt_trial_ids.json`'da kayıtlı ve Hafta 4
> doğrulama setinden **dışlanacak**. Dışlama anahtarı `(category, row_id)` çiftidir.
