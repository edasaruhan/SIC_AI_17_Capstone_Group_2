# Data Research

**Project:** *This Was Not For Me — Detecting Gift Purchases as a Distinct Class of Noise in E-Commerce Recommender Systems*

**Course:** AI in Marketing Capstone
**Deliverable:** Data Research

**Team Members:** [Team Member 1] / [Team Member 2] / [Team Member 3]

> **Reproducibility.** Every number and figure in this document is produced by the code in
> `repo/`. Nothing is hand-entered. To regenerate from scratch:
>
> ```bash
> cd repo && pip install -r requirements.txt && pip install -e .
> python -m gift_contamination.data.download          --category all
> python -m gift_contamination.data.preprocess        --category all
> python -m gift_contamination.analysis.keyword_scan  --category all
> python -m gift_contamination.analysis.eda
> ```
>
> Tables are written to `repo/reports/results/eda_tables.md`, figures to
> `repo/reports/figures/`. Analysis date: 25 August 2026.

---

## 1. Introduction

### 1.1 The marketing objective

E-commerce personalisation rests on one inference: **a purchase reveals a preference.**
Recommender systems trained on implicit feedback treat every completed transaction as
positive evidence about the buyer's taste. The inference fails systematically for one class
of transaction — purchases made **for someone else**. When a customer buys a toy for a
nephew or a perfume for a parent, the system records a preference the customer does not
hold, and propagates it for weeks.

The marketing objective is to **quantify that contamination and decide what to do about
it**: filter the signal, down-weight it, or model it. The decision owner is a
personalisation or CRM lead choosing how the training signal is prepared, upstream of the
model.

### 1.2 What data is needed, and why this data

Answering the question requires a dataset with four properties simultaneously:

| Requirement | Why | Amazon Reviews 2023 |
|---|---|---|
| **Free-text customer writing** | Gift intent is disclosed in language, not in behaviour. It is not recoverable from clicks. | `text` + `title` on every record |
| **Interaction-level records with a user identifier** | The intervention is applied to the interaction graph a recommender trains on | `user_id`, `parent_asin` |
| **Timestamps** | Seasonality is a label-free validity check on the detector | `timestamp` (Unix) |
| **Public and reproducible** | The closest prior work (Wang et al., WSDM 2020) uses proprietary Etsy data no third party can verify | Open, no application process |

No public dataset carries an explicit gift flag. That absence *is* the research gap — and it
is why the detection signal must be recovered from text.

### 1.3 Link to the research questions and KPIs

| # | Research question | What the data must support | Marketing KPI |
|---|---|---|---|
| RQ1 | What share of interactions are purchases for someone else, by category and month? | Text + timestamp + category | **Contamination rate** |
| RQ2 | Does removing gift interactions improve prediction of the customer's *own* next purchase? | Per-user chronological sequences | Recall@10 / NDCG@10 on self-purchases |
| RQ3 | Remove the signal, or model it? | Per-interaction labels | Choice of intervention |
| RQ4 | How long does a gift purchase distort the slate? | Timestamps + sequences | **Contamination half-life (M2)** |

This document establishes whether the data can carry that weight. **It can — with three
specific caveats, one of which changes the project plan.**

---

## 2. Data Source and Scope

### 2.1 Source and access

**Amazon Reviews 2023** (McAuley Lab, UCSD; Hou et al., 2024) — 571.54M reviews, May 1996
to September 2023, 33 product categories.

| Property | Value |
|---|---|
| **Data type** | **Third-party public** academic dataset. Not first-party; no commercial agreement, no application process, no cost. |
| **Access method** | HuggingFace Hub, category-sharded files under `raw/review_categories/` |
| **Format** | JSONL (one JSON object per line), UTF-8 |
| **Granularity** | One row per review = one user–item interaction |
| **Time period** | May 1996 – September 2023 (observed in our four categories: 1999-03 → 2023-09) |
| **Licence / use** | Academic research use |

> **Access note — a documented method breaks in 2026.** The dataset's published quickstart
> uses `load_dataset(..., trust_remote_code=True)`. `datasets` 4.0 removed the
> `trust_remote_code` parameter and 4.5 removed loading-script support entirely; since the
> McAuley-Lab repository is script-based, that call now raises `RuntimeError`. We fetch the
> raw `.jsonl` directly with `huggingface_hub.hf_hub_download` and read it lazily with
> polars. This is faster, resumes after interruption, and removes remote code execution
> from the pipeline. Recorded in `repo/docs/DECISIONS.md`.

### 2.2 Fields used

Verified against the delivered files (`repo/reports/results/schema_probe_*.json`), not
assumed from documentation:

| Field | Type | Why it is needed |
|---|---|---|
| `text` | str | The only source of gift evidence |
| `title` | str | Short but signal-dense ("Perfect gift!") |
| `rating` | float | Tests whether gift purchases are rated differently (§4.7) |
| `timestamp` | int | **Load-bearing** — the seasonality validation depends on it |
| `user_id` | str | Builds per-user chronological sequences |
| `parent_asin` | str | Metadata join key — **not `asin`**, which varies by size/colour variant |
| `verified_purchase` | bool | Quality filter |

Two fields are present but unused: `images` and `helpful_vote`.

> **A naming discrepancy worth recording.** The project homepage documents `sort_timestamp`
> and `helpful_votes`; the HuggingFace files deliver `timestamp` and `helpful_vote`. Our
> loader asserts the expected field set on the first 1,000 rows of every download and fails
> loudly on mismatch, rather than silently reading a missing column as null.

### 2.3 Category selection — a designed spectrum

Four categories, chosen so that the study can **falsify itself**:

| Role | Category | Raw reviews | File size | Expectation |
|---|---|---|---|---|
| **High** | `Toys_and_Games` | 16,260,406 | 7.32 GB | High gift density — most toys are bought for a child |
| **Moderate** | `Video_Games` | 4,624,615 | 2.68 GB | Moderate |
| **Low (control)** | `Grocery_and_Gourmet_Food` | 14,318,520 | 5.97 GB | Low — people buy their own food |
| **Pilot** | `All_Beauty` | 701,528 | 327 MB | Pipeline testing only |

**Why the control category is the most important one.** If the detector reports the same
contamination in groceries as in toys, it is not measuring gift intent — it is measuring
something else, and every downstream result is void. The category ordering is a prediction
registered *before* measurement. §4.6 reports the outcome.

Total download: **16.3 GB**.

---

## 3. Data Quality, Privacy, and Limitations

### 3.1 Preprocessing and what each filter removes

Four quality steps, applied once, with every stage counted (T1, full table in
`repo/reports/results/eda_tables.md`):

| Category | Raw | `verified_purchase` | ≥ 5 words | Deduplicated | **5-core** |
|---|---|---|---|---|---|
| Toys and Games | 16,260,406 | 14,859,195 (91.4%) | 12,574,302 (77.3%) | 12,417,784 (76.4%) | 2,164,018 (13.3%) |
| Video Games | 4,624,615 | 3,982,807 (86.1%) | 3,341,936 (72.3%) | 3,296,440 (71.3%) | 368,476 (8.0%) |
| Grocery and Gourmet Food | 14,318,520 | 13,176,519 (92.0%) | 10,940,987 (76.4%) | 10,774,599 (75.2%) | 2,434,594 (17.0%) |
| All Beauty | 701,528 | 634,969 (90.5%) | 543,045 (77.4%) | 537,261 (76.6%) | **0 (0.0%)** |

- **Unverified purchases:** 8–14% of rows. Removed as a quality control.
- **Reviews under five words:** 14.5–17.0% of verified rows. These carry no recoverable
  gift evidence ("Great!", "Five stars") and would enter the detector as guaranteed
  `unclear`.
- **Duplicate user–item pairs:** 1.1–1.5% of remaining rows. The earliest timestamp is retained.

### 3.2 The finding that changes the plan: the review graph is extremely sparse

**T2 — corpus profile**

| Category | Interactions | Users | Items | Int./user (mean) | Median | **Users ≥ 5** | Item Gini | Median words |
|---|---|---|---|---|---|---|---|---|
| Toys and Games | 12,417,784 | 6,824,523 | 801,297 | 1.82 | 1 | **5.71%** | 0.805 | 22 |
| Video Games | 3,296,440 | 2,185,291 | 122,578 | 1.51 | 1 | **3.03%** | 0.850 | 27 |
| Grocery and Gourmet Food | 10,774,599 | 5,801,858 | 541,797 | 1.86 | 1 | **5.85%** | 0.826 | 21 |
| All Beauty | 537,261 | 503,388 | 99,296 | 1.07 | 1 | **0.08%** | 0.684 | 22 |

**The median reviewer in every category wrote exactly one review.** Reviewing is a rare act;
the review graph is not the purchase graph.

The consequence is concrete. 5-core filtering — the recommender-systems standard, and a
fixed rule in our experimental design — must be applied iteratively, because removing a
user can push an item below the threshold and vice versa. Convergence took **20 iterations**
for Toys and Games and **17** for Video Games. Its effect:

- Toys and Games, Video Games and Grocery survive, retaining 8–17% of interactions. The
  resulting corpora (e.g. 2.16M interactions / 269K users / 104K items for Toys) are
  ordinary benchmark sizes and fully adequate for the recommender experiment.
- **`All_Beauty` collapses to zero.** Only 382 of its 503,388 reviewers (0.08%) have five
  or more interactions, so iterative 5-core eliminates the entire category in three passes.

**This is a real property of the data, not a bug** — and it invalidates the pilot category
chosen in the project plan for any experimental purpose. `All_Beauty` remains usable for
detector development, where per-user sequences are irrelevant, and every detection result
below includes it. It cannot host the recommender experiment. §5 states the consequence.

Our pipeline therefore maintains **two corpora per category**: a `clean` corpus (quality
filters + deduplication) for detection and descriptive analysis, and a `kcore` corpus for
the recommender experiment. Prevalence is a question about reviews; it should not be
answered on a graph pruned for a different purpose.

### 3.3 Field-level quality

- **Timestamps** are in **milliseconds** (detected automatically from the distribution
  median, not assumed). After conversion, **zero** rows fall outside the documented
  1996–2023 window in any category.
- **Ratings.** One row in Grocery and Gourmet Food carries `rating = 0.0`, outside the 1–5
  scale — **1 record in 14.3 million**. It is trivial in volume and instructive in effect:
  it silently shifted an entire series in our first rating chart, because that chart bound
  bars to array positions rather than to rating values. It is now excluded and counted.
  A defect too small to matter statistically was large enough to produce a wrong figure.
- **Missing values.** No nulls in the fields we use, after the empty-text filter.
- **Duplicates.** Handled explicitly (§3.1).

### 3.4 Class imbalance

The target class is a minority everywhere: the lexical proxy finds 1.85%–11.07% (§4.5).
The true rate is higher, since lexical matching has poor recall (§4.8). Two consequences:

- **Annotation sampling must be stratified**, not random, or the sample will contain too
  few positives to train on. A keyword-enriched subset is drawn separately and held **out**
  of the prevalence estimate, so enrichment does not inflate the headline number.
- **Precision must be prioritised over recall** in the deployed detector. A false positive
  suppresses personalisation for a customer who made no gift purchase — an active harm — while
  a false negative merely leaves existing contamination in place.

### 3.5 Bias and representativeness

**Selection bias is the most serious limitation, and it is structural.** Reviews exist for
only a subset of purchases, and gift purchases are plausibly reviewed at a *lower* rate
than self-purchases: the buyer does not possess the item and cannot assess it.

- Our measurement therefore describes **gift purchases among reviewed transactions**, not
  among all transactions.
- The direction of the bias is most likely downward, making our estimate a plausible
  **lower bound** — but the magnitude is unknown and cannot be estimated from this data.
- This bounds the *external* claim without weakening the *internal* experiment: the
  intervention is applied to exactly the interaction graph the recommender trains on.

Two further boundaries: the estimates do not generalise beyond the four categories studied,
and reviews are predominantly English, so detector quality for other registers and dialects
is untested and is reported as a limitation rather than assumed away.

### 3.6 Privacy, consent, and responsible use

The dataset is public and pseudonymous — `user_id` is an opaque identifier with no direct
personal data attached. **The residual risk is real and specific to our method:** review
text is free-form and frequently contains personal detail *precisely because our task
depends on that detail*. A review reading "bought this for my daughter's seventh birthday"
is simultaneously our most valuable signal and a disclosure about a child.

Controls actually implemented in this repository, not merely promised:

| Control | Implementation |
|---|---|
| No verbatim review text in any published output | This document reports only aggregate statistics; illustrative cases are paraphrased |
| Human-annotation files stay local | `data/annotations/human/` is in `.gitignore`; only the aggregated result (`keyword_precision.json`) is committed |
| Evidence text never enters shared artefacts | `keyword_scan` drops `title` and `text` before writing its output |
| No re-identification | No linkage across datasets, no enrichment of pseudonymous identifiers |

**On consent.** Reviewers consented to public display of their reviews on a commercial
platform. They did not consent to inference about their household composition. Detecting
"bought for my child" is, in effect, inferring that a customer has children. We treat such
inferred attributes as **strictly instrumental**: used to *exclude* an interaction from
training, never to *target*. The distinction between suppression and targeting is the
central ethical line in this project.

### 3.7 Freshness

The data ends **September 2023**, roughly three years before this study. Absolute
contamination rates may have drifted; the structural finding (gift purchases concentrate by
category and season) is not time-sensitive. 2023 is a partial year and is treated as such
in all time series.

---

## 4. Exploratory Analysis and Marketing Insights

### 4.1 Rating distribution — the signal is almost all positive

![F1](figures/F1_rating_distribution.png)

Between 58% and 65% of reviews carry the top rating. Second place is a distant contest:
1★ leads in Grocery (14.0%) and All Beauty (15.7%), 4★ in Toys (11.2%) and Video Games
(13.5%). Either way the distribution is J-shaped and top-heavy.

> **Marketing read.** A recommender trained on this signal sees almost no negative evidence.
> That is precisely why a *structural* noise class matters more here than a random one:
> there is no counterweight in the data to correct a mislearned preference.

### 4.2 Review length — enough text to work with

![F2](figures/F2_text_length.png)

Median review length is 21–27 words. 14.5–17.0% of verified reviews fall under the
five-word threshold and are excluded.

> **Marketing read.** Roughly one in six reviewed interactions is undecidable at any cost —
> a permanent coverage ceiling on any text-based detector, and a number to state up front
> rather than discover later.

**A settled side question.** Only **0.03%–0.45%** of reviews exceed 768 words (≈1,024
tokens). The project's plan justifies its distillation model partly by its 8,192-token
context window; on this corpus, that capacity is almost never exercised, and a 1,024-token
limit truncates under half a percent of reviews. The argument for the model may stand on
efficiency grounds, but not on truncation.

### 4.3 Reviewer activity — where contamination does the most damage

![F3](figures/F3_user_activity.png)

The survival curves are steep. 94–97% of reviewers in the three main categories have fewer
than five interactions; in All Beauty, 99.92% do.

> **Marketing read.** This is the empirical basis for the sparse-profile argument. For a
> customer with three recorded interactions, a single gift purchase misdirects a third of
> the available signal. Contamination does most damage exactly where personalisation is
> already weakest — the new or low-frequency customer, who is also the most valuable to
> convert into a repeat buyer. It also sets up marketing metric **M3**, which stratifies the
> intervention's effect by interaction count.

### 4.4 Catalogue concentration — the cost of a wasted slot

![F4](figures/F4_item_longtail.png)

Item popularity is heavily concentrated (Gini 0.68–0.85). The most popular **10% of items
absorbs 74–80%** of all interactions in the three main categories (Video Games 79.6%,
Grocery 76.0%, Toys 73.5%).

> **Marketing read.** Recommendation surfaces have fixed capacity — roughly twenty homepage
> positions, six email slots, one to three retargeting placements. When attention is this
> concentrated, a slot occupied by a contaminated recommendation displaces a product with
> genuinely high conversion probability. The opportunity cost per impression is not
> hypothetical.

### 4.5 Volume over time

![F5](figures/F5_volume_over_time.png)

Review volume grows by four orders of magnitude from 2000 to its 2020–2022 peak. The usable
corpus is effectively post-2015.

> **Marketing read.** Any conclusion drawn from this data describes the recent era, which is
> the relevant one for a personalisation decision today. It also means the annotation sample
> should be drawn with month-level balance, or it will silently become a study of 2021.

### 4.6 ★ Gift prevalence by category — the falsification test

![F7](figures/F7_gift_rate_by_category.png)

**T3 — lexical gift-proxy rates**

| Category | Reviews | Gift evidence | Speculative | Received | **Proxy rate** |
|---|---|---|---|---|---|
| Toys and Games | 12,417,784 | 11.30% | 0.92% | 0.28% | **11.07%** |
| Video Games | 3,296,440 | 4.50% | 0.14% | 0.13% | **4.40%** |
| All Beauty | 537,261 | 2.18% | 0.15% | 0.08% | **2.13%** |
| Grocery and Gourmet Food | 10,774,599 | 1.89% | 0.19% | 0.08% | **1.85%** |

**The predicted ordering holds: Toys (11.07%) > Video Games (4.40%) > Grocery (1.85%).**
The high-density category shows **six times** the contamination of the control. The
prediction was registered before measurement, and the control category behaved as a control
should.

> **Marketing read.** Contamination is not a uniform tax; it is category-specific and
> plannable. In a toy catalogue, roughly one reviewed interaction in nine is a lower-bound
> estimate of "not this customer's preference." In groceries the number is small enough that
> a personalisation team can reasonably deprioritise it. **This is already an actionable
> allocation decision, before any recommender is trained.**

### 4.7 Rating distribution, gift versus rest

![F8](figures/F8_rating_gift_vs_self.png)

| Category | Gift mean | Rest mean | Difference | Gift 5★ | Rest 5★ |
|---|---|---|---|---|---|
| Toys and Games | 4.517 | 4.104 | **+0.412** | 77.8% | 63.7% |
| Video Games | 4.482 | 3.974 | **+0.508** | 77.6% | 58.6% |
| Grocery and Gourmet Food | 4.490 | 4.049 | **+0.442** | 79.0% | 64.8% |
| All Beauty | 4.542 | 3.870 | **+0.673** | 78.6% | 57.2% |

Gift-flagged purchases are rated consistently and substantially higher — by 0.41 to 0.67
stars — in all four categories.

> **Marketing read.** This behaves exactly as a distinct purchase mode should: a buyer who
> did not use the product cannot criticise it. The consistency across four independent
> categories is corroborating evidence that the flag is separating a real behavioural class
> rather than partitioning noise. It also carries a warning for anyone using average rating
> as a quality signal — in gift-dense categories, that average is partly measuring who was
> asked, not how good the product is.

### 4.8 ★ Seasonality — external validity, obtained without labels

![F6](figures/F6_gift_rate_by_month.png)

| Category | Jan | Feb | Jun–Sep (trough) | Nov | Dec | Dec ÷ summer |
|---|---|---|---|---|---|---|
| Toys and Games | 13.72% | 11.88% | 9.67% | 10.60% | 12.94% | 1.34× |
| Video Games | 6.97% | 4.97% | 3.51% | 3.42% | 5.90% | 1.68× |
| All Beauty | 3.50% | 2.35% | 1.75% | 1.98% | 3.33% | 1.90× |
| Grocery and Gourmet Food | 2.68% | 2.15% | 1.53% | 1.61% | 2.70% | 1.77× |

**The detector reads text. It has no access to the calendar.** It nonetheless reproduces the
retail gift calendar in all four categories: a December–January maximum, a distinct February
secondary peak, and a summer trough.

**The peak lands in January, and that was predicted in advance.** Review timestamps are not
purchase timestamps — people write weeks after delivery, so a December gifting season
surfaces as a December–January reviewing season. This was stated as an expectation in the
project plan before the data was touched; it is confirmation, not a defect. The February
peak (Valentine's Day) survives the same lag.

One nuance the plan did not anticipate: **seasonal amplitude runs opposite to the base
rate.** Toys has the highest level (11.07%) but the flattest curve (1.34×), because toys are
gifted year-round for birthdays. All Beauty has a low base but the sharpest holiday
concentration (1.90×). Level and seasonality are two different signals, and only the level
distinguishes categories.

> **Marketing read.** Two operational implications. **Timing:** suppression rules matter most
> in a December–February window, when up to one in seven toy interactions is a gift. And
> because the *review* peak lags the *purchase* peak, a team acting on review-derived signals
> is acting on a delay it must model. **Budget:** retargeting spend placed in January against
> categories a customer entered in December is the highest-risk spend in the calendar. This
> figure is the direct input to marketing metric **M2**, contamination half-life.

### 4.9 ★ Why a keyword scan is not enough — the case for the LLM

The proxy above is deliberately naive. This section measures how naive, because the
project's central technical claim is that this problem *requires* semantic understanding.

**T4 — manual validation, 100 stratified reviews**

| Stratum | n | Labelled `gift_given` | 95% CI | Labelled *not self* (gift + household) | 95% CI |
|---|---|---|---|---|---|
| Flagged by the proxy | 60 | 35 (**58.3%**) | 45.7–69.9% | 49 (**81.7%**) | 70.1–89.4% |
| Speculative language only | 20 | 5 (25.0%) | 11.2–46.9% | 6 (30.0%) | 14.5–51.9% |
| No keyword match at all | 20 | 1 (5.0%) | 0.9–23.6% | 1 (5.0%) | 0.9–23.6% |

> **Disclosure.** This is an **author-assisted preliminary pass**, labelled once against the
> project's four-class schema. It is *not* the three-annotator human validation with Fleiss'
> κ, which remains scheduled for Week 4 of the implementation plan and is the study's actual
> inter-rater evidence. The sample and the scoring tool are in the repository; the team can
> relabel and rescore with one command.

Three findings, each with a direct consequence:

**1. Precision is 58%.** Two in five flagged reviews are not gifts. The failure modes are
readable in the sample. Some are outright false positives: *"between this and the red cherie
scent I am in heaven… my boyfriend adores it"* matches a "my ⟨relation⟩ loved it" pattern
while describing the reviewer's own purchase. Others are near-misses the four-class schema
was designed for — 14 of the 60 are **household** purchases (bought for a spouse or child
with no gift framing), which are not gifts but do not reflect the buyer's own taste either.
Read as a "not the buyer's own preference" detector rather than a gift detector, the same
proxy reaches **81.7%**. *The schema's `household` class is doing real work, and this is the
first empirical support for keeping it.*

**2. The error is not only in the obvious places.** Our audit surfaced a specific,
diagnosable flaw: the phrase *"stocking stuffer"* sits in the evidence family, but appears
speculatively about as often as it appears factually (*"would be an excellent stocking
stuffer"*). **We have deliberately not patched it.** Tuning patterns after seeing the
validation labels would be fitting to the test set. It is recorded as a diagnosed limitation
and as an input to prompt design.

**3. Recall is the bigger problem, and we can only bound it.** One of 20 reviews with **no
keyword match whatsoever** was a genuine gift — and that single case is itself ambiguous.
With n = 20 the interval is 0.9–23.6%, far too wide to state a recall figure. But the
arithmetic is unforgiving: the unflagged stratum is ~97% of the All Beauty corpus, so even
the *bottom* of that interval implies the lexical scan misses more gifts than it finds. This
is exactly the case the project plan named in advance — *"bought it and my daughter loved
it"* contains no gift vocabulary at all.

Finally, the three pattern families quantify a cheaper mistake. Collapsing them into a
single `gift` regex — the obvious first implementation — inflates the reported rate by
**1.05× to 1.12×** before any precision loss is counted.

> **Marketing read, and the decision this section forces.** A 58%-precision detector deployed
> as a suppression rule would wrongly suppress personalisation for roughly two customers in
> five that it touches. **That is worse than doing nothing**, because it degrades the
> experience of customers who made no gift purchase while claiming to improve it. A lexical
> rule is adequate to *size* the problem — which is what this document uses it for — and
> inadequate to *act* on it. The step to a semantic model is not a technical preference; it
> is the difference between a measurement and a deployable control.

---

## 5. Conclusion

### 5.1 Is the data suitable for the next stage?

**Yes for detection, yes for the experiment in three of four categories, and no for the
pilot category as an experimental site.**

| Requirement | Verdict | Evidence |
|---|---|---|
| Enough text to detect gift intent | **Yes** | Median 21–27 words; 83–86% of verified reviews clear the 5-word floor |
| Gift signal actually present | **Yes** | 1.85%–11.07% by lexical lower bound, with 6× spread across the designed spectrum |
| Detector separates a real class | **Yes — three independent checks** | Category ordering as predicted (§4.6); seasonality without calendar access (§4.8); distinct rating distribution (§4.7) |
| Timestamps usable for seasonality | **Yes** | Millisecond units confirmed; zero out-of-range rows |
| Sequences usable for sequential recommendation | **Yes, in 3 of 4** | 5-core retains 0.37M–2.43M interactions in Toys, Video Games and Grocery — ordinary benchmark scale |
| Pilot category usable for the experiment | **No** | `All_Beauty` retains 0 interactions after 5-core; 0.08% of its reviewers reach the threshold |
| Lexical detection sufficient on its own | **No** | 58% precision; recall bounded but clearly poor |

### 5.2 What this changes in the project plan

1. **`All_Beauty` is retired as the experimental pilot** and retained as the detector-development
   pilot, where its 537K reviews and 32-second download remain ideal. The recommender
   experiment should pilot on **Video Games** — at 368K interactions after 5-core it is the
   smallest of the three viable categories and the fastest to iterate on.
2. **Two corpora per category are now standard** in the pipeline: `clean` for detection and
   descriptive analysis, `kcore` for the experiment. The experimental rule that the user/item
   universe is frozen once, on the baseline condition, applies to `kcore` and is unaffected.
3. **The `household` class survives its first empirical test** (§4.9) and should be retained
   into the annotation schema rather than merged pre-emptively.
4. **The distillation context-length argument needs restating** — on this corpus, under 0.5%
   of reviews would be truncated at 1,024 tokens (§4.2).

### 5.3 What the data already supports

Before a single model is trained, the data yields a marketing-actionable result: **gift
contamination in reviewed Amazon interactions ranges from under 2% in groceries to at least
11% in toys, rises by a third to a half in the December–February window, and concentrates in
customer profiles too sparse to absorb it.** Every one of those numbers is a lower bound.

That is enough to prioritise where a personalisation team should look. It is not enough to
tell them what to do about it — which is the question the recommender experiment exists to
answer, and which requires a detector considerably better than the one used here.

---

## 6. References

**Dataset**

1. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952 · https://amazon-reviews-2023.github.io/ · https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

**Recommender systems and evaluation**

2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI 2009*, 452–461.
3. Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM 2018*, 197–206. https://doi.org/10.1109/ICDM.2018.00035
4. Krichene, W., & Rendle, S. (2020). On Sampled Metrics for Item Recommendation. *KDD '20*, 1748–1757.
5. Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. *RecSys '19*, 101–109. https://doi.org/10.1145/3298689.3347058

**Denoising and occasion-aware recommendation**

6. Wang, W., Feng, F., He, X., Nie, L., & Chua, T.-S. (2021). Denoising Implicit Feedback for Recommendation. *WSDM '21*, 373–381. https://doi.org/10.1145/3437963.3441800
7. Wang, J., Louca, R., Hu, D., Cellier, C., Caverlee, J., & Hong, L. (2020). Time to Shop for Valentine's Day: Shopping Occasions and Sequential Recommendation in E-commerce. *WSDM '20*, 645–653. https://doi.org/10.1145/3336191.3371836

**LLM annotation and validation methodology**

8. Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*.
9. Pangakis, N., Wolken, S., & Fasching, N. (2023). Automated Annotation with Generative AI Requires Validation. arXiv preprint.
10. Calderon, N., Reichart, R., & Dror, R. (2025). The Alternative Annotator Test for LLM-as-a-Judge. *ACL 2025*, 16051–16081.

**Industry context**

11. Amazon Technologies, Inc. US Patents 9,818,145; 10,445,809; 8,352,331; 11,367,117 — cited as documentary evidence of industry problem recognition, not of measured effect.
12. Gift recommendation systems: a review. (2023). *Electronic Commerce Research*. https://doi.org/10.1007/s10660-023-09790-6

**Tooling**

13. polars — DataFrame library. https://pola.rs/
14. HuggingFace `huggingface_hub` documentation. https://huggingface.co/docs/huggingface_hub
15. Dataset scripts deprecation, `datasets` 4.x. https://github.com/huggingface/datasets/issues/7693

---

## Appendix — reproducing this document

| Artefact | Path |
|---|---|
| Pipeline code | `repo/src/gift_contamination/` |
| Configuration (single source of truth) | `repo/configs/base.yaml`, `repo/configs/gift_keywords.yaml` |
| Preprocessing funnel counts | `repo/reports/results/preprocess_funnel_*.json` |
| Lexical proxy rates | `repo/reports/results/keyword_rates_*.json` |
| Manual validation result | `repo/reports/results/keyword_precision.json` |
| All tables (T1–T4) | `repo/reports/results/eda_tables.md` |
| All figures (F1–F8) | `repo/reports/figures/` and `figures/` |
| Interactive walkthrough | `repo/notebooks/01_data_research_eda.ipynb` |
| Deviations from the project specification | `repo/docs/DECISIONS.md` |
| Data-integrity tests | `repo/tests/` — `pytest tests -q` |

> **AI authorship disclosure.** In line with the disclosure in the Implementation Plan, this
> document was drafted with substantial assistance from a large language model (Claude,
> Anthropic). All figures and statistics are produced by the team's own code from the raw
> dataset and are reproducible with the commands above; the T4 labelling is disclosed as an
> author-assisted preliminary pass in §4.9.
