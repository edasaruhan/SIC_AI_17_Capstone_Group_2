# This Was Not For Me — Detecting Gift Purchases as a Distinct Class of Noise in E-Commerce Recommender Systems

**Capstone Final Report**

**Team:** AI in Marketing Capstone, Group 2 — [Team Member 1] / [Team Member 2] / [Team Member 3]

**Repository:** github.com/edasaruhan/SIC_AI_17_Capstone_Group_2 · **Date:** 22 September 2026

> Every number in this report is read from a JSON file committed in
> [`repo/reports/results/`](../repo/reports/results/); the source file is named in each
> section. Thresholds, contrasts, interpretation rules and marketing-metric definitions
> were written **before the experiment was run** and are dated in
> [`repo/docs/DECISIONS.md`](../repo/docs/DECISIONS.md). Four analyses were added after the
> results were seen; all are marked *post-hoc* where they appear, and none replaces a
> primary number. A single command reproduces the headline figures from the committed
> artefacts: `python repo/scripts/demo.py`.

---

## 1. Abstract

Recommender systems treat every purchase as a statement of preference. Gift purchases
break that assumption: the buyer did not choose the item for themselves, but the model
records it as taste and shifts subsequent recommendations toward it. This project measures
that contamination end to end on public Amazon review data.

A 4-billion-parameter instruction model (Qwen3-4B-Instruct-2507) labelled 47,200 reviews
into a five-class purchase-type schema; a single human annotator independently labelled
500 of them; a distilled ModernBERT-base student then labelled **4,598,612** interactions
across two categories. Six training-set conditions — including two *placebos* that delete
the same number of **random** rows — were run through RecBole with SASRec and BPR over
three seeds, for **72 runs** in total.

The core finding is a **placebo-controlled** one. Holding the amount of deleted data fixed,
removing gift rows leaves a model that predicts the user's own next purchase **better** than
removing the same number of random rows: +15.7 % Recall@10 for SASRec and +40.0 % for BPR
on Toys and Games, against +1.2 % and a non-detectable +1.4 % on the low-gift control
category. The gap between categories is itself significant in all four comparisons — a
two-point dose–response. In the sequential model on Toys, a single gift opens about
**2 percentage points** of extra recommendation share for the gift's subcategory, and that
excess **halves after roughly one self-purchase** (≈2–4 weeks). Flagging gifts with a shadow
token instead of deleting them was worse than both alternatives in three of four cells.

The detector itself did **not** meet the two accuracy thresholds pre-registered in the
concept note (macro-F1 ≥ 0.75, gift precision ≥ 0.80); measured values were 0.5405 and 0.68.
The project proceeded with that measurement recorded. Under the pre-registered rule every
detectable placebo contrast is reported as a **lower bound** — a reading that holds if the
detector's mistakes are no more informative than random rows, which §9 shows is not
guaranteed. What the design does establish is that the rows the detector flags as gifts
carry less information than average rows.

**Results at a glance**

| Question | Answer |
|---|---|
| **RQ1** — how common are gift purchases? | **19.0 %** of Toys and Games reviews [15.8–22.3] after human calibration (raw LLM rate 23.25 %). Grocery, All Beauty and Video Games sit at 5–8 %. All four categories peak in December–January (1.58–2.08×). |
| **RQ2** — does removing gifts improve prediction of the user's own next purchase? | Gift rows are worth **measurably less than random rows** (C1 > C4): +15.7 % SASRec / +40.0 % BPR on Toys, undetectable-to-small on Grocery → **dose–response holds**. Net of doing nothing, the gain is model-dependent: BPR +4.4 %, SASRec −7.1 % on Toys. |
| **RQ3** — delete or flag? | The flagging method tested here (shadow tokens) is worse than deleting in 3 of 4 cells and worse than doing nothing in 3 of 4. |
| **RQ4** — how long does the list stay contaminated? | SASRec × Toys: excess share starts at **2.0 points** and **halves after ~1 self-purchase** (≈2.4 weeks, approximate). BPR shows dilution rather than decay. |

The experiment's validity gate (Gate 2) passed in **all four** category × model cells, so the
contrasts are interpretable.

---

## 2. Problem and motivation

Collaborative filtering and sequential recommenders learn from implicit feedback: a purchase,
a click, a review. The industry-standard assumption is that these events reveal preference.
A gift purchase satisfies every observable condition of that assumption — a verified
purchase, a real review, a genuine rating — while violating its meaning. The buyer chose the
item for somebody else.

The consequence is not a rounding error in an offline metric. It is a marketing failure with
a name every practitioner recognises: the customer who buys a toy for a nephew in December
and is shown toys until March. Every slot spent on the nephew's taste is a slot not spent on
the customer's own. Retargeting budget follows the same signal.

What makes this hard is that gift purchases are **invisible in the interaction log**. There
is no `is_gift` column in the Amazon Reviews 2023 dataset, and gift wrapping is not recorded.
The evidence exists only in free text — *"bought this for my daughter"*, *"my nephew loved
it"* — which is exactly the kind of signal a language model can read and a keyword list
cannot. A lexical proxy built in Week 1 reached a manually measured precision of **0.5833**
(n = 60): roughly two in five keyword hits were not gifts at all, and it missed the cases
that carry no gift word.

Three things had to be true for this project to say anything: the detector had to work well
enough to be worth believing, the prevalence had to be large enough to matter, and the effect
on a recommender had to survive a control that rules out "you just deleted data."

---

## 3. Research questions

| | Question | Answered in |
|---|---|---|
| **RQ1** | How prevalent are gift purchases in e-commerce interaction data, and do they show the seasonal signature a gift should show? | §7.1 |
| **RQ2** | Does removing gift interactions from the training set improve a recommender's prediction of the user's **own** next purchase, compared with removing the same number of random interactions? | §7.3 |
| **RQ3** | Is it better to delete gift interactions or to keep them and flag them for the model? | §7.4 |
| **RQ4** | How long does a single gift keep distorting the recommendation list, and what does that cost in marketing terms? | §7.5 |

**The pre-registered interpretation rule.** For placebo contrasts the rule was fixed before
any run and is applied verbatim:

- confidence interval entirely above zero → **"lower bound"** (label noise drags the estimate
  toward the placebo, so the true effect is at least this large — an argument that assumes
  the detector's false positives are as informative as random rows; see §9, item 8);
- interval contains zero → **"not detected"** — never "no effect";
- interval entirely below zero → **"negative"**.

The rule exists so that a weak detector cannot be read as evidence of absence. It was
committed on 14 September 2026, two days before the first reported run.

---

## 4. Related work

Six strands of literature frame the problem; the full review is in
[`../literature-review/literature-review.md`](../literature-review/literature-review.md).

**Implicit feedback and the preference assumption.** The foundational implicit-feedback
models (BPR, weighted matrix factorisation) treat an interaction as a positive signal of
preference with an explicit confidence weight, but the weight encodes *how sure we are that
an interaction happened*, not *whose preference it expresses*. Sequential models such as
SASRec inherit the same assumption and add another: that the order of a user's purchases
traces a single coherent intent.

**Denoising implicit feedback.** A substantial literature removes or down-weights noisy
interactions — accidental clicks, misclicks, dissatisfied purchases — usually by detecting
them from *behavioural* signals (short dwell time, low rating, early loss values). Gift
purchases are invisible to all of these: the rating is often high, the review is long and
positive, and the purchase was entirely deliberate. This is the gap the project addresses.

**Occasion- and context-aware recommendation.** Context-aware systems model *when* and *why*
a purchase happens, and some model gifting as an occasion. They generally require the
occasion to be observed (a gift-registry flag, a checkout option). Inferring it from review
text is the missing step.

**Gift exchange as a commercial phenomenon.** Consumer-behaviour research establishes that
gift purchases follow different decision rules from self-purchases — different price
sensitivity, different brand preference, different risk aversion. That is precisely why a
model that averages the two learns something that describes neither.

**Industry evidence.** Public postmortems from large retailers describe the "wrong-audience"
problem qualitatively, but there is no public measurement of how large the effect is or how
long it lasts.

**Large language models as annotators.** Recent work shows LLM annotation reaching or
approaching crowdworker agreement on subjective text-classification tasks at a fraction of
the cost, with the standard caveat that the LLM must itself be validated against human
labels on a held-out sample. That validation is Week 4 of this project — and it is where the
project's weakest result lies (§6.4).

**The gap.** No published work measures gift prevalence at scale in a public interaction
dataset, and none isolates its effect on a recommender with a data-volume control. Both are
contributions of this report.

---

## 5. Data

**Source.** [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
four categories, 16.3 GB of raw review JSONL plus product metadata.

**Preprocessing funnel** (`preprocess_funnel_*.json`). Four filters, applied in this order:
keep verified purchases · require ≥ 5 words of review text (evidence must be readable) ·
drop exact duplicates · iterate 5-core until convergence.

| Category | raw | verified | ≥5 words | dedup | **5-core** |
|---|---:|---:|---:|---:|---:|
| Toys and Games | 16,260,406 | 14,859,195 | 12,574,302 | 12,417,784 | **2,164,018** |
| Grocery and Gourmet Food | 14,318,520 | 13,176,519 | 10,940,987 | 10,774,599 | **2,434,594** |
| Video Games | 4,624,615 | 3,982,807 | 3,341,936 | 3,296,440 | 368,476 |
| All Beauty | 701,528 | 634,969 | 543,045 | 537,261 | **0** |

The last row decided the category plan. All Beauty was the original pilot; its 5-core is
**empty** — its users simply do not repeat-purchase within the category — so it cannot host a
sequential experiment. Toys and Games was promoted to the primary category and Grocery and
Gourmet Food became the low-gift control. Video Games was labelled for the prevalence
question but not run through the experiment.

Note that prevalence (RQ1) is measured on the **clean** corpus (after dedup, before 5-core)
and the experiment (RQ2–RQ4) runs on the **5-core** corpus. These are different populations
and their gift rates differ; the report keeps them separate throughout.

**Exploratory analysis.** Sixteen figures and fifteen tables
([`../data-research/data-research.md`](../data-research/data-research.md)) established the
shape of the problem before any model was built: the long tail of item popularity, the
seasonal volume cycle, and — most usefully — that reviews flagged by the lexical proxy carry
their gift evidence early in the text (median position 0.2 of the review), which is what
made a 512-token student viable.

**Ethics and privacy.** The dataset is public and used under its academic terms. Review text
can contain personal information, so no committed artefact in this repository contains
verbatim review text, a user identifier, or an absolute file path. Data files, model weights
and per-user experiment outputs are excluded from version control by design; the repository
ships the *evidence* (JSON reports, figures) rather than the data.

---

## 6. Method

### 6.1 Annotation schema and prompt

Five classes, defined so that the disputed cases have a home:

| Label | Meaning |
|---|---|
| `gift_given` | The buyer bought it for someone else. |
| `household` | Bought for shared or family use — the buyer is one of the users. |
| `received` | The reviewer received the item as a gift from someone else. |
| `self` | Bought for the buyer's own use. |
| `unclear` | The text does not say. |

Two definitions of contamination are carried through the whole project:
**C1 (narrow)** = `gift_given`, and **C1b (broad)** = `gift_given` + `household` + `received`,
i.e. "the recipient did not choose this item". Every result is reported on both axes.

The prompt (`prompts/gift_detection_v3.md`) is a code artefact: it is versioned, never
edited in place, and every annotation row records the `prompt_version` that produced it.
Three rules in it carry most of the weight and were written from failure cases on a separate
200-review development set: speculation is not evidence (*"would make a great gift"* →
`self`), receiving is not giving, and an evidence span must be quoted verbatim from the
review or the label is downgraded.

### 6.2 LLM annotation

Qwen3-4B-Instruct-2507 served with vLLM (guided JSON decoding, float16) on Kaggle's free
2 × Tesla T4, one data-parallel process per card. Float16 is not a preference: bfloat16
requires compute capability 8.0 and the T4 is Turing.

11,800 rows per category × 4 categories = **47,200 labelled reviews** in 4.07 GPU-hours.
JSON parse failure rate was 0.03–0.16 % per category (schema-constrained decoding removes
parse failure as an error class), and 1.7–3.1 % of labels were downgraded because the quoted
evidence span could not be found verbatim in the review.

The sample is not a simple random draw. Three frames were used — a proportional `main` frame
(the one prevalence is computed from), a `boost` frame over keyword hits, and a
`boost_received` frame — so that rare classes get enough rows to measure. The frames are kept
separate in every downstream statistic; pooling them would inflate the gift rate by 7.1×.

### 6.3 Gate 1 — does the detector work at all?

Four criteria, with thresholds fixed on 28 August 2026, **before** the run
(`gate1_*.json`). Verdict: **PASS (4/4) in all four categories.**

| Criterion | Threshold | Toys and Games |
|---|---|---|
| 1 · December–January rate ÷ June–September rate | ≥ 1.25 | **1.5762** [1.4344, 1.7302] ✅ |
| 2 · gift rate among reviews the keyword proxy did **not** flag | ≥ 0.01 | **0.1801** ✅ |
| 3 · parse-failure and span-downgrade rates | ≤ 0.01 / ≤ 0.10 | 0.00034 / 0.0277 ✅ |
| 4 · agreement with a 200-review manually labelled trial set | ≥ 0.70 | **0.835** ✅ |

Criterion 2 is the one that justifies the whole approach: among reviews that contain no gift
keyword at all, 18 % are still gifts by the model's reading. A free regex cannot find those.
Criterion 4 is deliberately reported as *agreement*, not F1 — the prompt was written while
reading those exact 200 rows, so an F1 computed on them would be optimistic by construction.

### 6.4 Human validation — and the gate that was missed

500 reviews were labelled blind by one annotator (A) and compared with the LLM
(`validation_500.json`, F18). The methods were committed on 14 September, before A saw any
model output.

**This is the weakest part of the project, and it is reported as such.**

- The plan called for **three** annotators and Fleiss' κ ≥ 0.60. Only one annotator was
  available. Inter-annotator reliability was therefore **never measured**, and the Week 4
  gate is recorded as **INCOMPLETE** — it was never written as PASS. A second attempt was
  made on 21 September: two more annotators were to label the same 500 rows, under rules
  committed first (κ ≥ 0.60 unchanged, every published number stays on A's labels). The two
  sheets that came back had been filled in with the help of an AI model or a script. They
  were not counted as human labels and were not used, so κ remains unmeasured.
- The concept note pre-registered two accuracy thresholds: **macro-F1 ≥ 0.75** and
  **`gift_given` precision ≥ 0.80**. Measured: **macro-F1 0.5405** [0.4897, 0.5869] and
  **precision 0.68** (0.6485 when reweighted to the population). **Both were missed.**
- The registered response to a miss was "write prompt v4" or "raise the confidence
  threshold". Neither was done. Prompt v4 would have required re-running Gate 1 from
  scratch, and the 500 validation rows could not be used to tune a prompt without destroying
  the only independent reference. The `confidence` field had already been dropped on
  29 August after measurement showed `low` ≡ `unclear` at 100 % overlap. The thresholds were
  never written into any config, so in practice they never functioned as a gate at all.
  The full reasoning is dated in DECISIONS, 20 September 2026.

The measured consequence is carried explicitly into every result: in the population, the C1
label has **precision 0.65 and recall 0.80**; C1b has precision 0.84 and recall 0.74. One in
three rows called "gift" is not one — most often an item bought for the buyer's own child,
which the schema calls `household`. Missed gifts make the treatment condition more like the
placebo and bias RQ2 effects **toward zero**; that is why the pre-registered interpretation
rule says "lower bound" and not "effect". False positives only do the same if they are as
informative as random rows — and since most of them are purchases for the buyer's own child,
that is an assumption, not a fact (§9, item 8).

### 6.5 Distillation and full-corpus inference

Labelling 4.6 million rows with a 4B model was not affordable on free-tier GPUs, so a
ModernBERT-base student (149 M parameters) was fine-tuned on 41,988 LLM-labelled rows and
evaluated against a 4,667-row held-out split. ModernBERT was chosen for inference efficiency,
not for its 8,192-token context — measurement showed a 512-token limit would lose the
evidence in only 0.013–0.044 % of gift reviews.

A fidelity gate with thresholds fixed before training (`distill_report_base.json`):

| Criterion | Threshold | Measured | |
|---|---|---|---|
| 1 · C1-axis F1 vs the teacher, held-out | ≥ 0.85 | **0.9098** | ✅ |
| 2 · C1b-axis F1 vs the teacher, held-out | ≥ 0.85 | **0.9313** | ✅ |
| 3 · C1-axis F1 vs the **human** reference, not worse than the teacher by > 0.05 | — | student **0.7415** vs teacher 0.7312 | ✅ |

**PASS (3/3).** The student is not merely close to the teacher; on the human reference it is
marginally *better* than the teacher it was distilled from. Training took 28 minutes on one
T4. Inference over the full 5-core corpora — 2,164,018 Toys rows and 2,434,594 Grocery rows,
**4,598,612** in total — ran at 330–364 rows/s.

One limitation is structural and was recorded rather than fixed: the student was trained on
the `clean` distribution but applied to the `5-core` corpus, of which only 17.6 % of the
Toys LLM labels are members. The shift was measured and reported (DECISIONS, 15 September).

### 6.6 Experimental design

Temporal leave-one-out: each user's last interaction is the test item, the one before it is
validation. **The test item is always a `self` purchase** — the question is whether the model
predicts what the user buys *for themselves*. The split, the user universe and the item
universe are frozen in C0; conditions modify **training rows only**.

| Condition | What changes in training | Toys | Grocery |
|---|---|---:|---:|
| **C0** | nothing (baseline) | 1,626,714 rows | 1,896,612 rows |
| **C1** | `gift_given` rows deleted | −401,286 (24.7 %) | −58,209 (3.1 %) |
| **C4** | the **same number of random** rows deleted — *placebo* | −401,286 | −58,209 |
| **C1b** | broad-definition rows deleted | −859,194 (52.8 %) | −143,789 (7.6 %) |
| **C4b** | the same number of random rows deleted — *placebo* | −859,194 | −143,789 |
| **C3** | gift rows kept, but the item is moved to a shadow id (`<id>::gift`) that can never be recommended | 73,683 shadow items | 25,031 shadow items |

**The placebo is the design's core.** C1 deletes a quarter of the Toys training set; any
model trained on less data will do worse. Without C4 the experiment could only say "deleting
data hurts". With C4 it can ask the question that matters: *given that you are going to lose
this much data anyway, does it matter which rows you lose?*

C2 (keeping gift rows at a reduced loss weight) was scoped out and the code raises
`NotImplementedError` rather than silently doing something else.

Models: SASRec (sequential, primary) and BPR, RecBole 1.2.0 defaults, early stopping on
validation NDCG@10, 300-epoch cap. Three seeds (42, 1337, 2024) per cell → **72 runs**,
executed across eight Kaggle sessions on 2 × T4. Evaluation is full ranking — every item in
the universe is a candidate — at Recall@10 (primary), NDCG@10, HR@10, plus @20. Test users:
Toys **117,386**, Grocery **238,756**.

Two correctness details mattered enough to be enforced in code and locked by tests: shadow
items are masked with `-inf` at evaluation time so C3 cannot be rewarded for recommending an
id that does not exist in the other conditions, and every condition inherits **C0's**
already-purchased mask so that conditions are not silently evaluated against different
candidate sets.

### 6.7 Statistics

Per-user metrics are averaged over the three seeds first, then two conditions are compared on
**the same users** with a paired bootstrap (1,000 resamples; 2,000 in Gate 2; 95 % percentile
intervals). Resampling users rather than conditions is what makes the interval a statement
about user-level variability and not about seed noise; a test in the suite fails if the
implementation ever stops pairing. The flip side is that the interval does **not** include
training randomness — a post-hoc seed-level check is in §7.3.

Every seed used anywhere in the pipeline is derived from the single configured `seed: 42`
through `zlib.crc32`, never through Python's `hash()` — `hash()` varies with
`PYTHONHASHSEED` across processes, which would quietly break the reproducibility claim. The
pinned value is asserted in a test.

---

## 7. Results

### 7.1 RQ1 — How prevalent are gift purchases?

Source: `prevalence.json`, [F19](../repo/reports/figures/F19_prevalence_by_category.png).
Measured on the `clean` corpus, `main` frame, n = 10,000 per category. "Calibrated" corrects
the LLM's rate using what each of its classes actually corresponds to in the human labels.

| Category | C1 · raw LLM | **C1 · calibrated** [95 % CI] | C1b · raw | **C1b · calibrated** [95 % CI] |
|---|---:|---|---:|---|
| Toys and Games | 23.25 % | **19.0 %** [15.8–22.3] | 44.0 % | **44.2 %** [40.4–48.4] |
| Video Games | 7.86 % | **7.9 %** [5.8–10.6] | 15.2 % | **22.7 %** [17.8–28.1] |
| Grocery and Gourmet Food | 4.28 % | **5.4 %** [3.4–8.3] | 9.6 % | **18.3 %** [13.4–24.1] |
| All Beauty | 4.70 % | **5.5 %** [3.4–8.4] | 8.8 % | **18.0 %** [12.8–23.5] |

Roughly **one in five** Toys and Games reviews describes a purchase the buyer did not make
for themselves — and on the broad definition, closer to one in two. The raw LLM rate
(23.25 %) falls outside the calibrated interval, which is the calibration doing its job: the
model over-calls items bought for the buyer's own child.

After calibration Video Games can no longer be separated from the low-gift group (the
intervals overlap), so the category ordering that survives is: Toys and Games high,
everything else low.

**Seasonality** (December + January ÷ June–September, `gate1_*.json`): Toys **1.58**
[1.43–1.73], Video Games **2.08** [1.74–2.49], Grocery **1.71** [1.33–2.23], All Beauty
**2.06** [1.61–2.63]. All four categories show the elevation a gift signal should show. The
peak is December–January rather than November–December because a review is written *after*
the purchase — the lag is visible in the data and was anticipated when the criterion was
written.

On the experiment corpus (5-core, student labels, raw), the gift share is **24.8 %** for Toys
and **3.1 %** for Grocery. That ~8× ratio is what makes the dose–response test in §7.3
possible.

### 7.2 Gate 2 — is the experiment valid?

Source: `gate2.json`. Four criteria, thresholds fixed before the runs.

| Cell | 1 · identical test pairs | 2 · universe ⊆ C0 | 3 · C4 − C0 (Recall@10, points) | 3 · C4b − C0 | 4 · C0 seed CV | **Verdict** |
|---|---|---|---|---|---|---|
| Toys · SASRec | ✅ 18 runs, one hash | ✅ | −0.851 [−0.916, −0.789] | −2.246 [−2.336, −2.163] | 0.012 | **PASS** |
| Toys · BPR | ✅ | ✅ | −0.411 [−0.457, −0.365] | −1.085 [−1.137, −1.032] | 0.025 | **PASS** |
| Grocery · SASRec | ✅ | ✅ | −0.020 [−0.051, +0.010] | −0.164 [−0.196, −0.131] | 0.005 | **PASS** |
| Grocery · BPR | ✅ | ✅ | −0.023 [−0.054, +0.005] | −0.136 [−0.165, −0.107] | 0.002 | **PASS** |

Criterion 3 requires that the placebo **does not beat** C0: deleting random rows should cost
accuracy, and a setup in which it *improves* accuracy is broken. It holds in all four cells.
Seed coefficient of variation is far below the 0.10 threshold, so the three-seed averages are
stable.

> **An erratum, stated plainly.** Four project documents — ROADMAP, PROJECT_SPEC, the concept
> note and the implementation plan — describe Gate 2 as *"C0 and C4 must not differ
> significantly"*. That is **not** the criterion that was committed two days before the runs,
> and under that older wording Toys would have **failed**: deleting 401,286 random rows from
> Toys does measurably hurt, as it should. The binding criterion was not changed after the
> results were seen; the contradiction is recorded in DECISIONS, 18 September 2026, and each
> of the four documents now carries a dated erratum.

### 7.3 RQ2 — Does removing gift rows help?

Source: `experiment_stats.json`,
[F21](../repo/reports/figures/F21_condition_contrasts.png).

**Recall@10 levels by condition** (seed mean, %):

| Cell | C0 | C1 | C4 | C1b | C4b | C3 |
|---|---:|---:|---:|---:|---:|---:|
| Toys · SASRec | 4.313 | 4.007 | 3.462 | 3.292 | 2.067 | 3.883 |
| Toys · BPR | 1.616 | 1.687 | 1.205 | 1.578 | 0.530 | 1.522 |
| Grocery · SASRec | 3.011 | 3.027 | 2.991 | 2.939 | 2.847 | 2.941 |
| Grocery · BPR | 1.446 | 1.442 | 1.422 | 1.457 | 1.310 | 1.432 |

**Primary contrast — C1 − C4** (deleting gifts vs deleting the same number of random rows):

| Cell | Recall@10 difference (points) [95 % CI] | relative | NDCG@10 [95 % CI] | Pre-registered reading |
|---|---|---:|---|---|
| **Toys · SASRec** | **+0.545** [+0.482, +0.610] | **+15.7 %** | +0.248 [+0.219, +0.278] | lower bound |
| **Toys · BPR** | **+0.482** [+0.436, +0.529] | **+40.0 %** | +0.253 [+0.225, +0.283] | lower bound |
| Grocery · SASRec | +0.036 [+0.005, +0.068] | +1.2 % | +0.018 [+0.004, +0.032] | lower bound |
| Grocery · BPR | +0.020 [−0.009, +0.049] | +1.4 % | +0.020 [+0.005, +0.035] | Recall: not detected · NDCG: lower bound |

**Reading.** For an equal amount of data loss, a model whose lost rows were *gifts* predicts
the user's own next purchase substantially better. Gift interactions therefore carry **less
information about the user's own preference than an average interaction does** — which is a
direct measurement of contamination, not an inference from it. Because C1's precision is
0.65, some deleted rows were not gifts; under the pre-registered rule these numbers are
lower bounds, with the caveat in §9 (item 8) — strictly, they measure the rows the detector
flags as gifts.

**Secondary contrast — C1 − C0** (deleting gifts vs doing nothing):

| Cell | Recall@10 [95 % CI] | relative | direction |
|---|---|---:|---|
| Toys · SASRec | −0.306 [−0.360, −0.247] | −7.1 % | **negative** |
| Toys · BPR | +0.071 [+0.027, +0.114] | +4.4 % | **positive** |
| Grocery · SASRec | +0.016 [−0.012, +0.046] | +0.5 % | contains zero |
| Grocery · BPR | −0.003 [−0.032, +0.026] | −0.2 % | contains zero |

This is the question a practitioner actually asks, and the honest answer is **it depends on
the model**. For BPR, finding and deleting gifts beats using everything — a small but
significant gain. For SASRec on Toys it does not: throwing away a quarter of the training
rows costs more than the contamination in them. Two untested explanations are consistent with
this: a sequential model extracts something from gift rows beyond preference (sequence
continuity, item co-occurrence, and rows mislabelled as gifts that are really `self`), and in
C1 the deleted gifts also disappear from the input sequence at test time, so SASRec predicts
from a shorter history.

Taken together with the placebo result: gift rows are **below-average value in both models**;
in BPR they are **net harmful**, in SASRec still **net useful**.

**Robustness — C1b − C4b** (broad definition): +1.225 [+1.144, +1.297] Toys SASRec,
+1.047 [+0.995, +1.102] Toys BPR, +0.092 [+0.061, +0.123] Grocery SASRec, +0.147
[+0.118, +0.178] Grocery BPR. The direction is identical and now detectable in **all four**
cells, including the one where the narrow definition was not.

**Dose–response.** If the effect is really caused by gifts, it should be larger where there
are more gifts. Comparing the two categories directly (`dose_response`):

| Model · contrast | Toys | Grocery | Difference [95 % CI] |
|---|---:|---:|---|
| SASRec · C1 − C4 | +0.545 | +0.036 | **+0.509** [+0.436, +0.581] |
| BPR · C1 − C4 | +0.482 | +0.020 | **+0.462** [+0.409, +0.518] |
| SASRec · C1b − C4b | +1.225 | +0.092 | **+1.133** [+1.060, +1.214] |
| BPR · C1b − C4b | +1.047 | +0.147 | **+0.900** [+0.837, +0.962] |

The effect is significantly larger in the high-gift category in all four comparisons. This is
a **two-point** dose–response: it confirms the direction, and says nothing about the shape of
the curve.

**Post-hoc robustness (defined after the results were seen; not a primary result).** Two
questions a reviewer would ask, answered from the existing runs (`robustness_posthoc.json`):

- *Is the effect larger than training noise?* The intervals above resample users, not
  training runs. Seed by seed, Toys C1 − C4 is +0.485 / +0.599 / +0.551 (SASRec) and
  +0.451 / +0.555 / +0.441 (BPR): the same sign in every seed, 13–17 times the seed-level
  standard error. In Grocery the seed spread is as large as the user interval — SASRec is
  +0.016 / +0.049 / +0.043 (positive in all three, but small), BPR −0.010 / +0.074 / −0.004
  (sign changes; already "not detected").
- *Is the placebo matched at the right level?* C4 deletes as many rows as C1, but spreads them
  over more users: it touches the training history of 73 % of Toys test users against C1's
  44 %. Splitting test users by which condition touched their own history, the Toys effect is
  present in all four groups — including users **neither** condition touched (+0.68 SASRec,
  +0.52 BPR), whose own input is identical in C1 and C4. So the Toys result is not an artefact
  of the placebo disturbing more users. In Grocery the small positive average comes from users
  C1 did not touch; among users whose gifts C1 removed, the interval contains zero.

### 7.4 RQ3 — Delete, or flag?

| Cell | C3 − C1 (Recall@10) [95 % CI] | relative | C3 − C0 [95 % CI] | relative |
|---|---|---:|---|---:|
| Toys · SASRec | −0.124 [−0.177, −0.078] | −3.1 % | −0.430 [−0.481, −0.373] | −10.0 % |
| Toys · BPR | −0.165 [−0.205, −0.122] | −9.8 % | −0.093 [−0.137, −0.054] | −5.8 % |
| Grocery · SASRec | −0.085 [−0.115, −0.054] | −2.8 % | −0.069 [−0.098, −0.038] | −2.3 % |
| Grocery · BPR | −0.010 [−0.040, +0.018] | −0.7 % | −0.014 [−0.042, +0.014] | −0.9 % |

Giving the model a shadow token for each gift is not better than deletion in any cell, and is
significantly worse in three of four — worse than doing nothing in three of four as well.

This is a result about **this flagging method**, not about the idea of signalling. A likely
mechanism (untested) is that the method creates a new, sparse identity for every gifted item
— 73,683 of them in Toys — which are learned poorly from little data and can never be
recommended. In BPR, part of the user vector is then spent explaining items the model is
forbidden to suggest. A weighting-based approach (C2) would test the idea properly and was
out of scope here.

### 7.5 RQ4 — What does it cost, and for how long?

Source: `marketing_metrics.json`,
[F22](../repo/reports/figures/F22_m2_half_life.png),
[F23](../repo/reports/figures/F23_m1_waste_share.png). Definitions were written on
14 September before any run; M2's definition was separately approved on 19 September
**without the values being shown**.

**M2 — contamination half-life.** Among test users whose last gift fell in a subcategory they
had *only ever* entered through gifts (Toys 29,577; Grocery 11,614), the excess share is the
fraction of C0's top-10 given to that subcategory minus C1's. Plotted against *n*, the number
of the user's own purchases since the gift, and fitted with `A·exp(−λn)`; half-life = ln 2 / λ.

| Cell | Initial excess A | **Half-life** (own purchases) [95 % CI] | ≈ weeks [CI] |
|---|---:|---|---|
| **Toys · SASRec** (primary) | 2.0 points | **0.66** [0.53–0.82] | **≈2.4** [1.9–3.0] |
| **Grocery · SASRec** (primary) | 1.5 points | **9.5** [5.6–23.2] | ≈86 [51–212] |
| Toys · BPR (order-blind) | 3.1 points | 43 [19–∞] | undefined |
| Grocery · BPR (order-blind) | 2.4 points | 110 [20–∞] | undefined |

In the sequential model on Toys, a single gift buys the gift's subcategory about **two extra
slots out of a hundred**, and that advantage **halves after one self-purchase** and is gone
after four — roughly two to four weeks at the observed median review gap of 25.5 days. The
week conversion is approximate by construction: a review date is not a purchase date.

BPR behaves differently in a way that is itself informative. Its excess share also falls as
*n* grows, but the pre-registered fit does not yield a bounded half-life. BPR does not see
order, so what is happening is **dilution** rather than decay: the gift is not forgotten, it
is out-voted by later purchases. In marketing terms — in classical collaborative filtering,
contamination clears with new purchases, not with time.

> **Post-hoc sensitivity (found after the results were seen).** In most n = 0 users the last
> gift *is* the validation row — the interaction immediately before the test item (Toys:
> 11,486 of 15,434, 74 %; Grocery: 2,422 of 2,688, 90 %). Conditions modify training rows
> only, so that gift is not removed in C1, and SASRec sees it as the last item of the test
> input under both conditions. The n = 0 bucket is therefore structurally unlike the others.
> Refitting on n ≥ 1 buckets only (**not pre-registered, point estimate, no CI**): Toys
> SASRec 1.08 (≈3.9 weeks), Grocery SASRec 4.2, Toys BPR 4.4, Grocery BPR 5.6. The primary
> answer for the sequential model on Toys is the same order of magnitude either way — about
> one self-purchase. Grocery and BPR are sensitive to how n = 0 is handled.

**M1 — retargeting waste share.** The fraction of top-10 slots given to subcategories the
user entered *only* through gifts, among users who have such a subcategory (Toys 37,398 of
117,386 = 32 %; Grocery 13,307 of 238,756 = 6 %).

| Cell | C0 share | C1 share | **C0 − C1** [95 % CI] | **C4 − C1** (vs placebo) | C0 − C3 |
|---|---:|---:|---|---|---|
| Toys · SASRec | 19.8 % | 18.5 % | +1.3 pts [+1.2, +1.4] | +0.4 [+0.3, +0.5] | +1.0 [+0.9, +1.1] |
| **Toys · BPR** | 16.3 % | 11.4 % | **+4.9** [+4.8, +5.0] | **+4.2** [+4.0, +4.3] | +3.0 [+2.9, +3.1] |
| Grocery · SASRec | 13.4 % | 12.0 % | +1.4 [+1.2, +1.5] | +1.2 [+1.1, +1.3] | +0.8 [+0.7, +0.9] |
| Grocery · BPR | 13.1 % | 10.7 % | +2.4 [+2.2, +2.6] | +2.2 [+2.1, +2.4] | +1.5 [+1.3, +1.7] |

A third of Toys test users have a subcategory they only ever entered as a gift-giver. BPR
spends 16 % of their top-10 there; the model that never saw the gifts spends 11 %. Roughly
**five slots in every hundred** go to an interest the gift created, and 4.2 of those points
survive the data-loss control. For SASRec the gap is small (+1.3) and mostly explained by
data loss (C4 − C1 is only +0.4).

"Waste" is an upper bound by assumption: these slots were not measured as failing to convert,
and a user may later buy in that subcategory for themselves.

**M3 — by history length.** Users split into thirds by history length (cut points 5 and 7
interactions). On Toys the C1 − C4 effect is present in every third: SASRec +0.54 / +0.55 /
+0.55, flat across the range; BPR +0.54 / +0.52 / +0.33, smaller for long histories — the
gift is plausibly diluted in a longer profile, which was not tested. On Grocery the effect is
not detectable at this resolution.

---

## 8. Discussion

**What the evidence supports.** Gift purchases are a distinct and measurable class of noise
in implicit feedback. They are common enough to matter in at least one large category
(one in five reviews, one in two on the broad definition), they carry the seasonal signature
a gift should carry, and they are *systematically less informative* than average interactions
about what the user will buy for themselves. The last claim is the one the placebo buys: it
is not "deleting data changed the metric", it is "deleting **these** rows changed the metric
differently from deleting that many random rows", and the difference scales with how many
gifts the category contains.

**What it does not support.** It does not support "delete gifts from your training set". In
the sequential model that made accuracy worse, because a quarter of the Toys training data is
a lot to give up. The evidence is consistent with **down-weighting gift rows rather than
deleting them** — but that is an inference, and the experiment that would test it (C2) is
exactly the one this project scoped out.

**Marketing implications.** Three are directly supported:

1. **Budget.** In a high-gift category and a matrix-factorisation model (BPR), roughly five
   recommendation slots in a hundred (1.3 in SASRec) chase
   an interest that a gift created, for a third of the user base. If retargeting spend is
   allocated by recommender score — the common case — that share of spend is being aimed at
   the wrong person's taste.
2. **Timing.** Contamination in a sequential model is **short-lived**: one self-purchase
   halves it. A campaign suppression rule triggered by a suspected gift does not need to run
   for a season; it needs to survive about one purchase cycle, two to four weeks in Toys.
3. **Model choice matters more than it looks.** The same contamination expresses itself as
   *decay* in a sequential model and as *dilution* in a matrix-factorisation model. A team
   deciding how to mitigate needs to know which of the two they are running; the wrong mental
   model leads to the wrong suppression window.

**On the detector's weakness.** A reader is entitled to ask why results from a detector with
macro-F1 0.54 should be believed. Two parts of the answer do not depend on the detector's
accuracy: the placebo holds the amount of deleted data fixed, and the effect scales with how
many gifts a category contains. What the weak detector changes is *what* is being measured.
Missed gifts can only dilute the effect. False positives are different: by the human
reference about a third of the rows called "gift" are not gifts, mostly purchases for the
buyer's own child — not random rows. C1 − C4 is therefore best read as the effect of
removing the rows the detector calls gifts, rather than as a guaranteed lower bound on a pure
gift effect (§9, item 8). The weak detector also forbids any claim about a specific user or
a specific row, and no such claim is made.

---

## 9. Limitations

The full list of nineteen is in [`../repo/docs/SONUCLAR.md`](../repo/docs/SONUCLAR.md) §7. The
eight that a reader should weigh first:

1. **Two pre-registered detector thresholds were missed** (macro-F1 ≥ 0.75, gift precision
   ≥ 0.80; measured 0.5405 and 0.68). The registered remedies were not applied, for the
   reasons given in §6.4 and dated in DECISIONS. The project proceeded with the measurement
   on the record.
2. **The primary metric is Recall@10, not the NDCG@10 the concept note specified.** The
   change went into the config on 14 September, before any run, but was not recorded at the
   time. It flips the reading in exactly one cell: Grocery × BPR C1 − C4 is "not detected" on
   Recall and "lower bound" on NDCG. Both metrics are printed side by side in §7.3.
3. **One human annotator; reliability never measured.** The Week 4 gate is INCOMPLETE. Every
   calibration figure and every label-quality figure in this report rests on a single
   person's judgement. A second round in September returned sheets filled in with AI or
   script help; they were discarded (§6.4).
4. **The C1 label is noisy** (population precision 0.65). Roughly a third of rows called
   "gift" are, to a human, not gifts — usually items bought for the buyer's own child. All
   RQ2 effects are reported as lower bounds under the pre-registered rule (see item 8).
5. **The gift in the validation row is never removed** (noticed after the results). Conditions
   change training rows only, so the interaction immediately before the test item stays in
   place in every condition. In Toys that row is `gift_given` for **15.6 %** of test users
   (37.0 % on the broad set); in Grocery 2.6 % / 6.5 %. SASRec therefore sees it at test time
   even in C1, which makes C1 a partial cleanup. **In which direction that moves C1 − C4 and
   C1 − C0 was not tested**: a gift in the last position may mislead the model, but deleting
   gift rows already costs SASRec accuracy (C1 − C0 < 0), so the opposite is also possible.
   (An earlier version of this report said it "shrinks" the contrasts; that was an
   assumption and has been corrected.) BPR is unaffected, as it never trains on validation
   rows. Early stopping also selects on validation targets that include gifts in every
   condition; the effect of that was not measured.
6. **The intervals exclude training randomness.** They resample users over seed-averaged
   metrics. Toys is unaffected (every seed agrees in sign, §7.3); in Grocery the seed spread is
   as large as the user interval, so the Grocery "lower bound" rests on user sampling alone.
7. **The placebo is matched in count, not per user.** C4 touches 73 % of Toys test users'
   training histories against C1's 44 %. The post-hoc split in §7.3 shows the Toys effect
   survives among users neither condition touched; a per-user-matched placebo was not run.
8. **"Lower bound" rests on an assumption.** Missed gifts pull the contrast toward the
   placebo; false positives do so only if they are as informative as random rows. Most false
   positives are purchases for the buyer's own child, and the broad-axis result suggests such
   rows are also below-average in information. C1 − C4 therefore measures the effect of the
   rows the detector flags as gifts; that a pure gift effect is at least as large is not
   guaranteed. The pre-registered label was kept; its assumption is now stated.

Also on the record: the original 72 runs did not record how many epochs they trained. The 24
seed-42 cells were therefore re-run on Kaggle on 21–22 September with unchanged data and
today's code, under rules committed before the re-run started (`reproduction_seed42.json`).
All 24 reproduced the published output **bit for bit**: per-user metrics, top-K lists, test
metrics and the test-pair hash. They trained for 21–77 epochs, every one was stopped by early
stopping, and none came near the 300-epoch cap. For the other two seeds the count is
estimated from fit time (22–81 epochs, ±10 %). No hyperparameter search was performed. The experiment covers two categories and two model families. The calibration assumes human–LLM agreement is
category-independent, which does not hold in Toys. No multiple-comparison correction was
applied; the primary contrast was pre-specified and the rest are secondary.

---

## 10. Reproducibility

Everything needed to check the claims is in the repository; nothing that could identify a
person is.

```bash
git clone https://github.com/edasaruhan/SIC_AI_17_Capstone_Group_2
cd SIC_AI_17_Capstone_Group_2/repo
python scripts/demo.py                    # every headline number, from committed JSON, no data needed

python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt && pip install -e .
python -m pytest tests -q                 # the full suite
```

| Layer | Where |
|---|---|
| Code, config, tests | `repo/src/`, `repo/configs/`, `repo/tests/` |
| Committed evidence (JSON reports) | `repo/reports/results/` |
| Figures F1–F23 | `repo/reports/figures/` |
| Results report, decisions, overview | `repo/docs/SONUCLAR.md`, `DECISIONS.md`, `GENEL_BAKIS.md` |
| Kaggle notebook cells | `repo/scripts/kaggle_*.py` |

**Provenance.** Every generated artefact records the git revision that produced it
(`code_version`), and the stamp is marked `-dirty` if the working tree held uncommitted code
— a claim that is wrong is worse than one that is absent. Experiment reports additionally
record `label_source` (`distilled` vs `proxy`), a `reportable` flag that keeps smoke-test
output out of the analysis, and a `test_pairs_sha256` that proves all 18 runs in a cell were
scored against exactly the same test pairs.

**Two environments.** The analysis environment and the RecBole environment cannot coexist:
RecBole 1.2.0 uses `np.float_`, removed in NumPy 2. `repo/.venv` runs the pipeline and the
tests; `repo/.venv-recbole` runs the experiments.

**Determinism, checked rather than asserted.** Seven cells were run twice in different Kaggle
sessions and produced **bit-identical** output. On 22 September all 24 seed-42 cells were
re-run with the current code and again reproduced the published per-user output bit for
bit. `marketing_metrics` was run twice and produced
identical output. The analysis in this report was regenerated after a refactor on
20 September and every number was unchanged; only the provenance stamps differed.

**Tests.** The suite is not a coverage exercise — it protects the specific invariants that,
if broken, would change a published number without anyone noticing. Three were verified by
deliberately breaking the code to confirm the test fails: making C4 a copy of C1, making the
paired bootstrap unpaired, and replacing the crc32 seed derivation with `hash()`. All three
mutations passed the tests that existed before the September audit.

**What is not in the repository, and why.** Raw data (~16 GB), model weights, per-user
experiment outputs, and the 500 human labels with their review text. Any of these would put
verbatim review text into a public repository.

---

## 11. Conclusion and future work

Gift purchases are a real, measurable and previously unquantified class of noise in
e-commerce implicit feedback. In Toys and Games roughly one review in five is a gift; the
interactions a detector flags as gifts carry demonstrably less information about the buyer's
own next purchase than average interactions do, in every seed and even for users whose own
history the treatment never touched; the effect scales with how gift-heavy the category is; and in a
sequential recommender the resulting distortion decays after about one self-purchase, while
in a matrix-factorisation recommender it dilutes instead. Flagging gifts with shadow tokens
was worse than both deleting and ignoring them.

The most valuable next steps are, in order:

1. **Measure annotator reliability** — two more annotators labelling the same 500 rows by
   hand, for Fleiss' κ. The September attempt did not produce human labels (§6.4). It is the
   cheapest way to strengthen every downstream claim, and it closes the one gate that is
   still INCOMPLETE.
2. **Run C2 — down-weighting instead of deleting.** The results point at it directly: gift
   rows are below-average but not worthless, and deletion throws away the part that is still
   useful. This is the experiment most likely to produce a deployable recommendation.
3. **Improve the detector and re-measure.** A prompt v4 written against the *development* set,
   or a larger annotator model, with Gate 1 re-run from scratch. A more precise detector would
   make C1 − C4 a cleaner measure of the gift effect itself (§9, item 8).
4. **A per-user-matched placebo.** Deleting, for each user, as many random rows as C1 deletes
   from that user would remove by design the confound that §7.3 could only examine post hoc.
5. **Extend the dose–response.** Two points establish direction only. Four or five categories
   spanning the gift-rate range would establish shape.
6. **Broaden the model family.** GRU4Rec, ItemKNN and a popularity baseline would show
   whether "decay in sequential, dilution in MF" is the general pattern.
7. **Cross-model agreement on the labels.** A second annotator LLM from a different lab on a
   ~5K subsample would separate model-specific bias from genuine signal.

---

## 12. AI Authorship Disclosure

This project was carried out with substantial assistance from AI coding tools. Claude (Anthropic)
was used to write and review code, to draft and edit documentation including this report, and
to audit the repository for defects. All research questions, thresholds, gates and
interpretation rules were decided by the team and committed before the corresponding
measurements were taken. Every number reported here was produced by code in this repository
and is traceable to a committed artefact; none was generated by a language model as text.
The gift labels themselves are, by design, the output of language models — Qwen3-4B-Instruct-2507
as annotator and a distilled ModernBERT-base as the production labeller — and their accuracy
is measured against human labels and reported in §6.4 rather than assumed.

---

## 13. References

1. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and
   Items for Retrieval and Recommendation.* (Amazon Reviews 2023 dataset.)
2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian
   Personalized Ranking from Implicit Feedback.* UAI.
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback
   Datasets.* ICDM.
4. Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM.
   (SASRec.)
5. Wang, W., Feng, F., He, X., Nie, L., & Chua, T.-S. (2021). *Denoising Implicit Feedback
   for Recommendation.* WSDM.
6. Zhao, W. X., et al. (2021). *RecBole: Towards a Unified, Comprehensive and Efficient
   Framework for Recommendation Algorithms.* CIKM.
7. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving
   with PagedAttention.* SOSP. (vLLM.)
8. Warner, B., et al. (2024). *Smarter, Better, Faster, Longer: A Modern Bidirectional
   Encoder.* (ModernBERT.)
9. Gilardi, F., Alizadeh, M., & Kubli, M. (2023). *ChatGPT Outperforms Crowd Workers for
   Text-Annotation Tasks.* PNAS.
10. Belk, R. W. (1979). *Gift-Giving Behavior.* Research in Marketing.
11. Sherry, J. F. (1983). *Gift Giving in Anthropological Perspective.* Journal of Consumer
    Research.
12. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.

Full annotated bibliography with the six-theme synthesis:
[`../literature-review/literature-review.md`](../literature-review/literature-review.md) §12.

---

### Companion documents

| Document | Contents |
|---|---|
| [`../concept-note/concept-note.md`](../concept-note/concept-note.md) | Problem, hypotheses, original success criteria (with errata) |
| [`../literature-review/literature-review.md`](../literature-review/literature-review.md) | Six-theme review and gap analysis |
| [`../data-research/data-research.md`](../data-research/data-research.md) | Data source, EDA, F1–F16 |
| [`../technology-review/technology-review.md`](../technology-review/technology-review.md) | Model and library choices with reasoning |
| [`../implementation-plan/implementation-plan.md`](../implementation-plan/implementation-plan.md) | Weekly plan, gates, division of work |
| [`../model-refinement/`](../model-refinement/) | Model refinement and testing (.docx) |
| [`../deployment/`](../deployment/) | Deployment assessment (.docx) |
| [`../repo/docs/SONUCLAR.md`](../repo/docs/SONUCLAR.md) | The results report this document condenses (Turkish) |
| [`../repo/docs/DECISIONS.md`](../repo/docs/DECISIONS.md) | Every dated decision and its reasoning (Turkish) |
