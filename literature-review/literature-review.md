# Literature Review

**Project:** *Gift Purchases as a Distinct Class of Noise in Implicit-Feedback Recommender Systems*

**Course:** AI in Marketing Capstone
**Deliverable:** Literature Review

---

## 1. Introduction

### 1.1 The marketing problem

Modern e-commerce personalisation rests on a single, rarely questioned inference: **that a purchase reveals a preference.** Recommender systems trained on implicit feedback treat every completed transaction as a positive signal about the buyer's taste, and they build the customer's profile from the accumulation of those signals. The assumption is operationally convenient and mostly correct.

It is systematically wrong for one category of transaction: purchases made **for someone else**.

When a customer buys a toy for a nephew, a perfume for a parent, or a coffee machine for an office gift exchange, the system records a preference that the customer does not hold. The recommender then propagates that false signal forward — surfacing children's toys to a childless adult, cosmetics to someone who never wears them — for weeks or months after the transaction. Every marketer has seen this failure; almost none measure it.

The consequence is not merely an aesthetic defect in the customer experience. It is a **measurable misallocation of finite personalisation inventory**:

- **Slot displacement.** Recommendation surfaces have fixed capacity — roughly twenty positions on a homepage carousel set, six in a marketing email, one to three in a retargeting placement. A slot occupied by a contaminated recommendation is a slot denied to a product the customer might actually buy. The cost is an opportunity cost, incurred silently on every impression.
- **Retargeting waste.** In paid retargeting the cost is not merely opportunity cost but cash. The advertiser pays for an impression, the platform collects revenue for it, and the impression is aimed at a preference that never existed.
- **Disproportionate damage in sparse profiles.** For a customer with five recorded interactions, a single gift purchase misdirects twenty per cent of the available signal. Contamination is therefore most harmful precisely where personalisation is already weakest — the new or low-frequency customer, who is also the customer most valuable to convert into a repeat buyer.

Two properties distinguish this noise from the noise types that the recommender-systems literature already addresses. First, it is **structural rather than random**: it concentrates in particular product categories (toys, jewellery, books, gourmet food) and in particular calendar windows (November–December, mid-February, early May). Random noise attenuates in aggregate; structured noise does not. Second, and more usefully, it is **observable in text**. Buyers routinely disclose the recipient in their own reviews — "bought this for my daughter's birthday," "my wife loved it." Unlike a mis-click or an unobserved dissatisfaction, gift intent leaves a linguistic trace.

### 1.2 Marketing context

| Dimension | Specification |
|---|---|
| **Target customer** | Existing e-commerce customers with at least one prior purchase and a recorded review history |
| **Customer journey stage** | Retention and repeat purchase — not acquisition |
| **Channel** | Owned recommendation surfaces (homepage, product detail page, lifecycle email) and paid retargeting |
| **Campaign context** | Always-on personalisation rather than a discrete campaign |
| **Marketing outcome to improve** | Relevance of the recommendation slate to the customer's *own* subsequent purchases; reduction of wasted personalisation inventory |
| **Proxy KPIs** | Top-*N* recommendation accuracy on non-gift held-out purchases; share of recommendation slots attributable to gift-only categories; duration of category contamination following a gift purchase |

### 1.3 Research questions

- **RQ1.** What proportion of recorded e-commerce interactions are purchases made for someone else, and how does this proportion vary by product category and by month?
- **RQ2.** Does excluding or down-weighting gift interactions during training improve a recommender's ability to predict the customer's *own* subsequent purchase? Does the effect size scale with the gift density of the category?
- **RQ3.** Is it better to **remove** the gift signal or to **model** it as an additional input?
- **RQ4.** How long does a single gift purchase continue to distort the recommendation slate?

RQ2 is deliberately posed as a question of *sensitivity* rather than *improvement*. A finding that modern sequential architectures are robust to this contamination answers the question as completely as a finding that they are not, and the review below establishes why that framing is defensible.

### 1.4 Why this review is necessary

The project sits at the intersection of four literatures that rarely cite one another: the technical literature on denoising implicit feedback; the literature on occasion- and context-aware recommendation; the consumer-behaviour literature on gift exchange; and the recent methodological literature on using large language models as annotators. Each has addressed a component of the problem. None has addressed the whole. A review is necessary to establish precisely where the boundary of existing knowledge lies — and, equally important, to avoid overstating a novelty claim that the industrial record does not support. As Section 7 documents, the problem is already well known inside at least one major retailer; what is absent is public measurement.

---

## 2. Scope and Organisation

The review is organised **thematically** rather than chronologically, because the relevant work is contemporaneous across strands and the intellectual structure of the problem — not its chronology — determines where the gap lies.

| § | Theme | Role in the argument |
|---|---|---|
| 3 | Implicit feedback and the preference assumption | Establishes the assumption the project challenges |
| 4 | Denoising implicit feedback | The nearest technical literature; defines what "noise" currently means |
| 5 | Occasion- and context-aware recommendation | The nearest conceptual literature; contains the closest prior work |
| 6 | Gift exchange as a commercial phenomenon | Establishes commercial magnitude and behavioural mechanism |
| 7 | Industry evidence | Demonstrates practitioner awareness and the limits of current solutions |
| 8 | LLMs as annotators | The enabling method; establishes feasibility and required safeguards |

Sources were selected for peer-reviewed venue quality (WSDM, SIGIR, KDD, WWW, RecSys, ICDM, ACL, PNAS), direct topical relevance, and — for the industry strand — documentary verifiability.

---

## 3. Theme 1 — Implicit Feedback and the Preference Assumption

Contemporary recommender systems descend from two foundational lines of work. Item-based collaborative filtering (Sarwar et al., 2001) and its industrial realisation at scale (Linden, Smith & York, 2003) established co-occurrence between items as the primary personalisation signal. Bayesian Personalised Ranking (Rendle et al., 2009) then formalised learning from implicit feedback as a pairwise ranking problem, in which an observed interaction is assumed preferred to an unobserved one.

BPR's contribution was to make explicit an assumption that had been operating tacitly: *observed interaction implies positive preference relative to non-interaction.* This assumption is what allows the enormous volume of implicit data to substitute for scarce explicit ratings, and it is the reason implicit feedback became the default training signal in industry.

Sequential recommendation inherits the assumption without modifying it. GRU4Rec (Hidasi et al., 2016) introduced recurrent modelling of interaction sequences; SASRec (Kang & McAuley, 2018) replaced recurrence with self-attention, allowing the model to weight arbitrary earlier interactions when predicting the next one; BERT4Rec (Sun et al., 2019) applied bidirectional masked-item training. Each advance improved how effectively the model *extracts* preference from the interaction sequence. None questions whether every element of the sequence *encodes* preference.

**Synthesis.** The field's progress over two decades has been in representation and optimisation, taking signal validity as given. A purchase made for another person is, under every model in this lineage, indistinguishable from a purchase made for oneself. This is the assumption the present project tests.

**Marketing contribution of this theme.** It explains why the failure mode is invisible to standard practice: no offline metric computed on contaminated data will reveal contamination, because the contaminated interaction appears in both the training signal and the evaluation target.

---

## 4. Theme 2 — Denoising Implicit Feedback

A substantial and active literature does recognise that implicit feedback is imperfect.

**Wang, Feng, He, Nie & Chua (2021)** — the anchor paper — observe that noisy interactions tend to exhibit large loss values in the early stages of training, and exploit this to propose **Adaptive Denoising Training (ADT)**, comprising Truncated Loss (discarding large-loss samples above a dynamic threshold) and Reweighted Loss (adaptively down-weighting them). Evaluated on three benchmarks with three recommender backbones, ADT substantially improves recommendation quality over standard training. A journal extension (Wang et al., 2021b) adds strategies for incorporating auxiliary explicit feedback such as ratings.

Critically for the present project, the paper's own definition of noise is explicit: in e-commerce, *a large portion of clicks do not translate into purchases, and many purchases end in negative reviews.* Noise is thus conceptualised as (a) engagement without conversion and (b) **post-hoc dissatisfaction**.

Subsequent work extends the mechanism without altering the conceptualisation:

| Work | Mechanism | Noise conception |
|---|---|---|
| Yu & Qin (2020), SIGIR | Noisy-label-robust sampler design | Mislabelled implicit positives |
| Wang, Z. et al. (2021), ACM MM | Iterative relabelling for one-class CF | Unfavourable interactions |
| Qin, Wang & Li (2021), SIGIR | Contrastive learning for next-basket denoising | Spurious basket co-occurrence |
| Wang, Y. et al. (2022), WWW | Cross-model agreement to identify clean samples | Disagreement between models signals noise |
| Tian et al. (2022), SIGIR | Denoising unreliable interactions in graph CF | Unreliable edges |
| Chua et al. (2024), RecSys | Unified denoising training | Consolidation of prior paradigms |
| Zhao et al. (2024), SIGIR | Diffusion-based denoising | Generative reconstruction of clean signal |

**Synthesis and critical assessment.** The methodological sophistication of this literature is considerable, but three shared properties are decisive for the present project.

First, **noise is treated as statistically detectable but semantically opaque**. ADT and its successors identify noise by its behaviour under optimisation — high loss, model disagreement, reconstruction error — never by what the interaction *means*. The methods can flag an interaction as anomalous; they cannot say why.

Second, **the taxonomy of noise is inherited rather than derived**. Across the strand, the canonical examples remain mis-clicks, popularity bias, and dissatisfaction. No paper in this line proposes "purchased on behalf of another person" as a category. This is not an oversight so much as a consequence of the first property: a noise class that cannot be named from behavioural data alone will not enter a taxonomy built from behavioural data alone.

Third — and this is the sharpest limitation for a marketing application — **loss-based detection cannot distinguish a gift purchase from a dissatisfying purchase.** Both may produce anomalous loss. But they demand opposite marketing responses. A dissatisfying purchase indicates the customer's taste was correctly inferred and the product failed; a gift purchase indicates the taste was never the customer's at all. Conflating them means a marketer cannot act on either.

**Marketing contribution.** This literature supplies the intervention machinery the present project will use — truncation and reweighting are precisely the C1 and C2 conditions in our experimental design — while leaving the *identification* of the noise class open.

---

## 5. Theme 3 — Occasion- and Context-Aware Recommendation

The closest prior work to this project is **Wang, Louca, Hu, Cellier, Caverlee & Hong (2020)**, presented at WSDM 2020 by a Texas A&M–Etsy collaboration.

**Objective.** The paper begins from an observation nearly identical to ours: sequence-based recommenders capture either long-term intrinsic preference or immediate current need, but in e-commerce, *intrinsic user behaviour may be shifted by occasions such as birthdays, anniversaries, or gifting celebrations (Valentine's Day, Mother's Day), producing purchases that deviate from long-term preferences and are unrelated to recent actions.*

**Data.** Proprietary Etsy transaction logs — a platform where gift purchasing is unusually prevalent.

**Method.** An occasion-aware sequential recommendation framework that models occasion signals explicitly alongside intrinsic preference, rather than treating all interactions homogeneously. Code accompanying the paper is publicly available.

**Findings and contribution.** Incorporating occasion signals improves next-item prediction relative to occasion-agnostic sequential baselines, establishing empirically that occasion-driven deviation is real and consequential at commercial scale.

**Where the present project diverges.** Three differences are substantive rather than cosmetic.

1. **Opposite objective.** Wang et al. model the deviation in order to *predict it better* — to anticipate what a customer will buy for Valentine's Day. We treat the deviation as *contamination to be removed* so that the customer's own preference is estimated more cleanly. Their success metric is accuracy on the occasion purchase; ours is accuracy on the customer's subsequent purchase for themselves. Both are legitimate; they are not the same problem, and no published work compares them.

2. **Calendar versus text as the detection signal.** Wang et al. derive occasion from temporal structure. This is powerful for calendar-anchored occasions but structurally blind to the much larger class of gift purchases that are not calendar-anchored — a nephew's birthday in March, a housewarming in August, a get-well gift at any time. Textual detection has no such restriction. The two signals are complementary rather than competing, and their relative coverage has not been quantified.

3. **Proprietary versus public data.** The Etsy analysis cannot be replicated, extended, or contested by anyone outside Etsy. This is a real constraint on cumulative knowledge in the area, and one that a public-data study directly addresses.

Adjacent work on temporal and promotional context — for example frequency-domain modelling of occasion evolution for promotion-aware CTR prediction — confirms that occasion-driven behavioural shift is now an established concern in industrial recommendation, but shares the calendar-signal orientation.

**Synthesis.** The occasion literature has established that the phenomenon exists and that modelling it helps prediction. It has not asked whether *removing* it helps a different and equally important objective, nor has it detected occasions from what customers say rather than when they buy.

---

## 6. Theme 4 — Gift Exchange as a Commercial Phenomenon

A recent review in *Electronic Commerce Research* (2023) surveys gift recommendation systems and, in doing so, establishes two facts the technical literature generally omits.

First, on **commercial magnitude**: a significant share of worldwide e-commerce sales is attributable to gift purchases. If even a moderate fraction of transactions are gifts, then the contaminated-signal problem is not a curiosity at the margin of the data but a routine occurrence at its centre.

Second, on **mechanism**: the review synthesises psychological, marketing, and anthropological research on gift exchange, noting that gift-givers frequently struggle to predict recipients' reactions. This matters for our project in a specific way — it implies gift purchases are made under greater uncertainty and with weaker preference grounding than self-purchases, reinforcing the argument that they are poor evidence of the buyer's taste.

The review's own orientation, however, is the **inverse** of ours: it proposes a framework for helping the giver *choose* a gift. A CHI 2024 extended abstract on gift selection under sparse recipient-preference data shares this orientation, framing gift-giving as a cold-start problem on the recipient rather than a contamination problem on the giver.

**Synthesis.** The gift literature establishes that gift purchasing is commercially large, behaviourally distinct, and cognitively difficult — and then treats it exclusively as a *recommendation opportunity*. That gift purchases might simultaneously constitute a *measurement liability* for the platform is not considered in this strand.

---

## 7. Theme 5 — Industry Evidence

Peer-reviewed literature is not the only record of what is known. Patent filings assigned to Amazon Technologies — including US 9,818,145; US 10,445,809; US 8,352,331; and US 11,367,117 — describe the problem in terms close to our own formulation.

The filings state, in substance, that where a purchaser explicitly signals a gift transaction — by requesting gift wrapping or including a gift message — the item can be excluded from behavioural analysis; but that **in many cases the merchant cannot determine that a purchase was a gift**, producing distortions in the user profile. The filings further note that when few data points are available for a customer, each gift purchase exerts a disproportionately strong distorting effect.

Three inferences follow, and they are more valuable to this project than a null result in the academic record would have been.

1. **Commercial significance is documented, not asserted.** A firm does not file multiple patents on a problem it considers trivial. The project need not argue that the problem matters; the industrial record does that.

2. **The deployed solution depends on an explicit signal.** Gift-wrap selection and gift-message entry are opt-in behaviours. Where the buyer does not opt in — likely the majority of gift purchases, particularly digital deliveries and self-delivered gifts — the mechanism provides no coverage. Textual detection is precisely a method for the residual.

3. **The magnitude has never been published.** Patents describe a mechanism; they do not report prevalence, and they do not quantify the downstream effect on recommendation quality. Neither figure exists in the public record for any dataset.

**Caveat on evidential status.** Patent filings are not peer-reviewed, are written to maximise claim scope, and describe intended rather than necessarily deployed systems. They are cited here as evidence of *problem recognition*, not as evidence of *effect size*.

---

## 8. Theme 6 — Large Language Models as Annotators

The project's central methodological move — inferring purchase intent from review text at scale — depends on a technique that only recently became viable. A now-substantial literature assesses its reliability.

**The positive evidence.** Gilardi, Alizadeh & Kubli (2023), in PNAS, found that zero-shot ChatGPT labelling exceeded the accuracy of crowd workers on political-science text classification, with results consistent across text types and periods. Törnberg (2023) reported higher accuracy, higher reliability, and equal or lower bias than human classifiers on annotation of political messages. Ziems et al. (2024) found that zero-shot LLMs reach fair agreement with humans on taxonomic labelling while producing explanations that sometimes exceed crowdworker quality.

**The qualifications, which are equally well established.** A comparative study of LLMs and humans as span annotators (2025) found inter-annotator agreement between LLMs and humans to be only *moderate*, concluding that LLMs cannot straightforwardly replace human annotators — though the strongest models do reach the level of agreement that human crowdworkers achieve among themselves. The same study reports two findings that directly shape our design: detailed **guidelines** describing conventions and ambiguous-case handling improve annotation quality, whereas supplying specific **examples** did not yield consistent improvement and may distract the model; and validation against expert hand-annotation on a sample is recommended in all cases. Pangakis, Wolken & Fasching (2023) argue that automated annotation is safe *provided* it is validated against uncontaminated human labels. Calderon, Reichart & Dror (2025) formalise this into a statistical test for whether an LLM may justifiably replace human annotators on a given task. Recent work on the hidden risks of LLM annotation quantifies how much downstream conclusions can shift with defensible variations in model and prompt.

**The distillation pattern.** Work on knowledge distillation in automated annotation demonstrates the design that makes corpus-scale annotation economically rational: use the LLM to label a sample, train a supervised classifier on those labels, and apply the classifier to the full corpus. The cost asymmetry reported is stark — annotating a 6.2-million-document corpus directly with a frontier model was estimated at roughly \$8,990, against roughly \$15 for a 1,000-document sample, \$124 for crowdworkers, and \$187 for a trained research assistant on the same sample.

**Synthesis.** The literature converges on a defensible operating procedure rather than a blanket endorsement: LLM annotation is acceptable for well-specified classification tasks **when** the task definition is precise, **when** a human-annotated validation sample establishes per-class performance, **when** sensitivity to model and prompt is reported rather than suppressed, and **when** full-corpus application proceeds via a distilled classifier rather than direct large-model inference. Our design adopts all four conditions.

---

## 9. Cross-Theme Synthesis

Placing the strands side by side isolates the boundary of current knowledge.

| Strand | Does it recognise gift purchases? | How is the signal identified? | What is done with it? | Data |
|---|---|---|---|---|
| Implicit feedback foundations | No | — | Treated as preference | Public benchmarks |
| Denoising implicit feedback | No — noise is mis-clicks / dissatisfaction | Loss behaviour, model disagreement | Discarded or down-weighted | Public benchmarks |
| Occasion-aware recommendation | **Yes** | Calendar / temporal position | **Modelled to improve prediction** | Proprietary (Etsy) |
| Gift recommendation | **Yes** | Explicit user intent to give | Used to recommend *to the giver* | Varied |
| Industry patents | **Yes** | Gift-wrap / gift-message flag | Excluded from profiling | Proprietary |
| LLM annotation | N/A | Semantic content of text | — | N/A |

Read across, the table shows that **every element of the solution already exists in isolation.**

The denoising literature supplies removal and reweighting machinery but not the semantic category. The occasion literature supplies the semantic category but detects it from the calendar and uses it for the opposite objective on inaccessible data. The gift literature supplies the commercial motivation but points the analysis at the giver's next gift rather than the platform's next recommendation. The patents supply the practitioner's recognition but rely on a signal most buyers never emit. The LLM-annotation literature supplies a validated procedure for extracting semantic categories from text at scale but has not been applied to this construct.

The unassembled combination is the contribution.

Two further observations sharpen the picture. First, the strands are **methodologically incommensurable**: they use different datasets, different evaluation protocols, and different success criteria, so no existing comparison can adjudicate between "model the occasion" and "remove the gift." Second, the only two strands that recognise gift purchases as a profiling problem — occasion-aware recommendation and industry patents — both rest on **proprietary data**. The public record therefore contains no prevalence estimate and no effect estimate for any dataset.

---

## 10. Marketing Relevance and Gap

### 10.1 What existing approaches do well

- Sequential architectures extract preference from interaction order with high effectiveness (Kang & McAuley, 2018; Sun et al., 2019).
- Adaptive denoising demonstrably improves recommendation quality when noise is statistically identifiable (Wang et al., 2021).
- Occasion modelling improves prediction when the occasion is calendar-anchored and platform data is rich (Wang et al., 2020).
- Explicit gift flags allow clean exclusion when the buyer volunteers them (Amazon patents).
- LLM annotation, properly validated, can produce reliable labels at a cost that makes corpus-scale semantic analysis tractable (Gilardi et al., 2023; Pangakis et al., 2023).

### 10.2 What remains unresolved

**G1 — Prevalence is unmeasured in public data.** No published figure exists for the share of interactions in any public recommendation dataset that are purchases for another person, nor for how that share varies across product categories or across the calendar year. Without this figure, neither researchers nor practitioners can size the problem.

**G2 — The effect on recommendation quality is untested.** No study has removed or down-weighted gift interactions and measured the consequence for the buyer's own subsequent purchase. The denoising literature never isolates this noise class; the occasion literature never removes it.

**G3 — "Model it" versus "remove it" has never been compared.** Wang et al. (2020) model the deviation. The denoising literature removes anomalies. The two prescriptions have never been evaluated against each other on the same data under the same protocol, so a practitioner facing this problem has no evidence-based basis for choosing.

**G4 — Textual detection of purchase intent is unexplored in this setting.** LLM annotation has been validated on sentiment, stance, topic, and framing. It has not been applied to purchase-beneficiary inference, despite this being a construct that buyers routinely state in plain language.

### 10.3 How this project responds

The project addresses G1 and G2 directly, G3 as a designed comparison, and G4 as an enabling contribution:

- **On G1** — LLM-based detection over the Amazon Reviews 2023 corpus (Hou et al., 2024), validated against human annotation, produces the first public prevalence estimates by category and by month.
- **On G2** — A controlled experiment removes (C1) and down-weights (C2) detected gift interactions and measures top-*N* accuracy on held-out purchases that are themselves classified as self-purchases. A **placebo condition (C4)** removes an equal number of randomly selected interactions, distinguishing a genuine effect from the mechanical consequence of reducing training data.
- **On G3** — A condition that supplies the gift flag as an additional model input (C3) rather than removing the interaction operationalises the Wang et al. (2020) prescription within our protocol, permitting the first direct comparison.
- **On G4** — The annotation schema, prompt, human-validation results, and per-class performance figures are released, extending the LLM-annotation literature to a new construct.

**A note on the modesty of the claim.** The contribution is not a new algorithm. It is measurement of a known but unquantified problem, plus a controlled test of the two competing remedies. Section 7 makes clear that a claim of novelty in *problem identification* would be false. The claim is one of **public quantification**, and it is defensible on the record established above.

### 10.4 Marketing significance

For a marketing practitioner the gap translates into three unanswerable questions. How much of my personalisation inventory is misallocated by gift purchases? How long does a single gift purchase distort a customer's slate? Should my data science team suppress the gift signal or feed it to the model? The literature currently supports no answer to any of the three. This project produces evidence for all three.

The reframing is worth stating plainly. Contamination is conventionally an engineering concern. Its consequence — impressions spent against a preference that never existed — is a **budget** concern. Framing the problem in terms of wasted personalisation inventory rather than degraded NDCG makes it legible to the person who controls the budget.

---

## 11. Conclusion

The literature reviewed here converges on a clear picture. The assumption that a recorded purchase reveals the purchaser's preference is foundational to implicit-feedback recommendation and is challenged nowhere in that lineage. A mature denoising literature exists but conceptualises noise as mis-clicks and dissatisfaction, detects it through optimisation behaviour rather than meaning, and therefore cannot isolate — or act differently upon — purchases made for someone else. A smaller occasion-aware literature does recognise gift-driven deviation, but detects it from the calendar, models it to improve prediction of the deviation itself, and rests on proprietary data. Industry patents confirm that the problem is recognised commercially, while revealing that deployed solutions depend on explicit gift flags that most buyers never provide. Meanwhile, a validated methodology for extracting semantic categories from text at scale has matured, subject to well-specified conditions on validation and sensitivity reporting.

Four implications follow for project design.

1. **Detection must be semantic, not statistical.** Loss-based methods cannot distinguish a gift from a disappointment. Text can.
2. **Validation must be layered.** The LLM-annotation literature is explicit that unvalidated automated annotation is not acceptable. Human annotation with inter-annotator agreement, plus model and prompt sensitivity analysis, is a requirement rather than a refinement.
3. **A placebo condition is mandatory.** Because the primary intervention reduces training data, any improvement is confounded with data volume unless a random-removal control is run.
4. **"Remove" must be tested against "model."** The literature contains both prescriptions and no comparison. Running both within one protocol is the highest-value design decision available.

The project contributes the first public estimate of gift-purchase prevalence in a recommendation corpus, the first controlled measurement of its effect on recommendation quality, and the first direct comparison of the two competing remedies — expressed in metrics that connect to how personalisation budget is actually allocated.

---

## 12. References

**Recommender system foundations**

1. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based Collaborative Filtering Recommendation Algorithms. *WWW '01*. https://doi.org/10.1145/371920.372071
2. Linden, G., Smith, B., & York, J. (2003). Amazon.com Recommendations: Item-to-Item Collaborative Filtering. *IEEE Internet Computing*, 7(1), 76–80.
3. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI 2009*, 452–461. arXiv:1205.2618
4. Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based Recommendations with Recurrent Neural Networks. *ICLR 2016*.
5. Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM 2018*, 197–206. https://doi.org/10.1109/ICDM.2018.00035
6. Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. *CIKM 2019*.

**Denoising implicit feedback**

7. Wang, W., Feng, F., He, X., Nie, L., & Chua, T.-S. (2021). Denoising Implicit Feedback for Recommendation. *WSDM '21*, 373–381. https://doi.org/10.1145/3437963.3441800 · arXiv:2006.04153
8. Wang, W., Feng, F., He, X., Nie, L., & Chua, T.-S. (2021b). Learning Robust Recommender from Noisy Implicit Feedback. arXiv:2112.01160
9. Yu, W., & Qin, Z. (2020). Sampler Design for Implicit Feedback Data by Noisy-label Robust Learning. *SIGIR '20*, 861–870. https://doi.org/10.1145/3397271.3401155
10. Wang, Z., Xu, Q., Yang, Z., Cao, X., & Huang, Q. (2021). Implicit Feedbacks are Not Always Favorable: Iterative Relabeled One-Class Collaborative Filtering against Noisy Interactions. *ACM MM '21*, 3070–3078. https://doi.org/10.1145/3474085.3475446
11. Qin, Y., Wang, P., & Li, C. (2021). The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation. *SIGIR '21*, 859–868. https://doi.org/10.1145/3404835.3462836
12. Wang, Y., Xin, X., Meng, Z., Jose, J. M., Feng, F., & He, X. (2022). Learning Robust Recommenders through Cross-Model Agreement. *WWW '22*, 2015–2025. https://doi.org/10.1145/3485447.3512202
13. Tian, C., Xie, Y., Li, Y., Yang, N., & Zhao, W. X. (2022). Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. *SIGIR '22*, 122–132. https://doi.org/10.1145/3477495.3531889
14. Chua, H., Du, Y., Sun, Z., Wang, Z., Zhang, J., & Ong, Y.-S. (2024). Unified Denoising Training for Recommendation. *RecSys '24*, 612–621.
15. Zhao, J., Wang, W., Xu, Y., Sun, T., Feng, F., & Chua, T.-S. (2024). Denoising Diffusion Recommender Model. *SIGIR '24*, 1370–1379.

**Occasion- and context-aware recommendation**

16. Wang, J., Louca, R., Hu, D., Cellier, C., Caverlee, J., & Hong, L. (2020). Time to Shop for Valentine's Day: Shopping Occasions and Sequential Recommendation in E-commerce. *WSDM '20*, 645–653. https://doi.org/10.1145/3336191.3371836 · [PDF](https://people.engr.tamu.edu/caverlee/pubs/wang20wsdm-valentine.pdf) · [Code](https://github.com/wangjlgz/Occasion-Aware-Recommenation)

**Gift exchange and gift recommendation**

17. Gift recommendation systems: a review. (2023). *Electronic Commerce Research*. https://doi.org/10.1007/s10660-023-09790-6
18. The Art of Gift-Giving with Limited Preference Data: How Fashion Recommender Systems Can Help. (2024). *CHI EA '24*. https://doi.org/10.1145/3613905.3651000

**Industry evidence**

19. Amazon Technologies, Inc. US Patent 9,818,145 — item recommendation and gift-purchase handling.
20. Amazon Technologies, Inc. US Patent 10,445,809.
21. Amazon Technologies, Inc. US Patent 8,352,331.
22. Amazon Technologies, Inc. US Patent 11,367,117.

**LLM annotation and distillation**

23. Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*.
24. Törnberg, P. (2023). ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning. arXiv preprint.
25. Pangakis, N., Wolken, S., & Fasching, N. (2023). Automated Annotation with Generative AI Requires Validation. arXiv preprint.
26. Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). Can Large Language Models Transform Computational Social Science? *Computational Linguistics*.
27. Calderon, N., Reichart, R., & Dror, R. (2025). The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs. *ACL 2025*, 16051–16081. https://doi.org/10.18653/v1/2025.acl-long.782
28. LLMs as Span Annotators: A Comparative Study of LLMs and Humans. (2025). arXiv:2504.08697
29. Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels. (2024). arXiv:2406.17633
30. Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation. (2025). arXiv:2509.08825

**Dataset**

31. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952 · [Dataset](https://amazon-reviews-2023.github.io/)

---

*All sources verified as of August 2026. Patent filings are cited as documentary evidence of industry problem recognition and are not peer-reviewed.*
