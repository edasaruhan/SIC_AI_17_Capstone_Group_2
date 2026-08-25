# Technology Review

**Project:** *Gift Purchases as a Distinct Class of Noise in Implicit-Feedback Recommender Systems*

**Course:** AI in Marketing Capstone
**Deliverable:** Technology Review

---

## 1. Introduction

### 1.1 What is under evaluation

This project requires a technology stack spanning three distinct layers, each with genuine alternatives and non-obvious trade-offs:

| Layer | Function | Candidate technologies |
|---|---|---|
| **Detection** | Infer from review text whether a purchase was made for the buyer or for someone else | Keyword rules · open-weight LLMs (Qwen, Gemma) · commercial LLM APIs · human annotation |
| **Scaling** | Apply the detection decision to millions of interactions at acceptable cost | Direct LLM inference · knowledge distillation into an encoder (ModernBERT, DeBERTaV3, RoBERTa) |
| **Measurement** | Train recommenders under controlled conditions and evaluate the intervention | RecBole · Cornac · `implicit` · custom implementation; and, orthogonally, full-ranking versus sampled evaluation protocols |

A supporting layer — LLM serving infrastructure (vLLM, Ollama, llama.cpp) — determines whether the detection layer is economically feasible on the hardware available.

### 1.2 Why this review matters for the marketing outcome

Technology choice in this project is not a matter of engineering taste; three of the decisions below determine whether the marketing conclusion is *valid at all*.

**The detection technology determines what we can say about a customer.** A keyword rule that flags every review containing "gift" will classify a customer who wrote "this would make a great gift" as a gift purchaser when they bought the item for themselves. A marketer acting on that output suppresses personalisation for a customer who never made a gift purchase — actively degrading the experience in the name of improving it. The precision of the detector is therefore not an ML metric; it is the rate at which the system will make the wrong marketing decision about a real person.

**The evaluation protocol determines whether the result is real.** As Section 4.5 documents, the recommender-systems field has an established reproducibility problem in which methodological choices — not method quality — drive reported improvements. A project whose entire claim rests on a measured difference between two training conditions is maximally exposed to this failure mode.

**The scaling technology determines whether the project is affordable.** The difference between direct large-model inference over a full corpus and a distilled classifier is roughly three orders of magnitude in cost — the difference between a feasible student project at zero budget and an infeasible one.

---

## 2. Technology Overview

### 2.1 Open-weight large language models as semantic classifiers

Instruction-tuned LLMs can classify text against a specified schema without task-specific training. Recent open-weight families — Qwen and Gemma among them — are released under permissive licences and run on consumer hardware when quantised, placing corpus-scale semantic annotation within reach of a project with no cloud budget.

The capability relevant here is **structured extraction**: given a review, return a typed object specifying who the purchase was for, the confidence of that judgement, the relation of the recipient, the occasion, and a verbatim span of text supporting the decision. Modern serving stacks enforce the output schema through guided decoding, eliminating the parsing failures that made earlier LLM pipelines brittle.

**Typical marketing use.** Review mining and aspect extraction, support-ticket routing, lead qualification from free-text form fields, social listening classification, survey open-response coding, and content moderation. The common shape is the same: a semantic judgement that previously required a human reader, applied at a volume no human team could reach.

### 2.2 Efficient LLM serving

vLLM introduced **PagedAttention**, an attention algorithm modelled on virtual-memory paging in operating systems, to address fragmentation and redundant duplication in key-value cache memory. Because KV-cache memory for each request is large and grows and shrinks dynamically, inefficient management severely limits achievable batch size. The reported result is a 2–4× throughput improvement over prior state-of-the-art serving systems at equivalent latency (Kwon et al., 2023).

For an interactive chatbot this is a quality-of-service improvement. For an **offline batch annotation job** — this project's workload — throughput is the entire cost function, because the job is embarrassingly parallel and latency-insensitive.

### 2.3 Encoder models for distillation

ModernBERT (Warner et al., 2025) is a modernised encoder-only architecture trained on 2 trillion tokens with a native 8,192-token sequence length, incorporating rotary position embeddings, GeGLU layers, and alternating local–global attention. It is the first encoder to surpass DeBERTaV3-base on GLUE since that model's 2021 release, processes short-context inputs approximately twice as fast as DeBERTaV3, and is explicitly designed for inference on common GPUs.

Two properties matter for this project specifically. The **8,192-token context** means long reviews are classified in full rather than truncated at 512 tokens — and the sentence disclosing the recipient frequently appears late in a review, after the product discussion. The **inference efficiency** is what makes full-corpus application viable.

### 2.4 Recommendation frameworks

RecBole (Zhao et al., 2021) provides a unified implementation of a large algorithm library under a common data pipeline and evaluation protocol, configured declaratively. Its significance for this project is not convenience but **comparability**: results produced under a widely used standard protocol can be situated against published numbers, whereas a bespoke implementation cannot.

### 2.5 How these technologies are used in marketing today

| Technology | Established marketing application |
|---|---|
| Sequential recommenders (SASRec, GRU4Rec) | Homepage and PDP personalisation, lifecycle email content selection, app feed ranking |
| Matrix factorisation (BPR) | Baseline personalisation, candidate generation at scale |
| LLM text classification | Review and social-listening analysis, VOC coding, lead qualification, ticket routing |
| Encoder classifiers (BERT family) | High-throughput production classification where LLM latency or cost is prohibitive |
| Interaction denoising | Data-quality layers in production recommendation pipelines |

---

## 3. Relevance to the Marketing Project

### 3.1 Mapping technology to the marketing decision

The project's output is a marketing decision — *suppress the gift signal, or model it, and at what threshold* — and each technology layer produces an input to that decision.

```
Review corpus
     │
     ▼
[LLM detector]  ──►  "Who was this purchase for?"
     │                Answers RQ1: prevalence by category and month
     ▼
[Distilled encoder] ──► Corpus-scale labelling at feasible cost
     │
     ▼
[RecBole experiment] ──► Answers RQ2 (does removal help?) and RQ3 (remove vs. model?)
     │
     ▼
[Marketing metrics] ──► Wasted personalisation inventory · contamination half-life
     │
     ▼
DECISION: suppress / down-weight / model the gift signal
```

### 3.2 Position in the marketing workflow

The intervention sits in the **data preparation layer** of the personalisation pipeline — upstream of model training, downstream of event collection. This placement has a practical consequence worth stating: the remedy, if effective, requires no change to the recommendation model, the serving infrastructure, or the customer-facing experience. It is a filter on the training signal. For a marketing organisation, that is the cheapest class of intervention available, and it is one reason this problem is worth measuring even if the measured effect proves modest.

Secondary applications follow from the same detected signal: suppression rules in retargeting audiences, exclusion of gift-driven category entries from segmentation, and — as an opportunity rather than a deliverable — a gift-intent trigger for seasonal campaigns.

---

## 4. Comparison and Evaluation

Evaluation criteria are chosen for the marketing task rather than for generic benchmark performance. Specifically: the cost of a false positive expressed as a wrong decision about a customer; auditability by a non-technical stakeholder; total cost at corpus scale; and reproducibility of the resulting claim.

### 4.1 Axis 1 — Detection technology

| Criterion | Keyword rules | Open-weight LLM | Commercial LLM API | Human annotation |
|---|---|---|---|---|
| Handles implicit phrasing<sup>a</sup> | ✗ | ✓ | ✓ | ✓ |
| Rejects speculative gift language<sup>b</sup> | ✗ | ✓ | ✓ | ✓ |
| Cost at 40–60K documents | ~0 | ~0 (own GPU) | Moderate | High |
| Cost at full corpus (millions) | ~0 | Days of GPU time | Prohibitive | Impossible |
| Data leaves the organisation | No | No | **Yes** | No |
| Reproducible by third parties | ✓ | ✓ | ✗ (model versions deprecate) | ✗ |
| Auditability | High (rules readable) | Medium (evidence span) | Medium | High |
| Consistency | Perfect | High at temperature 0 | High | Moderate |

<sup>a</sup> "Bought it and my daughter loved it" — no gift vocabulary present.
<sup>b</sup> "This would make a great gift" — gift vocabulary present, self-purchase.

**Decision: open-weight LLM for annotation, keyword rules retained as a pilot diagnostic only.**

The two failure modes in the footnotes are not edge cases; they are the central difficulty. A keyword rule errs in both directions simultaneously, and in a marketing deployment those errors are asymmetric in consequence: a false positive suppresses personalisation for a customer who did not make a gift purchase.

Commercial APIs are rejected for three converging reasons. Customer review text would leave the organisation, which is unacceptable for a technique intended for production use under GDPR/KVKK-equivalent regimes. Reproducibility degrades as model versions are deprecated — a study that cannot be re-run is a weak scientific contribution. And the project constraint is a zero budget.

Human annotation is retained, but as **validation** rather than production. The literature is unanimous on this point: automated annotation requires validation against human labels, and formal statistical procedures now exist for determining whether an LLM may justifiably substitute for human annotators on a given task (Calderon et al., 2025).

### 4.2 Axis 2 — Model selection within open weights

| Model | ~VRAM (4-bit) | Licence | Assessment for this task |
|---|---|---|---|
| **Qwen3.5-9B** | ~6–7 GB | Apache 2.0 | **Primary.** Strong structured-output behaviour; runs comfortably on 8 GB consumer cards |
| **Gemma 4 12B** | ~8 GB | Apache 2.0 | **Secondary.** Retained specifically to measure cross-model agreement |
| Qwen3.5-4B | ~3 GB | Apache 2.0 | Fallback if throughput binds; likely adequate for four-way classification |
| gpt-oss-20b | ~16 GB | Apache 2.0 | Reasoning capacity exceeds task requirements; poor cost/benefit here |

**Why two models rather than one.** Running a second model is not redundancy. Recent work quantifies how substantially downstream conclusions can shift under defensible variations in model and prompt — a phenomenon sharp enough to have acquired a name in the methodological literature. Reporting agreement between two independent models converts an unexamined assumption into a measured quantity, and a divergence between them is a finding to report rather than a problem to conceal.

### 4.3 Axis 3 — Serving infrastructure

| Criterion | vLLM | Ollama | llama.cpp |
|---|---|---|---|
| Batch throughput | **Highest** (PagedAttention, continuous batching) | Low | Low–moderate |
| Guided JSON decoding | Native | Limited | Limited |
| Setup difficulty | Moderate (CUDA-sensitive) | **Very low** | Moderate |
| Suited to interactive use | Adequate | **Excellent** | Good |
| Suited to offline batch | **Excellent** | Poor | Moderate |

**Decision: vLLM.** The workload is a single large offline batch, where throughput is the entire cost function and interactive convenience is worth nothing. The reported 2–4× throughput advantage translates directly into the difference between an overnight job and a multi-day one. Native guided decoding is a second, independent reason: it eliminates output-parsing failure as a category of error rather than mitigating it.

### 4.4 Axis 4 — Distillation target

| Criterion | ModernBERT-base | DeBERTaV3-base | RoBERTa-base | No distillation (LLM direct) |
|---|---|---|---|---|
| Max context | **8,192** | 512 | 512 | Model-dependent |
| Relative inference speed | **Fastest** | ~0.5× | ~0.6× | Orders of magnitude slower |
| GLUE-class accuracy | State of the art among encoders | Strong | Good | Highest |
| Full-corpus feasibility | ✓ | ✓ | ✓ | ✗ |

**Decision: ModernBERT-base.**

The context-length argument is decisive and specific to this task. Reviews frequently open with product discussion and disclose the recipient only in a closing sentence. A 512-token model truncates exactly the region where the label-bearing evidence most often sits. ModernBERT removes the truncation decision entirely.

The economic argument for distillation as a pattern is documented rather than assumed. Reported annotation costs for a 6.2-million-document corpus were approximately \$8,990 for direct frontier-model inference against roughly \$15 for a 1,000-document sample — with crowdworkers at \$124 and a trained research assistant at \$187 for the same sample. The distillation pattern — label a sample with the large model, train a small model on those labels, apply the small model to everything — converts an infeasible cost into a trivial one, and is now an established design in the automated-annotation literature.

### 4.5 Axis 5 — Recommendation framework

| Criterion | RecBole | Cornac | `implicit` | Custom |
|---|---|---|---|---|
| Algorithm coverage | **Very broad** | Broad | MF/ALS only | As built |
| Unified evaluation protocol | ✓ | ✓ | Partial | Must be built |
| Full-ranking evaluation | ✓ | ✓ | ✓ | Must be built |
| Comparability to published results | **High** | Moderate | Low | **Very low** |
| Custom condition support (C0–C4) | Config + data-layer | Moderate | Limited | Total |
| Implementation risk | Low | Low | Low | **High** |

**Decision: RecBole.**

The justification rests on the reproducibility literature rather than on convenience, and this is the single most important technology argument in the review.

Ferrari Dacrema, Cremonesi & Jannach (2019) analysed recent neural recommendation approaches and found that reported gains frequently failed to survive comparison against properly tuned simple baselines — a result they extended in a 2021 TOIS analysis of reproducibility and progress in the field. A replicability study of BERT4Rec found difficulty reproducing the originally reported results and confirmed that weak baseline configurations were not confined to simple models.

For this project the implication is direct. Our claim is a **difference between two training conditions**, which is precisely the kind of claim most vulnerable to configuration artefacts. Writing bespoke implementations would mean any observed difference could be attributed to our code. Using a widely adopted framework moves that risk substantially, and permits a reviewer to check our numbers against a known reference.

### 4.6 Axis 6 — Evaluation protocol *(frequently overlooked; decisive here)*

| Criterion | Full ranking | Sampled metrics (e.g. 100 negatives) |
|---|---|---|
| Computational cost | High | Low |
| Consistency with global metrics | ✓ | **✗** |
| Can reverse model rankings | — | **Yes** |
| Prevalence in older literature | Lower | High |

**Decision: full ranking over the complete item catalogue.**

Krichene & Rendle (2020) showed that commonly used top-*k* metrics — Recall, Hit Ratio, Precision, Average Precision, NDCG — computed against a sampled set of negatives are **inconsistent** with their global counterparts, even in expectation. Conclusions about relative model performance can change depending on whether sampling is used, a finding subsequently confirmed specifically for sequential recommendation.

This is not a peripheral methodological preference. If sampled evaluation can reverse the ordering between two *models*, it can equally reverse the ordering between two *training conditions* — which is our entire result. Full-ranking evaluation is therefore treated as a requirement, and the additional compute is accepted as the cost of a defensible claim.

### 4.7 Axis 7 — Intervention strategy

This axis is a research question rather than a settled decision, and is evaluated experimentally.

| Strategy | Mechanism | Origin in literature | Marketing interpretation |
|---|---|---|---|
| **C1 — Hard removal** | Delete detected gift interactions | Truncated Loss (Wang et al., 2021) | "Gift purchases tell us nothing; discard them" |
| **C2 — Soft down-weighting** | Weight interactions by *w* < 1 | Reweighted Loss (Wang et al., 2021) | "Gift purchases tell us something, but less" |
| **C3 — Model as signal** | Supply gift flag as model input | Occasion-aware modelling (Wang et al., 2020) | "Gift purchases are informative if the model knows they are gifts" |
| **C4 — Placebo** | Remove an equal number of random interactions | Experimental control | Not a strategy — a validity check |

**On C4.** The primary intervention reduces training data. Any improvement is therefore confounded with data volume unless a random-removal control of identical size is run. If C1 does not outperform C4, the finding is an artefact. This condition is not optional, and its omission would invalidate the study.

**On C1 versus C3.** As established in the Literature Review, C3 operationalises the prescription of the occasion-aware literature and C1 that of the denoising literature. The two have never been compared under a common protocol. Running both is the highest-value design decision in the project.

---

## 5. Use Cases and Examples

### 5.1 Etsy — occasion-aware recommendation in production

**Problem.** On a marketplace with unusually high gift density, sequential recommenders trained on intrinsic preference mispredict during occasion-driven purchase episodes.

**Technology.** An occasion-aware sequential recommendation framework modelling occasion signals alongside intrinsic preference, developed by Texas A&M and Etsy researchers and published at WSDM 2020 (a full paper at a venue with a 15% acceptance rate).

**Outcome.** Improved next-item prediction over occasion-agnostic sequential baselines on proprietary Etsy data.

**What we learn.** Three things. The phenomenon is real and commercially significant at platform scale, which removes any need for us to argue that it exists. The calendar-signal approach is the deployed alternative to ours, which makes C3 a meaningful comparison rather than a straw man. And the reliance on proprietary data is the constraint our public-data study is designed to relieve — no external party can currently verify, extend, or contest this result.

### 5.2 Amazon — gift-flag exclusion from behavioural profiling

**Problem.** Gift purchases distort customer profiles used for recommendation.

**Technology.** Patent filings assigned to Amazon Technologies describe excluding purchases from behavioural analysis when the buyer supplies an explicit gift signal — gift wrapping or a gift message — while acknowledging that in many cases the merchant cannot determine that a purchase was a gift, and that the distorting effect is disproportionately strong when few data points exist for a customer.

**Outcome.** Not publicly reported. Patents describe mechanisms, not measured effects.

**What we learn.** The commercial significance of the problem is documented by a firm's willingness to file repeatedly on it. Simultaneously, the deployed remedy has a structural coverage gap: it depends on an opt-in behaviour, leaving every gift purchase where the buyer did not request wrapping unaddressed. Textual detection is a method for exactly that residual — and this framing positions our approach as complementary to industry practice rather than competing with it.

### 5.3 Computational social science — LLM annotation at research scale

**Problem.** Large text corpora require semantic labels; human annotation does not scale.

**Technology and outcome.** Gilardi, Alizadeh & Kubli (2023) found in PNAS that zero-shot ChatGPT annotation exceeded crowdworker accuracy on political-science classification, consistently across text types and time periods. Törnberg (2023) reported higher accuracy, higher reliability, and equal or lower bias than human classifiers.

**The countervailing evidence.** A 2025 comparative study of LLMs and humans as span annotators found inter-annotator agreement between LLMs and humans to be only moderate, concluding that LLMs cannot straightforwardly replace human annotators — while noting that the strongest models do reach the agreement level that human crowdworkers achieve among themselves.

**What we learn — and how it shaped our design.** The same study yields two directly actionable findings. Detailed **guidelines** covering conventions and ambiguous cases improve quality, whereas supplying specific **examples** produced no consistent improvement and may distract the model through length and complexity. Our prompt is accordingly guideline-heavy and few-shot-light, which inverts the conventional default. Second, validation against expert hand-annotation on a sample is recommended universally, which is why a 500-item human-annotated set with inter-annotator agreement statistics is treated as a deliverable rather than an optional check.

### 5.4 The recommender-systems reproducibility debate

**Problem.** Reported improvements in neural recommendation frequently fail to replicate.

**Findings.** Ferrari Dacrema et al. (2019, 2021) found that reported gains often did not survive comparison against properly tuned simple baselines. Rendle et al. (2020) revisited neural collaborative filtering against matrix factorisation with similar implications. Krichene & Rendle (2020) demonstrated that sampled evaluation metrics are inconsistent with global metrics. A BERT4Rec replicability study confirmed that these problems extend to more complex and genuinely effective models.

**What we learn.** This is a cautionary case rather than a success story, and it shapes four concrete design decisions: use a standard framework (§4.5), evaluate by full ranking (§4.6), include simple baselines that are *properly tuned* rather than included as decoration, and report bootstrap confidence intervals across multiple random seeds rather than single point estimates. A project reporting a small difference between two conditions has an obligation to be more careful than the field's median, not less.

---

## 6. Limitations, Risks, and Opportunities

### 6.1 Detection layer

| Risk | Consequence | Mitigation |
|---|---|---|
| **Hallucinated labels** | Interactions misclassified; wrong marketing decision about real customers | Mandatory verbatim `evidence_span`; records whose span does not appear in the source text are demoted to `unclear` and counted |
| **Prompt sensitivity** | Prevalence estimate is an artefact of one prompt | Two prompts × two models = four configurations; disagreement reported, not suppressed |
| **Systematic bias against non-standard English** | Under-detection for some customer populations | Error analysis stratified by review length and linguistic register; acknowledged as a limitation |
| **Ambiguous `household` class** | Low inter-annotator agreement corrupts the label set | If Fleiss' κ < 0.6, merge `household` into `unclear` and report the decision |

**On the `household` class.** Buying nappies for one's own baby is neither a gift nor an expression of the buyer's personal taste. Retaining it as a distinct class is analytically correct but raises annotation difficulty. The pre-registered fallback above prevents a low-agreement class from silently degrading the results.

### 6.2 Data layer

**Selection bias is the most serious limitation, and it is structural.** Reviews exist for only a subset of purchases, and gift purchases are plausibly reviewed at a *lower* rate than self-purchases — the buyer does not possess the item and cannot assess it. Consequently:

- The measured prevalence describes **gift purchases among reviewed transactions**, not among all transactions.
- The direction of bias is likely downward, making our estimate a plausible lower bound — but the magnitude is unknown and cannot be estimated from the data.

The honest framing is that this bounds the *external* claim without invalidating the *internal* experiment: the denoising intervention is applied to the reviewed-interaction graph, which is the graph the recommender actually trains on in this dataset.

**Privacy.** Review text is user-generated free text and may contain personal information, including names of recipients. No verbatim review text will appear in published outputs; reporting is restricted to aggregate statistics and paraphrased illustrative cases.

**Temporal displacement.** Review timestamps are not purchase timestamps. Seasonality peaks will be lagged. This is stated in advance as expected behaviour rather than defended after the fact.

### 6.3 Experiment layer

| Risk | Mitigation |
|---|---|
| Improvement reflects reduced data volume, not denoising | C4 placebo condition (mandatory) |
| Result is an artefact of one data split or seed | Repeated seeds; bootstrap confidence intervals |
| Removing gift interactions eliminates some items entirely | Item universe frozen on C0; vanished items logged and reported as a possible partial explanation of any C1–C0 difference |
| Weak baselines inflate apparent effect | Simple baselines tuned, not decorative (per Ferrari Dacrema et al.) |
| Null result | Reframed in advance: RQ2 asks about *sensitivity*, so a robustness finding answers it |

### 6.4 Opportunities

**Human-in-the-loop annotation.** The span-annotation literature suggests a hybrid design — LLM pre-annotation with human post-editing — as a promising direction. With more time, this would raise label quality above either component alone.

**Complementarity of calendar and text signals.** Our detector and the Etsy calendar approach have different blind spots. Measuring their overlap and disagreement would quantify each method's unique coverage and is a natural extension.

**Confidence-thresholded deployment.** Because the detector emits a confidence field, a production deployment could suppress personalisation only above a chosen threshold, trading coverage against precision. Sweeping this threshold and reporting the resulting precision–coverage curve converts a binary academic result into a tunable business control — arguably the most practically useful artefact the project could produce.

**Transferability.** The pipeline is domain-agnostic. Any platform holding user-generated text alongside transactions — travel, ticketing, grocery — could apply it directly.

---

## 7. Conclusion

The technology selection is driven by three constraints that a conventional stack would not satisfy.

**The detection problem is semantic, so the technology must be semantic.** The two failure modes that matter — gift language in a self-purchase, and a gift purchase with no gift language — are precisely the cases a lexical method inverts. An open-weight instruction-tuned LLM handles both, keeps customer text in-house, and remains reproducible by third parties in a way a commercial API does not. Qwen3.5-9B is the primary annotator with Gemma 4 12B as an independent second opinion, because the methodological literature demonstrates that conclusions can shift under defensible variations in model and prompt, and a measured disagreement is worth more than an unexamined assumption.

**Corpus scale forces distillation.** Direct large-model inference over millions of documents is neither affordable nor necessary. ModernBERT-base is the distillation target, chosen for inference efficiency and — decisively for this task — its 8,192-token context, which prevents truncation of the review regions where recipient disclosure most often appears.

**The claim is a between-condition difference, so measurement must be conservative.** The field's documented reproducibility problems mean a small measured difference is exactly the kind of result most likely to be an artefact. RecBole supplies a standard, comparable protocol; full-ranking evaluation avoids the sampled-metric inconsistency that can reverse conclusions; the placebo condition separates genuine denoising from mere data reduction; and repeated seeds with bootstrap intervals prevent a single fortunate split from being reported as a finding.

**Practical value if the project succeeds.** The intervention lives in the data preparation layer. It requires no change to the recommendation model, the serving stack, or the customer-facing experience — it is a filter on the training signal. For a marketing organisation this is the cheapest class of change available, and the outputs translate directly into the vocabulary of budget: what share of personalisation inventory is spent against preferences that never existed, and for how long a single gift purchase continues to distort a customer's slate.

**And if it does not.** A finding that modern sequential architectures are robust to this contamination is equally publishable and equally useful. It would tell marketing teams to stop worrying about a problem that industry patents implicitly assume is severe — and the same detection pipeline would remain valuable for gift-intent triggering and retargeting suppression regardless. The technology choices above are made so that either outcome is trustworthy.

---

## 8. References

**LLM serving and encoder architecture**

1. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP '23*. https://doi.org/10.1145/3600006.3613165
2. Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Adams, G. T., Howard, J., & Poli, I. (2025). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *ACL 2025*, 2526–2547. https://doi.org/10.18653/v1/2025.acl-long.127 · arXiv:2412.13663

**Recommendation frameworks, models, and reproducibility**

3. Zhao, W. X., Mu, S., Hou, Y., Lin, Z., Li, K., Chen, Y., Lu, Y., Wang, H., Tian, C., Pan, X., Min, Y., Feng, Z., Fan, X., Chen, X., Wang, P., Ji, W., Li, Y., Wang, X., & Wen, J.-R. (2021). RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. *CIKM 2021*. arXiv:2011.01731
4. Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM 2018*, 197–206. https://doi.org/10.1109/ICDM.2018.00035
5. Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based Recommendations with Recurrent Neural Networks. *ICLR 2016*.
6. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI 2009*, 452–461.
7. Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. *RecSys '19*, 101–109. https://doi.org/10.1145/3298689.3347058
8. Ferrari Dacrema, M., Boglio, S., Cremonesi, P., & Jannach, D. (2021). A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research. *ACM TOIS*, 39(2), Article 20.
9. Krichene, W., & Rendle, S. (2020). On Sampled Metrics for Item Recommendation. *KDD '20*, 1748–1757.
10. Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2020). Neural Collaborative Filtering vs. Matrix Factorization Revisited. *RecSys '20*, 240–248. https://doi.org/10.1145/3383313.3412488
11. Petrov, A., & Macdonald, C. (2022). A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. *RecSys '22*. arXiv:2207.07483

**Denoising and occasion-aware recommendation**

12. Wang, W., Feng, F., He, X., Nie, L., & Chua, T.-S. (2021). Denoising Implicit Feedback for Recommendation. *WSDM '21*, 373–381. https://doi.org/10.1145/3437963.3441800
13. Wang, J., Louca, R., Hu, D., Cellier, C., Caverlee, J., & Hong, L. (2020). Time to Shop for Valentine's Day: Shopping Occasions and Sequential Recommendation in E-commerce. *WSDM '20*, 645–653. https://doi.org/10.1145/3336191.3371836

**LLM annotation methodology**

14. Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*.
15. Törnberg, P. (2023). ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning. arXiv preprint.
16. Pangakis, N., Wolken, S., & Fasching, N. (2023). Automated Annotation with Generative AI Requires Validation. arXiv preprint.
17. Calderon, N., Reichart, R., & Dror, R. (2025). The Alternative Annotator Test for LLM-as-a-Judge. *ACL 2025*, 16051–16081. https://doi.org/10.18653/v1/2025.acl-long.782
18. LLMs as Span Annotators: A Comparative Study of LLMs and Humans. (2025). arXiv:2504.08697
19. Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels. (2024). arXiv:2406.17633
20. Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation. (2025). arXiv:2509.08825
21. Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). Can Large Language Models Transform Computational Social Science? *Computational Linguistics*.

**Dataset and industry sources**

22. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952 · https://amazon-reviews-2023.github.io/
23. Amazon Technologies, Inc. US Patents 9,818,145; 10,445,809; 8,352,331; 11,367,117.
24. vLLM documentation — PagedAttention design. https://docs.vllm.ai/en/latest/design/paged_attention/

---

*All technical claims verified against primary sources as of August 2026. Model VRAM figures are approximate and depend on quantisation method and sequence length. Patent filings describe intended mechanisms and are cited as evidence of problem recognition, not of deployed system behaviour or measured effect.*
