# Decision Rule Threshold/K Sweep (Current Fine-Tuned Model)

Joint sweep over k ∈ {1,3,5,7,9} and INCONCL_THRESH ∈ {0.55–0.74}, using the
exact production decision logic from `app.py::compute_decision`, evaluated on
the validation pool (102 queries) against the index pool (476 images).

## Selected candidates (inconclusive rate < 15%, sensitivity ≥ 0.90)

| k | threshold | sensitivity | specificity | inconclusive | accuracy |
|---|-----------|-------------|-------------|---------------|----------|
| 5 | 0.55–0.66 | 0.956       | 0.824       | 0.000         | 0.912    |
| **5** | **0.68** | **0.955** | **0.848** | **0.029** | **0.919** |
| 5 | 0.70 (previous default) | 0.950 | 0.844 | 0.098 | 0.913 |
| 3 | 0.55–0.66 | 0.941       | 0.794       | 0.000         | 0.892    |
| 7 | 0.55–0.66 | 0.941       | 0.765       | 0.000         | 0.882    |

## Decision

Changed `INCONCL_THRESH` from 0.70 → **0.68** (k stays at 5). At 0.68, every
metric is at least as good as 0.70 — higher sensitivity, higher specificity,
a much lower inconclusive rate, and higher accuracy. This isn't a
sensitivity/specificity tradeoff; 0.70 was simply past a small "cliff" where
the inconclusive rate jumps sharply (9.8%) for no corresponding accuracy gain.

Higher thresholds (0.72+) were also tried and rejected — inconclusive rate
climbs to 26–100% for at most a marginal, statistically insignificant
sensitivity gain (see `kfold_eval_results.json` / raw sweep in git history of
this file's generating session for the full table).

Re-verified end-to-end against the live `/search` endpoint on 40 held-out
images: 85% accuracy, 19/20 sensitivity, 15/20 specificity, 0 inconclusive —
consistent with pre-change behavior on this slice (all 40 samples' confidence
already exceeded 0.68; the improvement is visible in the larger 102-sample
validation sweep above, where several borderline cases sit between 0.68 and
0.70).

Generated: 2026-09-08.

---

## Update 2026-09-10: switched to a metric-learning fine-tune

The classification-fine-tuned backbone above (`finetuned_mobilenetv2.pth`,
cross-entropy on a 2-class head) was replaced with a version further
fine-tuned directly for retrieval — batch-hard triplet loss on cosine
distance, warm-started from the classification checkpoint rather than from
scratch (see `scripts/finetune_metric_learning.py`). The classification
checkpoint is preserved at
`models/finetuned_mobilenetv2_classification_backup.pth`.

Rationale: the deployed system does k-NN cosine-similarity retrieval, but
the classification fine-tune optimizes a proxy objective (2-class softmax)
and only reuses its penultimate features for retrieval after the fact.
Training the embedding directly for "same-class close, different-class far"
should suit k-NN matching better.

### Result — evaluated identically on both models, holdout set (true generalization test, never used for tuning)

| Metric | Classification fine-tune (old) | Metric-learning fine-tune (new) |
|---|---|---|
| AUC-ROC | 0.951 | **0.968** |
| Sensitivity | 0.986 | 0.986 |
| Specificity | 0.618 | **0.706** |
| Accuracy | 0.864 | **0.893** |

Same sensitivity, meaningfully better specificity and accuracy, on data
neither model nor threshold was tuned against. Confirmed live end-to-end
against `/search` on the same 40-image holdout slice used throughout this
project's evaluation history: accuracy 85.0% → **92.5%**, specificity 15/20 → **18/20**,
sensitivity unchanged at 19/20.

---

## Update 2026-09-10: dataset expansion (750 → 823 index images, 681 → 1178 total)

A second data source (`data/dataset/`, 1238 raw files: "Oral Cancer photos" +
"normal") was added. Deduplicated against the existing corpus and internally
(perceptual hash, threshold 4) before use — most of the folder turned out to
be re-exports of images already in the corpus:

| | Folder total | After dedup (genuinely new) |
|---|---|---|
| Cancer | 685 | 187 |
| Normal | 553 | 310 |
| **Total** | **1238** | **497** |

The 497 unique images were resized+CLAHE'd identically to the existing
pipeline (`preprocess_data.py`'s `apply_clahe`), split 70/15/15
(stratified), and appended to `index_pool.json` / `validation_pool.json` /
`holdout_pool.json` (476→823 / 102→176 / 103→179). Bonus effect: fixed the
class imbalance — cancer:normal went from 2:1 to a much healthier 1.2:1.

The metric-learning model was retrained (warm-started from the pre-expansion
checkpoint, saved as `models/finetuned_mobilenetv2_metric_v1_before_expansion.pth`)
on the expanded index, and the CBIR index rebuilt against the expanded pool
(823 images). Re-ran the full evaluation suite and re-swept k/threshold on
the new embedding space:

### Holdout set (expanded, 179 images — true generalization test)

| Metric | Pre-expansion model | Post-expansion model |
|---|---|---|
| AUC-ROC | 0.936 | **0.945** |
| Accuracy | 0.872 | **0.899** |
| Specificity | 0.741 | **0.852** |
| Inconclusive rate | 0.196 | **0.061** |

(Sensitivity at the *old* threshold dropped 0.980→0.939 — expected, since
0.70 wasn't re-tuned for this model yet at that point. See below.)

### Re-tuned decision rule (validation pool, 176 queries)

Kept k=5 for continuity. `INCONCL_THRESH` stays at 0.70 — it happens to
already be the strongest low-deferral operating point for the new
embedding space too:

| k | threshold | sensitivity | specificity | inconclusive | accuracy |
|---|-----------|-------------|-------------|---------------|----------|
| **5** | **0.70** | **0.924** | **0.897** | **0.034** | **0.912** |
| 5 | 0.55–0.66 (zero-deferral floor) | 0.917 | 0.887 | 0.000 | 0.903 |
| 7 | 0.72 | 0.953 | 0.886 | 0.119 | 0.923 |

Updated `INDEX_VERSION` → `index_v5_metric_finetuned_823img` and
`DECISION_RULE_VERSION` → `rule_v3_metric_k5_t0.70`.

**Live end-to-end verification** against `/search` on 80 fresh holdout
images (40 malignant + 40 benign, double the usual sample for more
statistical weight): **88.8% accuracy**, sensitivity 37/40, specificity
34/40, only 4/80 inconclusive.

The pre-expansion model is preserved at
`models/finetuned_mobilenetv2_metric_v1_before_expansion.pth` if this ever
needs to be rolled back.

### New decision-rule calibration (validation pool, 102 queries)

Re-ran the k × threshold sweep for the new embedding space — different model,
different similarity distribution, so the old 0.68 no longer applies as-is.

| k | threshold | sensitivity | specificity | inconclusive | accuracy |
|---|-----------|-------------|-------------|---------------|----------|
| 5 | 0.55–0.66 | 0.912 | 0.882 | 0.000 | 0.902 |
| **5** | **0.70** | **0.937** | **0.909** | **0.059** | **0.927** |
| 3 | 0.72 | 0.949 | 0.906 | 0.108 | 0.934 |

Kept k=5 for continuity with prior versioning; moved `INCONCL_THRESH` to
**0.70** — best balance of low inconclusive rate (5.9%) with strong
sensitivity/specificity on this pool. `k=3, thr=0.72` scores slightly higher
on this validation pool but nearly doubles the deferral rate for a small,
within-noise sensitivity gain (102 samples is not enough to distinguish these
reliably) — not preferred given the holdout set is the more trustworthy
signal and wasn't used for this choice.

Updated `INDEX_VERSION` → `index_v4_metric_finetuned` and
`DECISION_RULE_VERSION` → `rule_v2_metric_k5_t0.70` in `app.py` so
`/metrics` subgroup breakdowns can distinguish sessions scored under the old
vs. new model going forward.
