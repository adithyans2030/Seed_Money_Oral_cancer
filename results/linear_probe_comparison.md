# Linear Probe vs CBIR Baseline Comparison

This document compares the current non-parametric CBIR (Majority Vote) approach to a trained parametric classifier (Linear Probe via Logistic Regression) on the fixed MobileNetV2 embeddings.

## Methodology
- **Train Set:** 910 samples from the Index Pool.
- **Validation Set:** 195 samples from the Validation Pool.
- **CBIR Setup:** K=5, Similarity Threshold=0.02.
- **Linear Probe Setup:** LogisticRegression (L2 Penalty, max_iter=1000).

## Results

| Metric | CBIR Baseline | Linear Probe | Difference (LP - CBIR) |
|--------|---------------|--------------|------------------------|
| AUC-ROC | 0.429 | 0.466 | +0.037 |
| Sensitivity | 0.000 | 1.000 | +1.000 |
| Specificity | 1.000 | 0.000 | -1.000 |
| Accuracy | 0.251 | 0.749 | +0.497 |

## Conclusion
If the Linear Probe significantly outperforms CBIR, it indicates that the embeddings are linearly separable but the Euclidean/Cosine distance logic of CBIR is sub-optimal. If CBIR is comparable, it validates our instance-based approach which has the added benefit of explainability (showing retrieved images).
