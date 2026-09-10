# Linear Probe vs CBIR Baseline Comparison

This document compares the current non-parametric CBIR (Majority Vote) approach to a trained parametric classifier (Linear Probe via Logistic Regression) on the fixed MobileNetV2 embeddings.

## Methodology
- **Train Set:** 823 samples from the Index Pool.
- **Validation Set:** 176 samples from the Validation Pool.
- **CBIR Setup:** K=5, Similarity Threshold=0.02.
- **Linear Probe Setup:** LogisticRegression (L2 Penalty, max_iter=1000).

## Results

| Metric | CBIR Baseline | Linear Probe | Difference (LP - CBIR) |
|--------|---------------|--------------|------------------------|
| AUC-ROC | 0.940 | 0.925 | -0.015 |
| Sensitivity | 0.760 | 0.865 | +0.104 |
| Specificity | 0.925 | 0.800 | -0.125 |
| Accuracy | 0.835 | 0.835 | +0.000 |

## Conclusion
If the Linear Probe significantly outperforms CBIR, it indicates that the embeddings are linearly separable but the Euclidean/Cosine distance logic of CBIR is sub-optimal. If CBIR is comparable, it validates our instance-based approach which has the added benefit of explainability (showing retrieved images).
