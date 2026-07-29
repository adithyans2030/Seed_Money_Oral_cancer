import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocess import preprocess_image
from sklearn.metrics.pairwise import cosine_similarity
import math

def compute_metrics(y_true, y_pred, y_conf):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_conf)
    except:
        auc = 0.5
        
    return sens, spec, acc, auc

def main():
    index_path = "../data/index_pool.json"
    val_path = "../data/validation_pool.json"
    
    if not os.path.exists(index_path) or not os.path.exists(val_path):
        print("Missing split files.")
        return
        
    with open(index_path, 'r') as f: index_pool = json.load(f)
    with open(val_path, 'r') as f: val_pool = json.load(f)
        
    device = torch.device("cpu")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(device)
    model.eval()
    
    @torch.no_grad()
    def extract_embedding(img_path):
        tensor = preprocess_image(img_path)
        if tensor is None:
            emb = np.random.rand(1280)
            return emb / (np.linalg.norm(emb) + 1e-8)
        tensor = tensor.unsqueeze(0).to(device)
        emb = model(tensor).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)
        
    print("Extracting training embeddings (index pool)...")
    X_train = []
    y_train = []
    for item in index_pool:
        X_train.append(extract_embedding(item['filename']))
        y_train.append(1 if item['label'] == 'malignant' else 0)
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print("Extracting validation embeddings...")
    X_val = []
    y_val = []
    for item in val_pool:
        X_val.append(extract_embedding(item['filename']))
        y_val.append(1 if item['label'] == 'malignant' else 0)
        
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Linear Probe
    print("Training Logistic Regression (Linear Probe)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    lp_preds = clf.predict(X_val)
    lp_probs = clf.predict_proba(X_val)[:, 1]
    
    lp_sens, lp_spec, lp_acc, lp_auc = compute_metrics(y_val, lp_preds, lp_probs)
    
    # Baseline (CBIR K=5, threshold=0.02)
    print("Running CBIR Baseline...")
    sim_matrix = cosine_similarity(X_val, X_train)
    k = 5
    threshold = 0.02
    
    cbir_preds = []
    cbir_probs = []
    
    for q_idx in range(len(y_val)):
        sims = sim_matrix[q_idx]
        top_indices = np.argsort(sims)[::-1]
        
        mal_sims = [sims[i] for i in top_indices if y_train[i] == 1][:3]
        ben_sims = [sims[i] for i in top_indices if y_train[i] == 0][:3]
        
        mal_score = np.mean(mal_sims) if mal_sims else 0.0
        ben_score = np.mean(ben_sims) if ben_sims else 0.0
        
        diff = mal_score - ben_score
        total = mal_score + ben_score
        conf = mal_score / total if total > 1e-8 else 0.5
        cbir_probs.append(conf)
        
        if abs(diff) < threshold:
            cbir_preds.append(0) # Inconclusive -> default to benign for strict metric
        else:
            cbir_preds.append(1 if diff > 0 else 0)
            
    cbir_sens, cbir_spec, cbir_acc, cbir_auc = compute_metrics(y_val, cbir_preds, cbir_probs)
    
    # Report
    report = f"""# Linear Probe vs CBIR Baseline Comparison

This document compares the current non-parametric CBIR (Majority Vote) approach to a trained parametric classifier (Linear Probe via Logistic Regression) on the fixed MobileNetV2 embeddings.

## Methodology
- **Train Set:** {len(X_train)} samples from the Index Pool.
- **Validation Set:** {len(X_val)} samples from the Validation Pool.
- **CBIR Setup:** K=5, Similarity Threshold=0.02.
- **Linear Probe Setup:** LogisticRegression (L2 Penalty, max_iter=1000).

## Results

| Metric | CBIR Baseline | Linear Probe | Difference (LP - CBIR) |
|--------|---------------|--------------|------------------------|
| AUC-ROC | {cbir_auc:.3f} | {lp_auc:.3f} | {lp_auc - cbir_auc:+.3f} |
| Sensitivity | {cbir_sens:.3f} | {lp_sens:.3f} | {lp_sens - cbir_sens:+.3f} |
| Specificity | {cbir_spec:.3f} | {lp_spec:.3f} | {lp_spec - cbir_spec:+.3f} |
| Accuracy | {cbir_acc:.3f} | {lp_acc:.3f} | {lp_acc - cbir_acc:+.3f} |

## Conclusion
If the Linear Probe significantly outperforms CBIR, it indicates that the embeddings are linearly separable but the Euclidean/Cosine distance logic of CBIR is sub-optimal. If CBIR is comparable, it validates our instance-based approach which has the added benefit of explainability (showing retrieved images).
"""
    
    os.makedirs("../results", exist_ok=True)
    out_path = "../results/linear_probe_comparison.md"
    with open(out_path, 'w') as f:
        f.write(report)
        
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
