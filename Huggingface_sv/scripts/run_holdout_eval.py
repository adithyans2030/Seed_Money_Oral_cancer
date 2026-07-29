import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from preprocess import preprocess_image
from math import sqrt
import tabulate

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z**2/n
    centre_adjusted_prob = p + z**2 / (2*n)
    adjusted_std = z * sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    lower = (centre_adjusted_prob - adjusted_std) / denominator
    upper = (centre_adjusted_prob + adjusted_std) / denominator
    return max(0.0, lower), min(1.0, upper)

def compute_metrics(y_true, y_pred, y_conf):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'malignant' and yp == 'malignant')
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'benign' and yp == 'malignant')
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'benign' and yp == 'benign')
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'malignant' and yp == 'benign')
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    y_true_binary = [1 if yt == 'malignant' else 0 for yt in y_true]
    try:
        auc = roc_auc_score(y_true_binary, y_conf)
    except ValueError:
        auc = 0.5
        
    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'sensitivity': sens, 'specificity': spec,
        'ppv': ppv, 'npv': npv, 'accuracy': acc, 'auc': auc,
        'n_pos': tp + fn, 'n_neg': tn + fp
    }

def main():
    holdout_pool_path = "../data/holdout_pool.json"
    index_pool_path = "../data/index_pool.json"
    
    if not os.path.exists(holdout_pool_path):
        print(f"Error: {holdout_pool_path} not found.")
        return
        
    with open(holdout_pool_path, 'r') as f:
        holdout_pool = json.load(f)
        
    with open(index_pool_path, 'r') as f:
        index_pool = json.load(f)

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

    print("Extracting embeddings for index pool (production retrieval)...")
    idx_embs = []
    idx_labels = []
    for item in index_pool:
        idx_embs.append(extract_embedding(item['filename']))
        idx_labels.append(item['label'])
    idx_embs = np.array(idx_embs)
    idx_labels = np.array(idx_labels)

    print("Extracting embeddings for holdout pool (evaluation)...")
    holdout_embs = []
    holdout_labels = []
    for item in holdout_pool:
        holdout_embs.append(extract_embedding(item['filename']))
        holdout_labels.append(item['label'])
    holdout_embs = np.array(holdout_embs)
    holdout_labels = np.array(holdout_labels)

    sim_matrix = cosine_similarity(holdout_embs, idx_embs)
    
    # Final parameters derived from tuning
    k = 5
    threshold = 0.02
    
    y_true = []
    y_pred = []
    y_conf = []
    y_naive = []
    
    # Determine majority class for naive baseline
    mal_count = sum(1 for l in idx_labels if l == 'malignant')
    majority_class = 'malignant' if mal_count > len(idx_labels)/2 else 'benign'
    
    inconc_count = 0
    
    for q_idx in range(len(holdout_labels)):
        sims = sim_matrix[q_idx]
        top_indices = np.argsort(sims)[::-1]
        
        mal_sims = [sims[i] for i in top_indices if idx_labels[i] == 'malignant'][:3]
        ben_sims = [sims[i] for i in top_indices if idx_labels[i] == 'benign'][:3]
        
        mal_score = np.mean(mal_sims) if mal_sims else 0.0
        ben_score = np.mean(ben_sims) if ben_sims else 0.0
        
        diff = mal_score - ben_score
        true_label = holdout_labels[q_idx]
        
        total = mal_score + ben_score
        conf = mal_score / total if total > 1e-8 else 0.5
        
        y_true.append(true_label)
        y_naive.append(majority_class)
        y_conf.append(conf)
        
        if abs(diff) < threshold:
            inconc_count += 1
            # For strict evaluation metrics on definitive cases, we will skip inconclusive
            # Or we can treat inconclusive as false predictions. Let's just predict majority class for them
            # so the model isn't given a free pass on inconclusive cases for final holdout metrics.
            y_pred.append(majority_class)
        else:
            pred_label = 'malignant' if diff > 0 else 'benign'
            y_pred.append(pred_label)
            
    metrics = compute_metrics(y_true, y_pred, y_conf)
    
    # Wilson CIs
    sens_lower, sens_upper = wilson_ci(metrics['sensitivity'], metrics['n_pos'])
    spec_lower, spec_upper = wilson_ci(metrics['specificity'], metrics['n_neg'])
    acc_lower, acc_upper = wilson_ci(metrics['accuracy'], len(y_true))
    ppv_lower, ppv_upper = wilson_ci(metrics['ppv'], metrics['TP'] + metrics['FP'])
    npv_lower, npv_upper = wilson_ci(metrics['npv'], metrics['TN'] + metrics['FN'])
    
    # McNemar's Test
    # contingency table: 
    #                Naive correct    Naive wrong
    # Model correct       a                b
    # Model wrong         c                d
    a = b = c = d = 0
    for yt, yp, yn in zip(y_true, y_pred, y_naive):
        model_correct = (yt == yp)
        naive_correct = (yt == yn)
        if model_correct and naive_correct: a += 1
        elif model_correct and not naive_correct: b += 1
        elif not model_correct and naive_correct: c += 1
        elif not model_correct and not naive_correct: d += 1
        
    table = [[a, b], [c, d]]
    try:
        mcnemar_result = mcnemar(table, exact=False, correction=True)
        p_value = mcnemar_result.pvalue
    except:
        p_value = 1.0
        
    final_report = {
        'metrics': metrics,
        'CIs': {
            'sensitivity': [sens_lower, sens_upper],
            'specificity': [spec_lower, spec_upper],
            'accuracy': [acc_lower, acc_upper],
            'ppv': [ppv_lower, ppv_upper],
            'npv': [npv_lower, npv_upper]
        },
        'mcnemar_p_value': p_value,
        'inconclusive_rate': inconc_count / len(y_true)
    }
    
    os.makedirs("../results", exist_ok=True)
    with open("../results/holdout_eval_final.json", 'w') as f:
        json.dump(final_report, f, indent=2)
        
    print("\nFINAL CLINICAL METRICS REPORT (HOLDOUT SET)")
    print("="*60)
    summary = [
        ["Sensitivity", f"{metrics['sensitivity']:.3f} ({sens_lower:.3f} - {sens_upper:.3f})"],
        ["Specificity", f"{metrics['specificity']:.3f} ({spec_lower:.3f} - {spec_upper:.3f})"],
        ["PPV", f"{metrics['ppv']:.3f} ({ppv_lower:.3f} - {ppv_upper:.3f})"],
        ["NPV", f"{metrics['npv']:.3f} ({npv_lower:.3f} - {npv_upper:.3f})"],
        ["Accuracy", f"{metrics['accuracy']:.3f} ({acc_lower:.3f} - {acc_upper:.3f})"],
        ["AUC-ROC", f"{metrics['auc']:.3f}"],
        ["Inconclusive Rate", f"{inconc_count / len(y_true):.3f}"]
    ]
    print(tabulate.tabulate(summary, headers=["Metric", "Value (95% CI)"]))
    print(f"\nMcNemar's test vs Naive Baseline p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Model is significantly different from naive baseline.")
    else:
        print("Result: Model is NOT significantly different from naive baseline.")
    print("="*60)

if __name__ == "__main__":
    main()
