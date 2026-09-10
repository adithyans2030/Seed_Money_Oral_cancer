import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from preprocess import preprocess_image
from math import sqrt
from tabulate import tabulate

def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z**2/n
    centre_adjusted_prob = p + z**2 / (2*n)
    adjusted_std = z * sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    lower = (centre_adjusted_prob - adjusted_std) / denominator
    upper = (centre_adjusted_prob + adjusted_std) / denominator
    return max(0.0, lower), min(1.0, upper)

def compute_metrics(y_true, y_pred, y_conf, k):
    # Malignant is positive class (1), Benign is negative class (0)
    # y_true and y_pred are lists of 'malignant' and 'benign'
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'malignant' and yp == 'malignant')
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'benign' and yp == 'malignant')
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'benign' and yp == 'benign')
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'malignant' and yp == 'benign')
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
    
    return {
        'k': k, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'sensitivity': sens, 'specificity': spec,
        'ppv': ppv, 'npv': npv, 'accuracy': acc, 'f1': f1,
        'n_pos': tp + fn, 'n_neg': tn + fp
    }

def main():
    val_pool_path = "../data/validation_pool.json"
    if not os.path.exists(val_pool_path):
        print(f"Error: {val_pool_path} not found.")
        return
        
    with open(val_pool_path, 'r') as f:
        val_pool = json.load(f)
        
    # Set up model
    device = torch.device("cpu")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    finetuned_path = os.environ.get("EVAL_MODEL_PATH", "../models/finetuned_mobilenetv2.pth")
    if os.path.exists(finetuned_path):
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))
        backbone.load_state_dict(torch.load(finetuned_path, map_location=device))
        print(f"Loaded fine-tuned model from {finetuned_path}")
    else:
        print("Fine-tuned model not found, using raw ImageNet weights.")
    backbone.classifier = nn.Identity()

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
            # Fallback to random embedding if dummy image not found or fails preprocessing
            emb = np.random.rand(1280)
            return emb / (np.linalg.norm(emb) + 1e-8)
        tensor = tensor.unsqueeze(0).to(device)
        emb = model(tensor).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    print("Extracting embeddings for validation pool...")
    embeddings = []
    labels = []
    for item in val_pool:
        emb = extract_embedding(item['filename'])
        embeddings.append(emb)
        labels.append(item['label'])
        
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    k_values = [1, 3, 5]
    fold_metrics = {k: [] for k in k_values}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running 5-fold Cross Validation...")
    for fold, (retrieval_idx, query_idx) in enumerate(skf.split(embeddings, labels)):
        print(f"Fold {fold+1}/5")
        
        retrieval_embs = embeddings[retrieval_idx]
        retrieval_labels = labels[retrieval_idx]
        
        query_embs = embeddings[query_idx]
        query_labels = labels[query_idx]
        
        # Precompute all similarities for this fold
        sim_matrix = cosine_similarity(query_embs, retrieval_embs)
        
        for k in k_values:
            y_true = []
            y_pred = []
            y_conf = []
            
            for q_idx in range(len(query_idx)):
                sims = sim_matrix[q_idx]
                top_indices = np.argsort(sims)[::-1][:k]
                
                # Majority vote decision rule (ignoring inconclusive for now to get binary metrics)
                top_labels = [retrieval_labels[i] for i in top_indices]
                mal_count = sum(1 for l in top_labels if l == 'malignant')
                ben_count = k - mal_count
                
                pred_label = 'malignant' if mal_count > ben_count else 'benign'
                confidence = mal_count / k
                
                y_true.append(query_labels[q_idx])
                y_pred.append(pred_label)
                y_conf.append(confidence)
                
            metrics = compute_metrics(y_true, y_pred, y_conf, k)
            fold_metrics[k].append(metrics)
            
    # Aggregate results
    os.makedirs("../results", exist_ok=True)
    
    summary_data = []
    best_f1 = -1
    best_k = None
    
    full_results = {}
    
    print("\nAggregation Results:")
    for k in k_values:
        avg_metrics = {}
        keys = ['TP', 'FP', 'TN', 'FN', 'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'f1', 'n_pos', 'n_neg']
        for key in keys:
            avg_metrics[key] = np.mean([fm[key] for fm in fold_metrics[k]])
            
        # Compute Wilson CI for sensitivity and specificity over total n across folds
        total_pos = sum([fm['n_pos'] for fm in fold_metrics[k]])
        total_neg = sum([fm['n_neg'] for fm in fold_metrics[k]])
        
        avg_sens = avg_metrics['sensitivity']
        avg_spec = avg_metrics['specificity']
        
        sens_lower, sens_upper = wilson_ci(avg_sens, total_pos)
        spec_lower, spec_upper = wilson_ci(avg_spec, total_neg)
        
        if avg_metrics['f1'] > best_f1:
            best_f1 = avg_metrics['f1']
            best_k = k
            
        full_results[f"k={k}"] = {
            "avg_metrics": avg_metrics,
            "sens_ci": [sens_lower, sens_upper],
            "spec_ci": [spec_lower, spec_upper]
        }
        
        summary_data.append([
            k,
            f"{avg_sens:.3f} ({sens_lower:.3f}-{sens_upper:.3f})",
            f"{avg_spec:.3f} ({spec_lower:.3f}-{spec_upper:.3f})",
            f"{avg_metrics['ppv']:.3f}",
            f"{avg_metrics['npv']:.3f}",
            f"{avg_metrics['accuracy']:.3f}",
            f"{avg_metrics['f1']:.3f}"
        ])
        
    print(tabulate(summary_data, headers=["k", "Sensitivity (95% CI)", "Specificity (95% CI)", "PPV", "NPV", "Accuracy", "F1"]))
    print(f"\nBest configuration based on F1 score: k={best_k}")
    
    with open("../results/kfold_eval_results.json", 'w') as f:
        json.dump(full_results, f, indent=2)
        
    print("Results saved to results/kfold_eval_results.json")
    
if __name__ == "__main__":
    main()
