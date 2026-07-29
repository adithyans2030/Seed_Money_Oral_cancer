import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from preprocess import preprocess_image
import tabulate

def main():
    val_pool_path = "../data/validation_pool.json"
    if not os.path.exists(val_pool_path):
        print(f"Error: {val_pool_path} not found.")
        return
        
    with open(val_pool_path, 'r') as f:
        val_pool = json.load(f)
        
    # Set up model (for demo, we will embed and then retrieve from the same set, 
    # but excluding self, simulating a full index pool retrieval. However, since the 
    # instruction says use validation pool, we'll embed the validation pool and retrieve 
    # from the index pool or just do CV. Let's do LOOCV on the validation pool for tuning).
    # Wait, the instruction doesn't specify fold structure for tuning. Let's load the index pool too for retrieval!
    
    with open("../data/index_pool.json", 'r') as f:
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

    print("Extracting embeddings for index pool (retrieval)...")
    idx_embs = []
    idx_labels = []
    for item in index_pool:
        idx_embs.append(extract_embedding(item['filename']))
        idx_labels.append(item['label'])
    idx_embs = np.array(idx_embs)
    idx_labels = np.array(idx_labels)

    print("Extracting embeddings for validation pool (queries)...")
    val_embs = []
    val_labels = []
    for item in val_pool:
        val_embs.append(extract_embedding(item['filename']))
        val_labels.append(item['label'])
    val_embs = np.array(val_embs)
    val_labels = np.array(val_labels)

    sim_matrix = cosine_similarity(val_embs, idx_embs)
    
    # We will use the production k (hardcoded here as 5 if not passed, but we should use 5)
    k = 5 
    
    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    results = []
    
    for thresh in thresholds:
        tp = fp = tn = fn = inconc = 0
        
        for q_idx in range(len(val_labels)):
            sims = sim_matrix[q_idx]
            top_indices = np.argsort(sims)[::-1]
            
            # Get top k malignant and top k benign specifically for the difference rule
            mal_sims = [sims[i] for i in top_indices if idx_labels[i] == 'malignant'][:3]
            ben_sims = [sims[i] for i in top_indices if idx_labels[i] == 'benign'][:3]
            
            mal_score = np.mean(mal_sims) if mal_sims else 0.0
            ben_score = np.mean(ben_sims) if ben_sims else 0.0
            
            diff = mal_score - ben_score
            true_label = val_labels[q_idx]
            
            if abs(diff) < thresh:
                inconc += 1
                continue # Inconclusive cases are excluded from sensitivity/specificity
            
            pred_label = 'malignant' if diff > 0 else 'benign'
            
            if true_label == 'malignant' and pred_label == 'malignant':
                tp += 1
            elif true_label == 'benign' and pred_label == 'malignant':
                fp += 1
            elif true_label == 'benign' and pred_label == 'benign':
                tn += 1
            elif true_label == 'malignant' and pred_label == 'benign':
                fn += 1
                
        definitive = tp + fn + tn + fp
        inconc_rate = inconc / len(val_labels)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
        
        results.append({
            'threshold': thresh,
            'sens': sens,
            'spec': spec,
            'inconc_rate': inconc_rate,
            'f1': f1
        })
        
    # Plotting
    import matplotlib
    matplotlib.use('Agg')
    
    t_vals = [r['threshold'] for r in results]
    sens_vals = [r['sens'] for r in results]
    spec_vals = [r['spec'] for r in results]
    inconc_vals = [r['inconc_rate'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, sens_vals, marker='o', label='Sensitivity')
    plt.plot(t_vals, spec_vals, marker='s', label='Specificity')
    plt.plot(t_vals, inconc_vals, marker='^', linestyle='--', label='Inconclusive Rate')
    plt.axhline(y=0.20, color='r', linestyle=':', label='Max Inconclusive (20%)')
    
    plt.xlabel('Inconclusive Threshold')
    plt.ylabel('Metric Score / Rate')
    plt.title('Threshold Tuning on Validation Pool')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/threshold_tuning.png")
    
    # Recommend best
    valid_results = [r for r in results if r['inconc_rate'] < 0.20]
    if valid_results:
        best = max(valid_results, key=lambda x: x['f1'])
        print(f"\nRecommended Threshold: {best['threshold']} (F1: {best['f1']:.3f}, Inconclusive: {best['inconc_rate']*100:.1f}%)")
    else:
        print("\nNo threshold kept inconclusive rate < 20%. Using default 0.08.")
        best = next(r for r in results if r['threshold'] == 0.08)
        
    summary_data = [[r['threshold'], f"{r['sens']:.3f}", f"{r['spec']:.3f}", f"{r['inconc_rate']:.3f}", f"{r['f1']:.3f}"] for r in results]
    print(tabulate.tabulate(summary_data, headers=["Threshold", "Sensitivity", "Specificity", "Inconclusive Rate", "F1 on Definitive"]))

if __name__ == "__main__":
    main()
