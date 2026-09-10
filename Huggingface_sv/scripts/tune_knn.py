import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_image
import tabulate

def main():
    val_pool_path = "../data/validation_pool.json"
    index_pool_path = "../data/index_pool.json"
    
    with open(val_pool_path, 'r') as f:
        val_pool = json.load(f)
    with open(index_pool_path, 'r') as f:
        index_pool = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    finetuned_path = "../models/finetuned_mobilenetv2.pth"
    if os.path.exists(finetuned_path):
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )
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
    
    ks = [5]
    thresholds = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
    
    results = []
    
    for k in ks:
        for thresh in thresholds:
            tp = fp = tn = fn = inconc = 0
            
            for q_idx in range(len(val_labels)):
                sims = sim_matrix[q_idx]
                top_indices = np.argsort(sims)[::-1][:k]
                
                # Majority vote
                votes = [idx_labels[i] for i in top_indices]
                mal_count = votes.count('malignant')
                ben_count = votes.count('benign')
                
                pred_label = 'malignant' if mal_count > ben_count else 'benign'
                
                # Confidence score: mean similarity of top-k matches for the winning class
                winning_sims = [sims[i] for i in top_indices if idx_labels[i] == pred_label]
                confidence = np.mean(winning_sims) if winning_sims else 0.0
                
                true_label = val_labels[q_idx]
                
                if confidence < thresh:
                    inconc += 1
                    continue
                
                if true_label == 'malignant' and pred_label == 'malignant':
                    tp += 1
                elif true_label == 'benign' and pred_label == 'malignant':
                    fp += 1
                elif true_label == 'benign' and pred_label == 'benign':
                    tn += 1
                elif true_label == 'malignant' and pred_label == 'benign':
                    fn += 1
                    
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            inconc_rate = inconc / len(val_labels)
            f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
            
            results.append([k, thresh, sens, spec, inconc_rate, f1])
            
    print(tabulate.tabulate(results, headers=["K", "Threshold", "Sensitivity", "Specificity", "Inconclusive Rate", "F1 on Definitive"]))

if __name__ == "__main__":
    main()
