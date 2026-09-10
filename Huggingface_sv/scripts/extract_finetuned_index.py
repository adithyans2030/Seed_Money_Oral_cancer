import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from preprocess import preprocess_image

def main():
    print("=== Extracting Fine-Tuned Embeddings for CBIR Index ===")
    index_pool_path = "../data/index_pool.json"
    with open(index_pool_path, 'r') as f:
        dataset = json.load(f)
        
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
        print("Fine-tuned model not found! Using raw ImageNet weights.")
        
    backbone.classifier = nn.Identity()
    
    model = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(device)
    model.eval()

    all_embeddings = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"Extracted {idx}/{len(dataset)}...")
                
            path = item["filename"]
            label = item["label"]
            
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess_image(img)
                if tensor is None:
                    continue
                tensor = tensor.unsqueeze(0).to(device)
                
                emb = model(tensor).squeeze().cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2 normalization
                
                all_embeddings.append(emb)
                all_paths.append(path)
                all_labels.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    np.savez("../cbir_index_finetuned.npz", embeddings=np.array(all_embeddings), paths=np.array(all_paths), labels=np.array(all_labels))
    print(f"Saved {len(all_embeddings)} embeddings to cbir_index_finetuned.npz")

if __name__ == "__main__":
    main()
