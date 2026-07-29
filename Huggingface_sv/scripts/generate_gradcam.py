import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import preprocess_image
from sklearn.metrics.pairwise import cosine_similarity

def get_gradcam_heatmap(model, target_layer, tensor_img, target_embedding):
    """
    Computes Grad-CAM for the cosine similarity between the image and target_embedding.
    """
    model.eval()
    
    # Hooks
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
        
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    f_hook = target_layer.register_forward_hook(forward_hook)
    b_hook = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    tensor_img.requires_grad = True
    
    features = model.features(tensor_img)
    pooled = nn.AdaptiveAvgPool2d((1, 1))(features)
    query_emb = nn.Flatten()(pooled)
    query_emb_norm = F.normalize(query_emb, p=2, dim=1)
    
    target_emb_tensor = torch.tensor(target_embedding, dtype=torch.float32, device=query_emb_norm.device).unsqueeze(0)
    target_emb_norm = F.normalize(target_emb_tensor, p=2, dim=1)
    
    # Scalar to maximize: cosine similarity
    score = (query_emb_norm * target_emb_norm).sum()
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    f_hook.remove()
    b_hook.remove()
    
    if len(gradients) == 0 or len(activations) == 0:
        return None
        
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
        
    import cv2
    cam = cv2.resize(cam, (tensor_img.shape[3], tensor_img.shape[2]))
    return cam

def overlay_heatmap(img_path, heatmap):
    import cv2
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img, 0.6, colormap, 0.4, 0)
    return superimposed_img

def main():
    try:
        import cv2
    except ImportError:
        print("Please run: pip install opencv-python")
        return
        
    index_path = "../data/index_pool.json"
    val_path = "../data/validation_pool.json"
    
    if not os.path.exists(index_path):
        return
        
    with open(index_path, 'r') as f: index_pool = json.load(f)
    with open(val_path, 'r') as f: val_pool = json.load(f)
        
    device = torch.device("cpu")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = backbone.to(device)
    model.eval()
    
    target_layer = model.features[-1]
    
    # 1. We need malignant embeddings from the index to act as targets
    # For simplicity, we can load cbir_index.npz and pick the first malignant one,
    # or just extract one.
    idx_npz = "cbir_index.npz"
    if not os.path.exists(idx_npz):
        print("Missing cbir_index.npz")
        return
        
    data = np.load(idx_npz, allow_pickle=True)
    idx_embs = data['embeddings']
    idx_labels = data['labels']
    
    mal_embs = idx_embs[idx_labels == 'malignant']
    if len(mal_embs) == 0:
        print("No malignant embeddings found in index.")
        return
        
    out_dir = "../results/gradcam"
    os.makedirs(out_dir, exist_ok=True)
    
    # Pick 5 random malignant query images from val set
    val_mal = [item for item in val_pool if item['label'] == 'malignant']
    random.seed(42)
    sample_queries = random.sample(val_mal, min(5, len(val_mal)))
    
    # We also need a full model sequential for standard extraction
    extract_model = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(device)
    extract_model.eval()
    
    @torch.no_grad()
    def get_emb(img_path):
        t = preprocess_image(img_path)
        if t is None: return None
        return extract_model(t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        
    for i, query in enumerate(sample_queries):
        img_path = query['filename']
        tensor_img = preprocess_image(img_path)
        if tensor_img is None: continue
        
        # Find best malignant match in index
        q_emb = get_emb(img_path)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        sims = cosine_similarity([q_emb], mal_embs)[0]
        best_match_idx = np.argmax(sims)
        target_emb = mal_embs[best_match_idx]
        
        tensor_img = tensor_img.unsqueeze(0).to(device)
        heatmap = get_gradcam_heatmap(model, target_layer, tensor_img, target_emb)
        
        if heatmap is not None:
            overlay = overlay_heatmap(img_path, heatmap)
            if overlay is not None:
                out_file = os.path.join(out_dir, f"gradcam_mal_{i}.png")
                
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax[0].imshow(img)
                ax[0].set_title("Original")
                ax[0].axis("off")
                
                ax[1].imshow(overlay)
                ax[1].set_title("Grad-CAM (Similarity to Match)")
                ax[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(out_file)
                plt.close()
                print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
