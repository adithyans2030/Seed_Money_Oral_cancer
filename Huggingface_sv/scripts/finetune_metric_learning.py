"""
Metric-learning fine-tune for the CBIR embedding model.

The deployed system does k-NN cosine-similarity retrieval, but the current
finetuned_mobilenetv2.pth was trained with a 2-class classification head
(cross-entropy) and its penultimate features are reused as embeddings after
the fact — a proxy objective, not what's actually optimized for retrieval.

This script instead fine-tunes directly for retrieval using batch-hard
triplet loss on cosine distance, warm-started from the existing classification
fine-tune (not from scratch) so it refines rather than discards what already
works. Saves to a NEW file — finetuned_mobilenetv2_metric.pth — so it can be
evaluated against the current model before replacing anything.
"""
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_CLASSES_PER_SIDE = 8   # images per class per batch -> batch size 16
EPOCHS = 25
PATIENCE = 6
MARGIN = 0.3
LR = 1e-4


class OralCancerDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if augment else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        from PIL import Image
        item = self.data_list[idx]
        label = 1 if item['label'] == 'malignant' else 0
        try:
            img = Image.open(item['filename']).convert('RGB')
            if self.aug_transform:
                img = self.aug_transform(img)
            tensor = preprocess_image(img)
            if tensor is None:
                tensor = torch.zeros(3, 224, 224)
        except Exception as e:
            print(f"Error loading {item['filename']}: {e}")
            tensor = torch.zeros(3, 224, 224)
        return tensor, label


def build_model(device, warm_start_path):
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if warm_start_path and os.path.exists(warm_start_path):
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))
        backbone.load_state_dict(torch.load(warm_start_path, map_location=device))
        print(f"Warm-started from {warm_start_path}")
    backbone.classifier = nn.Identity()
    backbone.to(device)

    # Same unfreezing strategy as the classification fine-tune: only the last
    # 3 feature blocks are trainable, everything else stays frozen.
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.features[-3:].parameters():
        param.requires_grad = True

    return backbone


def embed_batch(backbone, x):
    feats = backbone.features(x)
    pooled = F.adaptive_avg_pool2d(feats, (1, 1))
    emb = torch.flatten(pooled, 1)
    return F.normalize(emb, p=2, dim=1)


def cosine_dist(a, b):
    return 1.0 - (a * b).sum(dim=-1)


def batch_hard_triplet_loss(embeddings, labels, margin):
    """Standard batch-hard mining: hardest positive + hardest negative per anchor."""
    n = embeddings.size(0)
    dist = 1.0 - embeddings @ embeddings.T  # cosine distance matrix, [n, n]
    labels = labels.unsqueeze(0)
    same = (labels == labels.T)
    diff = ~same
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same_no_self = same & ~eye

    losses = []
    for i in range(n):
        pos_dists = dist[i][same_no_self[i]]
        neg_dists = dist[i][diff[i]]
        if pos_dists.numel() == 0 or neg_dists.numel() == 0:
            continue
        hardest_pos = pos_dists.max()
        hardest_neg = neg_dists.min()
        losses.append(F.relu(hardest_pos - hardest_neg + margin))
    if not losses:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return torch.stack(losses).mean()


def make_balanced_batches(pool, per_side):
    mal = [i for i, item in enumerate(pool) if item['label'] == 'malignant']
    ben = [i for i, item in enumerate(pool) if item['label'] == 'benign']
    random.shuffle(mal)
    random.shuffle(ben)
    batches = []
    n_batches = max(len(mal) // per_side, len(ben) // per_side)
    for b in range(n_batches):
        m = [mal[(b * per_side + k) % len(mal)] for k in range(per_side)]
        n = [ben[(b * per_side + k) % len(ben)] for k in range(per_side)]
        batches.append(m + n)
    return batches


@torch.no_grad()
def compute_val_auc(backbone, device, index_pool, val_pool):
    backbone.eval()

    def embed_pool(pool):
        embs, labels = [], []
        for item in pool:
            from PIL import Image
            img = Image.open(item['filename']).convert('RGB')
            tensor = preprocess_image(img)
            if tensor is None:
                continue
            tensor = tensor.unsqueeze(0).to(device)
            e = embed_batch(backbone, tensor).cpu().numpy()[0]
            embs.append(e)
            labels.append(item['label'])
        return np.array(embs), np.array(labels)

    idx_embs, idx_labels = embed_pool(index_pool)
    val_embs, val_labels = embed_pool(val_pool)
    sims = cosine_similarity(val_embs, idx_embs)

    k = 5
    confs = []
    for row in sims:
        top_idx = np.argsort(row)[::-1][:k]
        top_labels = idx_labels[top_idx]
        mal_sims = row[top_idx][top_labels == 'malignant']
        conf = mal_sims.mean() if len(mal_sims) > 0 else 0.0
        confs.append(conf)

    y_true = (val_labels == 'malignant').astype(int)
    try:
        auc = roc_auc_score(y_true, confs)
    except ValueError:
        auc = 0.5
    backbone.train()
    return auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("../data/index_pool.json") as f:
        index_pool = json.load(f)
    with open("../data/validation_pool.json") as f:
        val_pool = json.load(f)

    print(f"Train (index) pool: {len(index_pool)}, Validation pool: {len(val_pool)}")

    warm_start = "../models/finetuned_mobilenetv2.pth"
    backbone = build_model(device, warm_start)

    baseline_auc = compute_val_auc(backbone, device, index_pool, val_pool)
    print(f"Baseline (warm-start, before triplet training) validation AUC: {baseline_auc:.4f}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, backbone.parameters()), lr=LR
    )

    train_ds = OralCancerDataset(index_pool, augment=True)

    best_auc = baseline_auc
    best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        backbone.train()
        batches = make_balanced_batches(index_pool, BATCH_CLASSES_PER_SIDE)
        epoch_loss = 0.0
        n_batches = 0

        for batch_indices in batches:
            imgs = []
            labels = []
            for i in batch_indices:
                tensor, label = train_ds[i]
                imgs.append(tensor)
                labels.append(label)
            x = torch.stack(imgs).to(device)
            y = torch.tensor(labels, device=device)

            optimizer.zero_grad()
            embeddings = embed_batch(backbone, x)
            loss = batch_hard_triplet_loss(embeddings, y, MARGIN)
            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_auc = compute_val_auc(backbone, device, index_pool, val_pool)
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (best val_auc={best_auc:.4f})")
            break

    print(f"\nBest validation AUC achieved: {best_auc:.4f} (baseline was {baseline_auc:.4f})")

    backbone.load_state_dict(best_state)
    # Re-attach a (randomly-initialised, unused-at-inference) classifier head so the
    # saved state_dict matches the format app.py's loader expects — only the
    # features.* weights actually matter, and those come from best_state below.
    save_backbone = models.mobilenet_v2(weights=None)
    num_ftrs = save_backbone.classifier[1].in_features
    save_backbone.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2))
    # Copy over the trained feature weights; classifier head is untrained/unused (Identity at inference).
    save_state = save_backbone.state_dict()
    for k, v in best_state.items():
        if k in save_state and save_state[k].shape == v.shape:
            save_state[k] = v
    save_path = "../models/finetuned_mobilenetv2_metric.pth"
    torch.save(save_state, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
