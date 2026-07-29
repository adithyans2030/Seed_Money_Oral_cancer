import os
import zipfile
import tempfile
import urllib.request
import io
import numpy as np
import imagehash
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import time
from urllib.error import URLError

DATASET_B_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral Cancer\Oral Cancer Dataset"
OLD_INDEX_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\Huggingface_sv\cbir_index.npz"
BASE_DIR = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\Huggingface_sv"

IMG_SIZE = 224

def strip_exif(image: Image.Image) -> Image.Image:
    data_copy = image.tobytes()
    clean = Image.frombytes(image.mode, image.size, data_copy)
    return clean

def main():
    print("=== Step 1: Loading & Deduplicating Datasets ===")
    
    unique_images = {}

    print("-> Loading Dataset A from HuggingFace...")
    old_data = np.load(OLD_INDEX_PATH, allow_pickle=True)
    old_paths = old_data["paths"]
    old_labels = old_data["labels"]
    
    for i, path in enumerate(old_paths):
        path_str = str(path).replace("\\", "/")
        parts = path_str.split("/")
        class_folder = parts[-2]
        filename = parts[-1]
        url = f"https://huggingface.co/datasets/GPrabhanjana/oral-images/resolve/main/{class_folder}/{filename}"
        
        success = False
        for attempt in range(3):
            try:
                req = urllib.request.urlopen(url, timeout=10)
                img_bytes = req.read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = strip_exif(img)
                
                label_int = 0 if str(old_labels[i]) == "benign" else 1
                resolution = img.width * img.height
                img_hash = str(imagehash.phash(img))
                
                if img_hash not in unique_images or resolution > unique_images[img_hash]["resolution"]:
                    unique_images[img_hash] = {
                        "image": img,
                        "label": label_int,
                        "path": path_str,
                        "resolution": resolution
                    }
                success = True
                break
            except Exception as e:
                time.sleep(1)
        if not success:
            print(f"Failed to download HF image {url} after 3 attempts.")

    print(f"-> Loading Dataset B from {DATASET_B_PATH}...")
    for root, _, files in os.walk(DATASET_B_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path_str = os.path.join(root, file).replace("\\", "/")
                
                if "NON CANCER" in path_str.upper() or "NON_CANCER" in path_str.upper() or "BENIGN" in path_str.upper():
                    label_int = 0
                elif "CANCER" in path_str.upper() or "MALIGNANT" in path_str.upper():
                    label_int = 1
                else:
                    continue 
                
                try:
                    img = Image.open(path_str).convert("RGB")
                    img = strip_exif(img)
                    resolution = img.width * img.height
                    img_hash = str(imagehash.phash(img))
                    
                    if img_hash not in unique_images or resolution > unique_images[img_hash]["resolution"]:
                        unique_images[img_hash] = {
                            "image": img,
                            "label": label_int,
                            "path": f"new_data/{'malignant' if label_int == 1 else 'benign'}/{file}",
                            "resolution": resolution
                        }
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    dataset = list(unique_images.values())
    print(f"Total unique images after deduplication: {len(dataset)}")
    
    print("\n=== Step 2 & 5: Standardizing & Extracting Embeddings ===")
    device = torch.device("cpu")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_embeddings = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            if idx % 100 == 0:
                print(f"Extracted {idx}/{len(dataset)}...")
            tensor = transform(item["image"]).unsqueeze(0).to(device)
            emb = model(tensor).squeeze().cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            all_embeddings.append(emb)
            all_paths.append(item["path"])
            all_labels.append(item["label"])

    all_embeddings = np.array(all_embeddings)
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    print("\n=== Step 4: Stratified Split (70/15/15) ===")
    indices = np.arange(len(all_labels))
    
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Index size (70%): {len(idx_train)} images")
    print(f"Val size (15%): {len(idx_val)} images")
    print(f"Test size (15%): {len(idx_test)} images")

    print("\n=== Step 5: Saving Rebuilt Indexes ===")
    np.savez(os.path.join(BASE_DIR, "cbir_index.npz"),
             embeddings=all_embeddings[idx_train],
             paths=all_paths[idx_train],
             labels=all_labels[idx_train])
             
    np.savez(os.path.join(BASE_DIR, "cbir_val.npz"),
             embeddings=all_embeddings[idx_val],
             paths=all_paths[idx_val],
             labels=all_labels[idx_val])
             
    np.savez(os.path.join(BASE_DIR, "cbir_test.npz"),
             embeddings=all_embeddings[idx_test],
             paths=all_paths[idx_test],
             labels=all_labels[idx_test])

    print("Strict merge & rebuild completed successfully!")

if __name__ == "__main__":
    main()
