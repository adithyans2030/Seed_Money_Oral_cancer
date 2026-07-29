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
import random

ZIP_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral Cancer.zip"
OLD_INDEX_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\Huggingface_sv\cbir_index.npz"
NEW_INDEX_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\Huggingface_sv\cbir_index.npz"

IMG_SIZE = 224

def strip_exif(image: Image.Image) -> Image.Image:
    data_copy = image.tobytes()
    clean = Image.frombytes(image.mode, image.size, data_copy)
    return clean

def main():
    print("Loading old index to get existing HF image paths...")
    old_data = np.load(OLD_INDEX_PATH, allow_pickle=True)
    old_paths = old_data["paths"]
    old_labels = old_data["labels"]
    
    # We will store (image, label, path_str)
    all_valid_images = []
    
    print(f"Downloading {len(old_paths)} existing HF images for deduplication and re-extraction...")
    existing_hashes = set()
    
    for i, path in enumerate(old_paths):
        path_str = str(path).replace("\\", "/")
        parts = path_str.split("/")
        class_folder = parts[-2]
        filename = parts[-1]
        url = f"https://huggingface.co/datasets/GPrabhanjana/oral-images/resolve/main/{class_folder}/{filename}"
        
        try:
            req = urllib.request.urlopen(url)
            img_bytes = req.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = strip_exif(img)
            
            # Compute hash
            img_hash = str(imagehash.phash(img))
            existing_hashes.add(img_hash)
            
            all_valid_images.append({
                "image": img,
                "label": str(old_labels[i]),
                "path": path_str,
                "source": "old"
            })
        except Exception as e:
            print(f"Failed to download/process old image {url}: {e}")
            
    print(f"Successfully loaded {len(all_valid_images)} old images. Unique hashes: {len(existing_hashes)}")
    
    print(f"Extracting new images from {ZIP_PATH}...")
    new_images = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # Infer label from folder structure
                    path_str = os.path.join(root, file).replace("\\", "/")
                    if "NON CANCER" in path_str.upper() or "NON_CANCER" in path_str.upper() or "BENIGN" in path_str.upper():
                        label = "benign"
                    elif "CANCER" in path_str.upper() or "MALIGNANT" in path_str.upper():
                        label = "malignant"
                    else:
                        continue # Unknown class
                    
                    try:
                        img = Image.open(path_str).convert("RGB")
                        img = strip_exif(img)
                        img_hash = str(imagehash.phash(img))
                        
                        # Deduplicate (threshold <= 8 usually implies visually similar, but exact match hash is identical. 
                        # We will use exact phash match for dedup for safety against removing distinct similar lesions)
                        # We use exact match of the phash string
                        if img_hash not in existing_hashes:
                            existing_hashes.add(img_hash)
                            new_images.append({
                                "image": img,
                                "label": label,
                                "path": f"new_data/{label}/{file}",
                                "source": "new"
                            })
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")

    print(f"Found {len(new_images)} new unique images after dedup.")
    all_valid_images.extend(new_images)
    
    # Class balancing (Undersample CANCER to max 2x NON CANCER)
    benign_list = [item for item in all_valid_images if item["label"] == "benign"]
    malignant_list = [item for item in all_valid_images if item["label"] == "malignant"]
    
    print(f"Before balancing: {len(malignant_list)} malignant, {len(benign_list)} benign.")
    
    max_malignant = len(benign_list) * 2
    if len(malignant_list) > max_malignant:
        print(f"Undersampling malignant from {len(malignant_list)} to {max_malignant}...")
        random.seed(42)
        malignant_list = random.sample(malignant_list, max_malignant)
        
    final_dataset = benign_list + malignant_list
    print(f"Final dataset size: {len(final_dataset)} images ({len(malignant_list)} malignant, {len(benign_list)} benign).")
    
    # Feature Extraction
    print("Loading MobileNetV2 for feature extraction...")
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

    final_embeddings = []
    final_paths = []
    final_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for idx, item in enumerate(final_dataset):
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(final_dataset)}...")
            tensor = transform(item["image"]).unsqueeze(0).to(device)
            emb = model(tensor).squeeze().cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            final_embeddings.append(emb)
            final_paths.append(item["path"])
            final_labels.append(item["label"])

    print(f"Saving merged index to {NEW_INDEX_PATH}...")
    np.savez(
        NEW_INDEX_PATH,
        embeddings=np.array(final_embeddings),
        paths=np.array(final_paths),
        labels=np.array(final_labels)
    )
    print("Index rebuilt successfully!")

if __name__ == "__main__":
    main()
