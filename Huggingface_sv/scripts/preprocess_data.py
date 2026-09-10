import os
import glob
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from datetime import datetime

# Configurations
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

UNDERSAMPLE_RATIO = 2  # CANCER kept = NON_CANCER_count * UNDERSAMPLE_RATIO
MIN_RES = 100
TARGET_SIZE = (224, 224)
PHASH_THRESHOLD = 8

REPO_DIR = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project"
RAW_IMAGE_DIR = os.path.join(REPO_DIR, "data", "Oral Cancer", "Oral Cancer Dataset")
PROCESSED_IMAGE_DIR = os.path.join(REPO_DIR, "data", "Oral Cancer Dataset Processed")
CSV_PATH = os.path.join(REPO_DIR, "oral_cancer_prediction_dataset.csv")

def apply_clahe(img_np):
    # img_np is expected to be RGB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def process_images():
    report = []
    global_hashes = {} # {hash: filepath}
    duplicates_removed = 0
    resolution_rejected = 0
    
    if os.path.exists(PROCESSED_IMAGE_DIR):
        shutil.rmtree(PROCESSED_IMAGE_DIR)
    os.makedirs(os.path.join(PROCESSED_IMAGE_DIR, "CANCER"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_IMAGE_DIR, "NON CANCER"), exist_ok=True)
    
    processed_counts = {"CANCER": 0, "NON CANCER": 0}
    valid_paths = {"CANCER": [], "NON CANCER": []}
    
    for cls in ["NON CANCER", "CANCER"]: # Process NON CANCER first to keep them if dups exist
        folder = os.path.join(RAW_IMAGE_DIR, cls)
        files = glob.glob(os.path.join(folder, "*.*"))
        report.append(f"Found {len(files)} raw images in {cls}")
        
        for f in files:
            try:
                with Image.open(f) as img:
                    img = img.convert('RGB')
                    w, h = img.size
                    
                    if w < MIN_RES or h < MIN_RES:
                        resolution_rejected += 1
                        continue
                        
                    # Compute phash
                    hsh = imagehash.phash(img)
                    
                    # Check against global_hashes with threshold
                    is_dup = False
                    for existing_hash in global_hashes:
                        if hsh - existing_hash <= PHASH_THRESHOLD:
                            is_dup = True
                            break
                    
                    if is_dup:
                        duplicates_removed += 1
                        continue
                        
                    global_hashes[hsh] = f
                    valid_paths[cls].append(f)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
    report.append(f"Resolution rejected (<{MIN_RES}px): {resolution_rejected}")
    report.append(f"Perceptual duplicates removed (<= {PHASH_THRESHOLD}): {duplicates_removed}")
    report.append("Note: Cross-dataset dedup against HuggingFace images could not be performed because the raw HuggingFace images are not available on disk (they exist only as embeddings in cbir_index.npz).")
    
    # Undersampling
    nc_count = len(valid_paths["NON CANCER"])
    target_cancer = int(nc_count * UNDERSAMPLE_RATIO)
    c_count_before = len(valid_paths["CANCER"])
    
    if c_count_before > target_cancer:
        valid_paths["CANCER"] = random.sample(valid_paths["CANCER"], target_cancer)
        report.append(f"Undersampled CANCER from {c_count_before} down to {target_cancer} (Ratio {UNDERSAMPLE_RATIO}:1)")
    else:
        report.append(f"No undersampling needed. CANCER count: {c_count_before}")
        
    # Resize, CLAHE, and Save
    for cls in ["NON CANCER", "CANCER"]:
        for f in valid_paths[cls]:
            try:
                with Image.open(f) as img:
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img_np = np.array(img)
                    img_np = apply_clahe(img_np)
                    out_img = Image.fromarray(img_np)
                    out_path = os.path.join(PROCESSED_IMAGE_DIR, cls, os.path.basename(f))
                    out_img.save(out_path, quality=95)
                    processed_counts[cls] += 1
            except Exception as e:
                print(f"Error saving {f}: {e}")
                
    report.append(f"Final Processed Images: {processed_counts}")
    return report

def process_csv():
    report = []
    if not os.path.exists(CSV_PATH):
        report.append("CSV not found.")
        return report
        
    df = pd.read_csv(CSV_PATH)
    initial_rows = len(df)
    report.append(f"Original CSV Rows: {initial_rows}")
    
    # Near-duplicate check: two rows are duplicates if they share the exact
    # same risk-factor/demographic profile — the model's actual input features.
    # Deliberately EXCLUDES the diagnosis/outcome and any downstream-of-diagnosis
    # columns (Cancer Stage, Early Diagnosis, Survival Rate, Treatment Type, ...):
    # including those in the key lets two identical-outcome rows "not match" and
    # two different-outcome rows "match", which silently skews the class balance
    # (this previously turned a 50/50 dataset into a 77/23 one).
    model_feature_cols = [
        "Age", "Gender", "Tobacco Use", "Alcohol Consumption",
        "HPV Infection", "Betel Quid Use", "Chronic Sun Exposure",
        "Poor Oral Hygiene", "Diet (Fruits & Vegetables Intake)", "Family History of Cancer",
        "Compromised Immune System", "Oral Lesions", "Unexplained Bleeding",
        "Difficulty Swallowing", "White or Red Patches in Mouth"
    ]
    key_cols = [c for c in model_feature_cols if c in df.columns]

    if key_cols:
        df_dedup = df.drop_duplicates(subset=key_cols, keep='first')
        dups_removed = initial_rows - len(df_dedup)
        report.append(f"Near-duplicates removed (matching on {key_cols}): {dups_removed}")
    else:
        df_dedup = df.drop_duplicates()
        dups_removed = initial_rows - len(df_dedup)
        report.append(f"Exact duplicates removed: {dups_removed}")
        
    df_dedup.to_csv(CSV_PATH.replace('.csv', '_processed.csv'), index=False)
    report.append(f"Final CSV Rows: {len(df_dedup)}")
    
    # Class balance
    if 'Diagnosis' in df_dedup.columns:
        counts = df_dedup['Diagnosis'].value_counts().to_dict()
        report.append(f"Final CSV Diagnosis Balance: {counts}")
        
    return report

if __name__ == "__main__":
    print("Starting Preprocessing Pipeline...")
    img_report = process_images()
    csv_report = process_csv()
    
    full_report = ["# Data Provenance Report", f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    full_report.extend(["## Image Dataset Processing", ""])
    full_report.extend([f"- {r}" for r in img_report])
    full_report.extend(["", "## Tabular Dataset Processing", ""])
    full_report.extend([f"- {r}" for r in csv_report])
    full_report.extend(["", "## IMPORTANT NEXT STEPS", "> **Note:** `create_splits.py` MUST be run immediately after this script completes to generate the new training/validation/test splits from the newly processed data directory."])
    
    report_path = os.path.join(REPO_DIR, "data_provenance_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(full_report))
    print(f"Done. Report saved to {report_path}")
