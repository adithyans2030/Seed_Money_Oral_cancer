import os
import glob
import json
import argparse
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    parser = argparse.ArgumentParser(description="Create locked stratified splits for OralGuard validation.")
    parser.add_argument("--img_dir", type=str, default="../data/images", help="Path to the directory containing all merged images (structured as CANCER and NON_CANCER subfolders)")
    parser.add_argument("--out_dir", type=str, default="../data", help="Output directory for the JSON manifest files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    holdout_path = os.path.join(args.out_dir, "holdout_pool.json")
    if os.path.exists(holdout_path):
        print("WARNING: holdout_pool.json already exists! Refusing to overwrite to prevent data leakage.")
        return

    # Find all images
    image_paths = []
    labels = []
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                path = os.path.join(root, file)
                # Infer label from parent folder name
                parent_folder = os.path.basename(root).lower()
                if "cancer" in parent_folder and "non" not in parent_folder:
                    label = "malignant"
                else:
                    label = "benign"
                image_paths.append(path)
                labels.append(label)
                
    if not image_paths:
        print(f"No images found in {args.img_dir}. Creating DUMMY splits for testing purposes.")
        # Create dummy data for development/validation of the pipeline
        for i in range(1300):
            label = "malignant" if i < 975 else "benign" # 3:1 class imbalance
            image_paths.append(f"dummy_path/img_{i}.jpg")
            labels.append(label)

    print(f"Total images loaded: {len(image_paths)}")
    print(f"Global distribution: {Counter(labels)}")

    # We need: 70% index, 15% validation, 15% holdout
    # First split: 70% index, 30% temp (validation + holdout)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    for index_idx, temp_idx in sss1.split(image_paths, labels):
        index_paths = [image_paths[i] for i in index_idx]
        index_labels = [labels[i] for i in index_idx]
        
        temp_paths = [image_paths[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]

    # Second split: split the 30% temp into 50% validation and 50% holdout (i.e. 15% of total each)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    for val_idx, holdout_idx in sss2.split(temp_paths, temp_labels):
        val_paths = [temp_paths[i] for i in val_idx]
        val_labels = [temp_labels[i] for i in val_idx]
        
        holdout_paths = [temp_paths[i] for i in holdout_idx]
        holdout_labels = [temp_labels[i] for i in holdout_idx]

    # Save to JSON
    def save_json(paths, lbls, filename):
        data = [{"filename": p, "label": l} for p, l in zip(paths, lbls)]
        filepath = os.path.join(args.out_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        counts = Counter(lbls)
        total = len(lbls)
        print(f"Saved {filename} - Total: {total}")
        for k, v in counts.items():
            print(f"  {k}: {v} ({(v/total)*100:.1f}%)")

    print("\n--- Split Distributions ---")
    save_json(index_paths, index_labels, "index_pool.json")
    save_json(val_paths, val_labels, "validation_pool.json")
    save_json(holdout_paths, holdout_labels, "holdout_pool.json")

    # Add README warning
    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Data Splits\n\n")
        f.write("WARNING: DO NOT retrain or tune thresholds using `holdout_pool.json`. This pool is sacred and must only be used for final reporting.\n")
    
    print("\nLocked splits created successfully.")

if __name__ == "__main__":
    main()
