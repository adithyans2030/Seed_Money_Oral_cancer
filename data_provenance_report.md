# Data Provenance Report
Generated on: 2026-07-30 12:01:14

## Image Dataset Processing

- Found 250 raw images in NON CANCER
- Found 500 raw images in CANCER
- Resolution rejected (<100px): 7
- Perceptual duplicates removed (<= 8): 49
- Note: Cross-dataset dedup against HuggingFace images could not be performed because the raw HuggingFace images are not available on disk (they exist only as embeddings in cbir_index.npz).
- Undersampled CANCER from 467 down to 454 (Ratio 2:1)
- Final Processed Images: {'CANCER': 454, 'NON CANCER': 227}

## Tabular Dataset Processing

- Original CSV Rows: 84922
- Near-duplicates removed (matching on ['Age', 'Gender', 'Tobacco Use', 'Alcohol Consumption', 'Cancer Stage', 'Early Diagnosis', 'Oral Cancer (Diagnosis)']): 80632
- Final CSV Rows: 4290

## IMPORTANT NEXT STEPS
> **Note:** `create_splits.py` MUST be run immediately after this script completes to generate the new training/validation/test splits from the newly processed data directory.