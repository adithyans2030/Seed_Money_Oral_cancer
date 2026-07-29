import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

CSV_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\oral_cancer_prediction_dataset.csv"
MODEL_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\Huggingface_sv\risk_model.pkl"

def load_and_preprocess_data(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    features = [
        "Age", "Gender", "Tobacco Use", "Alcohol Consumption",
        "HPV Infection", "Betel Quid Use", "Chronic Sun Exposure",
        "Poor Oral Hygiene", "Diet (Fruits & Vegetables Intake)", "Family History of Cancer",
        "Compromised Immune System", "Oral Lesions", "Unexplained Bleeding",
        "Difficulty Swallowing", "White or Red Patches in Mouth"
    ]
    target = "Oral Cancer (Diagnosis)"

    # Keep only needed columns and drop NA
    df = df[features + [target]].dropna()

    print("Encoding features...")
    # Map binary columns
    yes_no_cols = [
        "Tobacco Use", "Alcohol Consumption", "HPV Infection", "Betel Quid Use",
        "Chronic Sun Exposure", "Poor Oral Hygiene", "Family History of Cancer",
        "Compromised Immune System", "Oral Lesions", "Unexplained Bleeding",
        "Difficulty Swallowing", "White or Red Patches in Mouth", target
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Map Gender (Male=1, Female=0)
    df["Gender"] = df["Gender"].apply(lambda x: 1 if str(x).strip().lower() == "male" else 0)

    # Map Diet (Low=0, Medium=1, High=2)
    diet_map = {"Low": 0, "Medium": 1, "High": 2}
    df["Diet (Fruits & Vegetables Intake)"] = df["Diet (Fruits & Vegetables Intake)"].map(diet_map)

    # Ensure everything is numeric
    df = df.apply(pd.to_numeric)

    X = df[features].values
    y = df[target].values

    return X, y, features

def main():
    X, y, feature_names = load_and_preprocess_data(CSV_PATH)

    print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"Target distribution: {np.bincount(y)}")

    # Initialize XGBoost Classifier
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    print("Running 5-fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"CV ROC AUC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    print("Training final model on full dataset...")
    model.fit(X, y)

    # Feature Importance
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Feature Importances:")
    for name, imp in ranked[:5]:
        print(f"  - {name}: {imp:.4f}")

    # Ensure target directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
