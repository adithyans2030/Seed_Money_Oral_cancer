import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

CSV_PATH = r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\oral_cancer_prediction_dataset.csv"
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
    # Use the processed CSV which is deduped!
    processed_csv_path = CSV_PATH.replace('.csv', '_processed.csv')
    path_to_use = processed_csv_path if os.path.exists(processed_csv_path) else CSV_PATH
    
    X, y, feature_names = load_and_preprocess_data(path_to_use)

    print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features.")
    
    counts = np.bincount(y)
    print(f"Target distribution: 0 (No Cancer): {counts[0]}, 1 (Cancer): {counts[1]}")
    
    # Calculate optimal scale_pos_weight (ratio of negative to positive instances)
    scale_pos_weight = counts[0] / counts[1]
    print(f"Setting scale_pos_weight to {scale_pos_weight:.4f} to penalize false positives...")

    # We will use an imblearn Pipeline to ensure SMOTE is only applied to training folds
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')), # Fix for NaNs introduced during map()
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight
        ))
    ])

    # Define hyperparameter grid for RandomizedSearchCV
    param_grid = {
        'xgb__n_estimators': [50, 100, 200, 300],
        'xgb__max_depth': [3, 4, 5, 6, 8],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__gamma': [0, 0.1, 0.5, 1, 5],
        'xgb__min_child_weight': [1, 3, 5, 7]
    }

    print("\nRunning RandomizedSearchCV to find optimal hyperparameters (this may take a minute)...")
    cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid,
        n_iter=20, 
        scoring='roc_auc', 
        cv=cv_search,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X, y)
    best_pipeline = random_search.best_estimator_
    print("\nBest Hyperparameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    print("\nRunning 5-fold Stratified Cross Validation with BEST model and SMOTE...")
    cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    sensitivities = []
    specificities = []
    
    # For calibration curve over all folds
    y_true_all = []
    y_prob_all = []

    for fold, (train_idx, test_idx) in enumerate(cv_eval.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit the best pipeline (includes SMOTE on train only)
        best_pipeline.fit(X_train, y_train)
        
        # Predict
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        auc_scores.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)

    print(f"\n--- Cross-Validation Results (5-Fold) ---")
    print(f"ROC AUC:     {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
    print(f"Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

    # Plot Calibration Curve
    print("\nGenerating Calibration Curve...")
    prob_true, prob_pred = calibration_curve(y_true_all, y_prob_all, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="XGBoost (Tuned + SMOTE)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    calib_path = os.path.join(os.path.dirname(MODEL_PATH), "calibration_curve.png")
    plt.savefig(calib_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to {calib_path}")

    print("\nTraining final model on full dataset...")
    # Train the pipeline on everything for production
    best_pipeline.fit(X, y)
    final_model = best_pipeline.named_steps['xgb']

    # Feature Importance (Global)
    importances = final_model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Feature Importances (Global):")
    for name, imp in ranked[:5]:
        print(f"  - {name}: {imp:.4f}")

    # Ensure target directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print(f"\nSaving tuned model to {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    main()
