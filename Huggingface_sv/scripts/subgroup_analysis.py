import os
import math
import numpy as np
import scipy.stats as stats
from supabase import create_client

def wilson_ci(p, n, z=1.96):
    if n == 0: return 0.0, 0.0
    den = 1 + z**2/n
    cap = p + z**2 / (2*n)
    std = z * math.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    return max(0.0, (cap - std)/den), min(1.0, (cap + std)/den)

def compute_metrics(tp, fp, tn, fn):
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens_l, sens_u = wilson_ci(sens, tp + fn)
    spec_l, spec_u = wilson_ci(spec, tn + fp)
    return {
        "sensitivity": sens,
        "sensitivity_ci": (sens_l, sens_u),
        "specificity": spec,
        "specificity_ci": (spec_l, spec_u),
        "total": tp + fp + tn + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def fetch_data():
    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_KEY")
    
    if not supa_url or not supa_key:
        print("Warning: SUPABASE_URL or SUPABASE_KEY not set. Using dummy data for subgroup analysis.")
        return generate_dummy_data()
        
    try:
        client = create_client(supa_url, supa_key)
        
        # We need sessions and clinician_labels
        sess_res = client.table("sessions").select("*").execute()
        labels_res = client.table("clinician_labels").select("*").execute()
        
        sessions = {s['id']: s for s in sess_res.data}
        joined_data = []
        for lbl in labels_res.data:
            sid = lbl['session_id']
            if sid in sessions:
                joined_data.append({
                    "session": sessions[sid],
                    "label": lbl
                })
        return joined_data
    except Exception as e:
        print(f"Failed to fetch from Supabase: {e}. Using dummy data.")
        return generate_dummy_data()

def generate_dummy_data():
    import random
    data = []
    genders = ['Male', 'Female', 'Other']
    ages = [25, 35, 45, 55, 65, 75]
    for i in range(200):
        actual = random.choice(['malignant', 'benign'])
        # Add some bias to age < 40 for testing degradation
        age = random.choice(ages)
        pred = actual
        if age < 40 and random.random() < 0.3:
            pred = 'benign' if actual == 'malignant' else 'malignant'
        elif random.random() < 0.1:
            pred = 'benign' if actual == 'malignant' else 'malignant'
            
        data.append({
            "session": {
                "age": age,
                "gender": random.choice(genders),
                "cbir_predicted_label": pred
            },
            "label": {
                "actual_diagnosis": actual
            }
        })
    return data

def main():
    data = fetch_data()
    
    if not data:
        print("No data available for subgroup analysis.")
        return
        
    subgroups = {
        "Gender": {},
        "Age Band": {}
    }
    
    # Initialize groups
    baseline = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "inconclusive": 0}
    
    for row in data:
        sess = row["session"]
        lbl = row["label"]
        
        true_dx = lbl.get("actual_diagnosis", "").lower()
        if true_dx not in ['malignant', 'benign']:
            continue
            
        cbir_pred = sess.get("cbir_predicted_label", "").lower()
        pred = "inconclusive"
        if "malignant" in cbir_pred: pred = "malignant"
        elif "benign" in cbir_pred: pred = "benign"
        
        age = sess.get("age")
        if age is not None:
            band = "<40" if age < 40 else (">60" if age > 60 else "40-60")
        else:
            band = "Unknown"
            
        gender = sess.get("gender", "Unknown")
        if gender: gender = gender.capitalize()
        
        def update_counts(counts_dict):
            if pred == "inconclusive":
                counts_dict["inconclusive"] += 1
            else:
                if true_dx == 'malignant' and pred == 'malignant': counts_dict["tp"] += 1
                elif true_dx == 'benign' and pred == 'malignant': counts_dict["fp"] += 1
                elif true_dx == 'benign' and pred == 'benign': counts_dict["tn"] += 1
                elif true_dx == 'malignant' and pred == 'benign': counts_dict["fn"] += 1
        
        update_counts(baseline)
        
        if gender not in subgroups["Gender"]:
            subgroups["Gender"][gender] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "inconclusive": 0}
        update_counts(subgroups["Gender"][gender])
        
        if band not in subgroups["Age Band"]:
            subgroups["Age Band"][band] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "inconclusive": 0}
        update_counts(subgroups["Age Band"][band])

    # Analyze
    report = ["# Subgroup Analysis Report", ""]
    
    def format_group(name, counts):
        m = compute_metrics(counts["tp"], counts["fp"], counts["tn"], counts["fn"])
        inc_rate = counts["inconclusive"] / (m["total"] + counts["inconclusive"]) if (m["total"] + counts["inconclusive"]) > 0 else 0
        return (
            f"**{name}** (N={m['total'] + counts['inconclusive']})\n"
            f"- Sensitivity: {m['sensitivity']:.3f} (95% CI: {m['sensitivity_ci'][0]:.3f}-{m['sensitivity_ci'][1]:.3f})\n"
            f"- Specificity: {m['specificity']:.3f} (95% CI: {m['specificity_ci'][0]:.3f}-{m['specificity_ci'][1]:.3f})\n"
            f"- Inconclusive Rate: {inc_rate:.3f}\n"
        )
        
    report.append("## Baseline Metrics")
    report.append(format_group("All Patients", baseline))
    
    for category, groups in subgroups.items():
        report.append(f"## By {category}")
        for g_name, counts in groups.items():
            report.append(format_group(g_name, counts))
            
            # Chi-square vs rest of population for Sensitivity
            tp = counts["tp"]
            fn = counts["fn"]
            rest_tp = baseline["tp"] - tp
            rest_fn = baseline["fn"] - fn
            
            if (tp + fn) > 0 and (rest_tp + rest_fn) > 0:
                table = [[tp, fn], [rest_tp, rest_fn]]
                try:
                    chi2, p_val, dof, ex = stats.chi2_contingency(table, correction=True)
                    sig = "SIGNIFICANT DEGRADATION!" if p_val < 0.05 and (tp/(tp+fn) < rest_tp/(rest_tp+rest_fn)) else ""
                    report.append(f"  - Sensitivity p-value vs rest: {p_val:.4f} {sig}")
                except:
                    pass
            report.append("")
            
    os.makedirs("../results", exist_ok=True)
    with open("../results/subgroup_report.md", "w") as f:
        f.write("\n".join(report))
        
    print("Subgroup analysis saved to results/subgroup_report.md")

if __name__ == "__main__":
    main()
