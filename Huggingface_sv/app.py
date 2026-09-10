"""
OralGuard FastAPI Backend
=========================
Endpoints:
  GET  /                  — health check
  POST /search            — CBIR image search + decision rule + Supabase logging  (P3)
  POST /predict-risk      — Evidence-based questionnaire risk scoring + Supabase logging  (P4)
  POST /combined-risk     — Fusion of both tools into a single triage output       (P5)
"""

import hashlib
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import math
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from scripts.preprocess import preprocess_image
from scripts.gradcam_utils import get_guided_gradcam, generate_gradcam_base64

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oralguard")

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
load_dotenv(override=True)  # loads .env if present locally; env vars take precedence on HF Spaces

INDEX_PATH     = "cbir_index_finetuned.npz"
IMG_SIZE       = 224
TOP_K          = 6        # matches to return per class
INCONCL_THRESH = 0.70    # Tuned threshold for MobileNetV2 cosine similarity mean of top k

# Local image mount + public URL base — override via env vars for deployment
# (defaults match this developer's local machine and are not valid elsewhere).
IMAGES_DIR      = os.environ.get(
    "IMAGES_DIR",
    r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral-Cancer-Detection---Seed-Money-Project\data\Oral Cancer\Oral Cancer Dataset",
)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

# Validation Versions
INDEX_VERSION = "index_v5_metric_finetuned_823img"
DECISION_RULE_VERSION = "rule_v3_metric_k5_t0.70"  # must track INCONCL_THRESH above

# ──────────────────────────────────────────────────────────
# SUPABASE CLIENT (Priority 1)
# ──────────────────────────────────────────────────────────
supabase_client = None
try:
    from supabase import create_client, Client as SupabaseClient

    _supa_url = os.environ.get("SUPABASE_URL", "")
    _supa_key = os.environ.get("SUPABASE_KEY", "")

    if _supa_url and _supa_key:
        supabase_client: Optional[SupabaseClient] = create_client(_supa_url, _supa_key)
        logger.info("Supabase client initialised successfully.")
    else:
        logger.warning(
            "SUPABASE_URL or SUPABASE_KEY not set. "
            "DB logging disabled — set these as HuggingFace Spaces secrets."
        )
except Exception as e:
    logger.error(f"Failed to initialise Supabase client: {e}. DB logging disabled.")


def log_to_supabase(table: str, payload: Dict[str, Any]) -> Optional[str]:
    """
    Write a row to a Supabase table.
    Returns the inserted row's `id` on success, or None on failure.
    NEVER raises — DB failures must not crash the API response.
    """
    if supabase_client is None:
        return None
    try:
        result = supabase_client.table(table).insert(payload).execute()
        if result.data:
            return result.data[0].get("id")
    except Exception as e:
        logger.error(f"Supabase insert to '{table}' failed: {e}")
    return None


def update_supabase(table: str, row_id: str, payload: Dict[str, Any]) -> bool:
    """Update or insert an existing row (upsert). Returns True on success. Never raises."""
    if supabase_client is None:
        return False
    try:
        # Include id in payload for upsert
        upsert_payload = {"id": row_id, **payload}
        supabase_client.table(table).upsert(upsert_payload).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase upsert on '{table}' id={row_id} failed: {e}")
    return False


# ──────────────────────────────────────────────────────────
# LOAD CBIR INDEX (runs once at startup)
# ──────────────────────────────────────────────────────────
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"{INDEX_PATH} not found in container.")

data = np.load(INDEX_PATH, allow_pickle=True)
index = {
    "embeddings": data["embeddings"],
    "paths":      data["paths"],
    "labels":     data["labels"],
}
logger.info(f"Loaded {len(index['paths'])} indexed images from {INDEX_PATH}.")

# ──────────────────────────────────────────────────────────
# LOAD MOBILENETV2 FEATURE EXTRACTOR (runs once at startup)
# ──────────────────────────────────────────────────────────
# Setup CBIR Model (MobileNetV2, L2-normalised features) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

finetuned_path = "models/finetuned_mobilenetv2.pth"
if not os.path.exists(finetuned_path):
    finetuned_path = "../models/finetuned_mobilenetv2.pth"  # local monorepo dev layout
if os.path.exists(finetuned_path):
    # Reconstruct the classification head temporarily to match the fine-tuned state_dict
    num_ftrs = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2)
    )
    backbone.load_state_dict(torch.load(finetuned_path, map_location=device))
    logger.info(f"Loaded fine-tuned model from {finetuned_path}")
else:
    logger.info("Fine-tuned model not found, falling back to raw ImageNet features.")

backbone.classifier = nn.Identity()  # strip classification head
model = backbone.to(device)
model.eval()

logger.info("MobileNetV2 feature extractor loaded.")

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract L2-normalised MobileNetV2 feature vector."""
    # TODO: Critical Inference Requirement
    # Ensure preprocess_image() applies CLAHE in the EXACT same order as the training pipeline:
    # Order MUST be: Resize (224x224) -> CLAHE normalization. NOT CLAHE -> resize.
    # Otherwise, cosine similarity scores will be systematically wrong.
    tensor = preprocess_image(image)
    if tensor is None:
        raise ValueError("Image preprocessing failed or resolution too low.")
    tensor = tensor.unsqueeze(0).to(device)
    emb = model(tensor).squeeze().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def hash_image_bytes(raw: bytes) -> str:
    """SHA-256 hash of raw image bytes — used as a privacy-safe identifier."""
    return hashlib.sha256(raw).hexdigest()


def strip_exif(image: Image.Image) -> Image.Image:
    """Return a copy of the image with EXIF metadata stripped."""
    data_copy = image.tobytes()
    clean = Image.frombytes(image.mode, image.size, data_copy)
    return clean


def compute_decision(
    benign_results: List[Dict],
    malignant_results: List[Dict],
) -> Dict[str, Any]:
    """
    Derive a binary decision + confidence from top-k (k=5) matches across both classes.

    Returns:
        {
            label: "Benign pattern" | "Malignant pattern" | "Inconclusive",
            action: "monitor" | "urgent_refer" | "refer",
            confidence: float (0–1, mean similarity of winning class),
            malignant_score: float,
            benign_score: float,
        }
    """
    # Combine all results and sort by similarity descending
    all_results = benign_results + malignant_results
    all_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
    
    # Take top k
    k = 5
    top_k = all_results[:k]

    mal_matches = [r for r in top_k if r.get("label", "").lower() == "malignant"]
    ben_matches = [r for r in top_k if r.get("label", "").lower() == "benign"]

    mal_count = len(mal_matches)
    ben_count = len(ben_matches)
    
    # Calculate scores for payload consistency
    mal_score = float(np.mean([r["similarity"] for r in mal_matches])) if mal_matches else 0.0
    ben_score = float(np.mean([r["similarity"] for r in ben_matches])) if ben_matches else 0.0

    # Majority vote
    if mal_count > ben_count:
        winning_class = "Malignant pattern"
        confidence = mal_score
        action = "urgent_refer"
    else:
        winning_class = "Benign pattern"
        confidence = ben_score
        action = "monitor"

    # Inconclusive check based on tuned threshold
    if confidence < INCONCL_THRESH:
        label = "Inconclusive"
        action = "refer"
    else:
        label = winning_class

    return {
        "label":           label,
        "action":          action,
        "confidence":      round(confidence, 4),
        "malignant_score": round(mal_score, 4),
        "benign_score":    round(ben_score, 4),
    }


# ──────────────────────────────────────────────────────────
# RISK SCORE → LABEL
# ──────────────────────────────────────────────────────────
def score_to_label(score: float) -> str:
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Moderate"
    return "Low"


# ──────────────────────────────────────────────────────────
# RULE-BASED QUESTIONNAIRE RISK SCORING
#
# Replaces an XGBoost model previously trained on a public Kaggle CSV whose
# risk-factor columns were statistically independent of its diagnosis label
# (verified: correlation ~0 and p>0.1 for every one of the 15 features across
# 85k rows) — i.e. the "model" had no real signal to learn. This scores
# answers directly against published oral-cancer epidemiology instead.
#
# Two tiers, weighted very differently on purpose:
#   - CAUSE factors are background/lifestyle exposures — they raise long-run
#     risk but are not evidence of disease, so alone they cap out around
#     "Moderate" even at their worst combination.
#   - SYMPTOM factors describe a current physical finding (a non-healing
#     sore, a patch that won't wipe off, etc.) — these resemble oral
#     cancer's actual clinical presentation, so a single one alone can
#     already reach "Moderate", and two together reach "High".
#
# Weights are directional, sourced from general oncology literature (e.g.
# tobacco/betel quid/tobacco+alcohol synergy carrying the largest published
# relative-risk increases). They are a starting point, not a clinical
# guarantee — a clinician should review the exact point values before this
# is treated as final.
# ──────────────────────────────────────────────────────────

CAUSE_WEIGHTS: Dict[str, int] = {
    "tobacco_use":            16,
    "betel_quid_use":         15,
    "alcohol_consumption":    11,
    "hpv_infection":          10,
    "compromised_immune":      8,
    "family_history":          5,
    "chronic_sun_exposure":    3,
    "poor_oral_hygiene":       2,
}
DIET_WEIGHTS = {0: 3, 1: 1, 2: 0}   # diet: 0=Low, 1=Medium, 2=High fruit/veg intake
SYMPTOM_WEIGHTS: Dict[str, int] = {
    # Each weight exceeds RISK_SCALE * 0.35 alone, so any single reported
    # symptom reliably reaches at least "Moderate" regardless of age/diet —
    # a screening tool should never let one red-flag symptom alone read as "Low".
    "oral_lesions":           42,   # non-healing sore/ulcer/lump — classic presenting sign
    "white_red_patches":      39,   # leukoplakia/erythroplakia are themselves potentially malignant disorders
    "unexplained_bleeding":   37,   # "unexplained" already excludes the common benign (brushing/flossing) cause
    "difficulty_swallowing":  36,
}
MIN_NOTABLE_CONTRIBUTION = 5   # points below this are too marginal to name as a "top contributing factor"
TOBACCO_ALCOHOL_SYNERGY = 8   # well-documented multiplicative (not additive) combined effect
TOBACCO_BETEL_SYNERGY   = 6   # common combined smoking + chewing use pattern
RISK_SCALE = 100.0            # denominator the accumulated points are normalised against

DISPLAY_NAMES = {
    "age": "age", "gender": "gender", "tobacco_use": "tobacco use",
    "alcohol_consumption": "alcohol consumption", "hpv_infection": "HPV infection",
    "betel_quid_use": "betel quid use", "chronic_sun_exposure": "chronic sun exposure",
    "poor_oral_hygiene": "poor oral hygiene", "diet": "diet",
    "family_history": "family history of cancer", "compromised_immune": "compromised immune system",
    "oral_lesions": "a non-healing oral sore/lesion", "unexplained_bleeding": "unexplained bleeding",
    "difficulty_swallowing": "difficulty swallowing", "white_red_patches": "white or red patches",
}


def _age_points(age: int) -> int:
    if age >= 60:
        return 8
    if age >= 45:
        return 5
    if age >= 30:
        return 2
    return 0


def compute_rule_based_risk(ans: "RiskAnswers") -> Dict[str, Any]:
    """
    Additive, literature-weighted risk score. Returns the exact same shape
    the old XGBoost/SHAP path returned: risk_score, risk_label, top_features,
    explanation_text — so no downstream consumer needs to change.
    """
    contributions: List[tuple] = []  # (feature_name, points)

    binary_causes = {
        "tobacco_use":         ans.tobacco_use,
        "betel_quid_use":      ans.betel_quid_use,
        "alcohol_consumption": ans.alcohol_consumption,
        "hpv_infection":       ans.hpv_infection,
        "compromised_immune":  ans.compromised_immune,
        "family_history":      ans.family_history,
        "chronic_sun_exposure": ans.chronic_sun_exposure,
        "poor_oral_hygiene":   ans.poor_oral_hygiene,
    }
    for name, present in binary_causes.items():
        if present:
            contributions.append((name, CAUSE_WEIGHTS[name]))

    diet_pts = DIET_WEIGHTS.get(ans.diet, 1)
    if diet_pts > 0:
        contributions.append(("diet", diet_pts))

    symptoms = {
        "oral_lesions":          ans.oral_lesions,
        "white_red_patches":     ans.white_red_patches,
        "unexplained_bleeding":  ans.unexplained_bleeding,
        "difficulty_swallowing": ans.difficulty_swallowing,
    }
    for name, present in symptoms.items():
        if present:
            contributions.append((name, SYMPTOM_WEIGHTS[name]))

    if ans.tobacco_use and ans.alcohol_consumption:
        contributions.append(("tobacco_alcohol_combined", TOBACCO_ALCOHOL_SYNERGY))
    if ans.tobacco_use and ans.betel_quid_use:
        contributions.append(("tobacco_betel_combined", TOBACCO_BETEL_SYNERGY))

    age_pts = _age_points(ans.age)
    if age_pts > 0:
        contributions.append(("age", age_pts))

    total_points = sum(pts for _, pts in contributions)
    risk_score = min(total_points / RISK_SCALE, 1.0)
    risk_label = score_to_label(risk_score)

    # Rank contributors by actual point weight — exact, not approximate.
    # Symptoms naturally surface first since they're weighted higher.
    ranked = sorted(contributions, key=lambda x: x[1], reverse=True)
    excluded = ("age", "gender", "tobacco_alcohol_combined", "tobacco_betel_combined")
    named = [n for n, pts in ranked if n not in excluded and pts >= MIN_NOTABLE_CONTRIBUTION]
    top_features = named[:2]

    if len(top_features) == 2:
        explanation_text = (
            f"Your {DISPLAY_NAMES.get(top_features[0], top_features[0])} and "
            f"{DISPLAY_NAMES.get(top_features[1], top_features[1])} are the top contributors to your risk score."
        )
    elif len(top_features) == 1:
        explanation_text = (
            f"Your {DISPLAY_NAMES.get(top_features[0], top_features[0])} is the primary contributor to your risk score."
        )
    else:
        explanation_text = "No significant risk factors or symptoms were identified in your responses."

    return {
        "risk_score":       risk_score,
        "risk_label":       risk_label,
        "top_features":     top_features,
        "explanation_text": explanation_text,
    }


# ──────────────────────────────────────────────────────────
# COMBINED RISK FUSION TABLE
# (questionnaire label) × (CBIR label) → (urgency, recommendation)
# ──────────────────────────────────────────────────────────
_FUSION: Dict[tuple, Dict[str, str]] = {
    ("Low",      "Benign pattern"):    ("green", "Low — Monitor",
        "Your risk screener and image analysis are both reassuring. Continue routine dental check-ups and monthly self-examinations."),
    ("Low",      "Inconclusive"):      ("amber", "Low — Routine Referral",
        "Your risk profile is low, but the image analysis was inconclusive. We recommend a routine dental appointment to clarify."),
    ("Low",      "Malignant pattern"): ("amber", "Moderate — Clinical Review",
        "Despite a low questionnaire risk, the image shows patterns associated with malignancy. A dental or oral medicine review is advised."),
    ("Moderate", "Benign pattern"):    ("amber", "Moderate — Routine Referral",
        "You have moderate risk factors. The image looks benign, but a routine oral cancer screening with your dentist is recommended."),
    ("Moderate", "Inconclusive"):      ("amber", "Moderate — Clinical Review",
        "A moderate risk profile combined with an inconclusive image warrants a clinical examination. Book an appointment soon."),
    ("Moderate", "Malignant pattern"): ("red",   "High — Urgent Referral",
        "Your questionnaire risk profile and image findings are both concerning. Please seek evaluation from an oral specialist within 1–2 weeks."),
    ("High",     "Benign pattern"):    ("amber", "High — Clinical Review",
        "Your symptom and risk factor profile is high. Even though the image appears benign, you should be seen by a clinician to rule out malignancy."),
    ("High",     "Inconclusive"):      ("red",   "High — Urgent Referral",
        "High questionnaire risk combined with an inconclusive image requires prompt clinical evaluation. Please book an urgent appointment."),
    ("High",     "Malignant pattern"): ("red",   "High — Emergency Referral",
        "Both your questionnaire and image analysis indicate very high concern. Please seek urgent evaluation from an oral oncologist as soon as possible."),
}

def fuse_risk(q_label: str, cbir_label: str) -> Dict[str, str]:
    # Fallbacks for when only one tool is used
    if q_label and not cbir_label:
        urgency = "red" if q_label == "High" else ("amber" if q_label == "Moderate" else "green")
        combined_label = f"{q_label} Risk (Questionnaire Only)"
        recommendation = "Please complete the image matcher for a more comprehensive triage. Based purely on your questionnaire, consult a clinician if symptoms persist."
        return {"urgency": urgency, "combined_risk_label": combined_label, "recommendation": recommendation}
        
    if cbir_label and not q_label:
        urgency = "red" if cbir_label == "Malignant pattern" else ("amber" if cbir_label == "Inconclusive" else "green")
        combined_label = f"{cbir_label} (Image Analysis Only)"
        recommendation = "Please complete the risk screener for a more comprehensive triage. Based purely on the image, consult a clinician if you are concerned."
        return {"urgency": urgency, "combined_risk_label": combined_label, "recommendation": recommendation}

    key = (q_label, cbir_label)
    if key in _FUSION:
        urgency, combined_label, recommendation = _FUSION[key]
    else:
        urgency         = "amber"
        combined_label  = "Indeterminate — Review Recommended"
        recommendation  = (
            "We could not determine a precise triage level from the available data. "
            "Please consult a healthcare professional for a full evaluation."
        )
    return {
        "urgency":            urgency,
        "combined_risk_label": combined_label,
        "recommendation":     recommendation,
    }


# ──────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────────────

class RiskAnswers(BaseModel):
    """
    Questionnaire answers mapped from frontend IDs to CSV feature names.
    All binary features: 0 = No, 1 = Yes.
    age: integer
    gender: 1 = Male, 0 = Female/Other
    diet: 0 = Low, 1 = Medium, 2 = High (fruits & vegetables intake)
    """
    model_config = ConfigDict(populate_by_name=True)

    age:                    int
    gender:                 int           # 1=Male, 0=Female/Other
    tobacco_use:            int = Field(alias="q_tobacco")
    alcohol_consumption:    int = Field(alias="q_alcohol")
    hpv_infection:          int = Field(alias="q_hpv")
    betel_quid_use:         int = Field(alias="q_betel")
    chronic_sun_exposure:   int = Field(alias="q_sun")
    poor_oral_hygiene:      int = Field(alias="q_hygiene")
    diet:                   int = Field(alias="q_diet")
    family_history:         int = Field(alias="q_family")
    compromised_immune:     int = Field(alias="q_immune")
    oral_lesions:           int = Field(alias="q_lesions")
    unexplained_bleeding:   int = Field(alias="q_bleeding")
    difficulty_swallowing:  int = Field(alias="q_swallowing")
    white_red_patches:      int = Field(alias="q_patches")

    @field_validator(
        'tobacco_use', 'alcohol_consumption', 'hpv_infection', 'betel_quid_use',
        'chronic_sun_exposure', 'poor_oral_hygiene', 'diet', 'family_history',
        'compromised_immune', 'oral_lesions', 'unexplained_bleeding',
        'difficulty_swallowing', 'white_red_patches', mode='before'
    )
    @classmethod
    def coerce_int(cls, v):
        if isinstance(v, str):
            if v.lower() in ('yes', 'true', '1'): return 1
            if v.lower() in ('no', 'false', '0'): return 0
            try:
                return int(v)
            except ValueError:
                return 0
        if isinstance(v, bool):
            return 1 if v else 0
        return int(v)

    # Optional metadata for session logging
    region:                 Optional[str] = None
    occupation:             Optional[Literal['Manual Labour', 'Office', 'Other']] = None
    questionnaire_raw:      Optional[Dict[str, Any]] = None  # raw {c1:"yes", ...} for logging


class PredictRiskRequest(BaseModel):
    answers: RiskAnswers
    session_id: Optional[str] = None


class CombinedRiskRequest(BaseModel):
    session_id: str

class ClinicianLabelRequest(BaseModel):
    session_id: str
    clinician_id: str
    actual_diagnosis: str
    notes: Optional[str] = None


# ──────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="OralGuard API",
    description="Oral cancer CBIR search, questionnaire risk scoring, and combined triage.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to Netlify domain after validation
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
else:
    logger.warning(
        f"IMAGES_DIR '{IMAGES_DIR}' not found — /images static mount disabled. "
        "Set the IMAGES_DIR env var to enable local image serving."
    )


# ──────────────────────────────────────────────────────────
# ENDPOINT: GET /
# ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":    "OralGuard API is running.",
        "version":    "2.0.0",
        "index_size": len(index["paths"]),
        "risk_scoring": "rule-based",
        "supabase":   "connected" if supabase_client is not None else "not configured",
    }


# ──────────────────────────────────────────────────────────
# ENDPOINT: POST /search  (Priority 3)
# ──────────────────────────────────────────────────────────
@app.post("/search")
async def search(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Accept an oral lesion image, run CBIR, apply decision rule, log to Supabase.

    Returns:
      - results.benign / results.malignant  — top-6 matches per class
      - decision                            — label, action, confidence scores
      - session_id                          — UUID for cross-tool linking
    """
    try:
        raw_bytes = await file.read()

        # ── Input validation ──────────────────────────────
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail="Uploaded file is not a recognised image type.",
            )
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=422, detail="Empty file received.")

        # ── Privacy: hash before processing ──────────────
        image_hash = hash_image_bytes(raw_bytes)

        # ── Decode + strip EXIF ───────────────────────────
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            # Phone cameras commonly store an EXIF orientation tag instead of
            # rotating pixels — apply it before discarding EXIF, or sideways/
            # upside-down photos get matched in the wrong orientation.
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = strip_exif(img)
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Could not decode image. Please upload a valid JPG, PNG, or WEBP.",
            )

        # ── Feature extraction ────────────────────────────
        try:
            query_emb = extract_embedding(img)
        except ValueError as ve:
            raise HTTPException(status_code=422, detail=str(ve))

        sims = cosine_similarity(
            query_emb.reshape(1, -1),
            index["embeddings"],
        )[0]

        # ── Split results by class ────────────────────────
        benign_results    = []
        malignant_results = []

        for idx_i, sim in enumerate(sims):
            original_path = str(index["paths"][idx_i])
            parts         = original_path.replace("\\", "/").split("/")
            class_folder  = parts[-2]
            filename      = parts[-1]
            label_raw     = str(index["labels"][idx_i]).lower()
            label         = "malignant" if label_raw in ["1", "malignant"] else "benign"

            # Mount directory is 'Oral Cancer Dataset' which has 'CANCER' and 'NON CANCER'
            # original_path might point to 'Oral Cancer Dataset Processed' or 'Oral Cancer Dataset'
            # Let's just construct the local URL directly since we know the folder structure
            local_folder = "CANCER" if label == "malignant" else "NON CANCER"
            public_url = f"{PUBLIC_BASE_URL}/images/{local_folder}/{filename}"
            result = {
                "image_path": public_url,
                "label":      label,
                "similarity": float(sim),
                "index_i":    idx_i
            }

            if label == "benign":
                benign_results.append(result)
            else:
                malignant_results.append(result)

        # ── Sort + trim to TOP_K ──────────────────────────
        benign_results    = sorted(benign_results,    key=lambda x: x["similarity"], reverse=True)[:TOP_K]
        malignant_results = sorted(malignant_results, key=lambda x: x["similarity"], reverse=True)[:TOP_K]

        for i, r in enumerate(benign_results):
            r["rank"] = i + 1
        for i, r in enumerate(malignant_results):
            r["rank"] = i + 1

        # ── Decision rule (Priority 3) ────────────────────
        decision = compute_decision(benign_results, malignant_results)

        # ── Grad-CAM Explainability (Priority 8) ──────────
        # Only compute/show this when the decision actually IS "Malignant
        # pattern" — the heatmap explains similarity to the top malignant
        # match specifically ("why this looks concerning"), so surfacing it
        # on a Benign or Inconclusive result would show a hot malignant-
        # similarity map alongside a reassuring label, which reads as a
        # contradiction rather than an explanation.
        gradcam_base64 = None
        if malignant_results and decision["label"] == "Malignant pattern":
            best_mal = malignant_results[0]
            target_emb = index["embeddings"][best_mal["index_i"]]
            tensor_img = preprocess_image(img)
            if tensor_img is not None:
                tensor_img = tensor_img.unsqueeze(0).to(device)
                # Only the last 3 blocks (features[16:19]) were ever unfrozen
                # during fine-tuning — every earlier layer, including any with
                # higher spatial resolution, still holds untouched generic
                # ImageNet weights with no oral-lesion-specific signal. Using
                # one of those for Grad-CAM (tried features[13]) produces a
                # sharper-looking but meaningless map — confirmed by testing,
                # the hotspot moved onto background/teeth instead of the lesion.
                # features[-1] is coarse (7x7) but is the only layer that
                # actually reflects what the model was trained to recognize.
                target_layer = backbone.features[-1]
                with torch.enable_grad():
                    heatmap = get_guided_gradcam(backbone, target_layer, tensor_img, target_emb)
                if heatmap is not None:
                    gradcam_base64 = generate_gradcam_base64(img, heatmap)

        # Cleanup temp index
        for r in benign_results + malignant_results:
            r.pop("index_i", None)
        
        payload = {
            "image_hash":         image_hash,
            "cbir_top_k":         {
                "benign":    benign_results,
                "malignant": malignant_results,
            },
            "cbir_predicted_label": decision["label"],
            "cbir_confidence":      decision["confidence"],
            "index_version":        INDEX_VERSION,
            "decision_rule_version": DECISION_RULE_VERSION,
        }

        # ── Log to Supabase (never crashes the response) ──
        if session_id:
            update_supabase("sessions", session_id, payload)
        else:
            session_id = log_to_supabase("sessions", payload) or str(uuid.uuid4())

        return {
            "results": {
                "benign":    benign_results,
                "malignant": malignant_results,
            },
            "decision":       decision,
            "gradcam_base64": gradcam_base64,
            "session_id":     session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/search unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during image analysis.")


# ──────────────────────────────────────────────────────────
# ENDPOINT: POST /predict-risk  (Priority 4)
# ──────────────────────────────────────────────────────────
@app.post("/predict-risk")
async def predict_risk(request: PredictRiskRequest):
    """
    Score oral cancer risk from questionnaire answers using an evidence-based
    weighted rule engine (see compute_rule_based_risk for rationale).

    Request body: { "answers": { ...15 features... } }
    Returns: { risk_label, risk_score, top_features, session_id }
    """
    try:
        ans = request.answers
        result = compute_rule_based_risk(ans)
        risk_score = result["risk_score"]
        risk_label = result["risk_label"]
        top_features = result["top_features"]
        explanation_text = result["explanation_text"]

        payload = {
            "age":                      ans.age,
            "gender":                   "Male" if ans.gender == 1 else "Female/Other",
            "region":                   ans.region,
            "occupation":               ans.occupation,
            "questionnaire_answers":    ans.questionnaire_raw or {},
            "questionnaire_risk_label": risk_label,
            "questionnaire_risk_score": round(risk_score, 4),
        }

        # ── Log to Supabase ───────────────────────────────
        session_id = request.session_id
        if session_id:
            update_supabase("sessions", session_id, payload)
        else:
            session_id = log_to_supabase("sessions", payload) or str(uuid.uuid4())

        return {
            "risk_label":       risk_label,
            "risk_score":       round(risk_score, 4),
            "top_features":     top_features,
            "explanation_text": explanation_text,
            "session_id":       session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predict-risk unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during risk prediction.")


# ──────────────────────────────────────────────────────────
# ENDPOINT: POST /combined-risk  (Priority 5)
# ──────────────────────────────────────────────────────────
@app.post("/combined-risk")
async def combined_risk(request: CombinedRiskRequest):
    """
    Fuse questionnaire risk label + CBIR decision into a single triage output.
    Requires both tools to have been used in the same session.

    Request body: { "session_id": "<uuid>" }
    Returns: { urgency, combined_risk_label, recommendation }
    """
    if supabase_client is None:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Cannot retrieve session data.",
        )

    try:
        # ── Fetch session row ─────────────────────────────
        try:
            result = (
                supabase_client
                .table("sessions")
                .select("questionnaire_risk_label, cbir_predicted_label")
                .eq("id", request.session_id)
                .execute()
            )
        except Exception as e:
            logger.error(f"/combined-risk DB fetch error: {e}")
            raise HTTPException(status_code=503, detail="Database unavailable")

        if not result or not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="Session not found")

        row = result.data[0]
        q_label   = row.get("questionnaire_risk_label")
        cbir_label = row.get("cbir_predicted_label")

        if not q_label and not cbir_label:
            raise HTTPException(
                status_code=400,
                detail="No results found for this session. Complete the questionnaire or image matcher first."
            )

        # ── Fusion logic ──────────────────────────────────
        fusion = fuse_risk(q_label, cbir_label)

        # ── Update session row ────────────────────────────
        update_supabase(
            "sessions",
            request.session_id,
            {"combined_risk_label": fusion["combined_risk_label"]},
        )

        return fusion

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/combined-risk unhandled error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error generating combined risk.")


def verify_admin_key(x_admin_key: str = Header(...)):
    expected = os.environ.get("X_ADMIN_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Admin key not configured on server.")
    if x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return True

@app.get("/metrics")
def get_metrics(admin: bool = Depends(verify_admin_key)):
    """
    Priority 8: Live Validation Metrics
    """
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase not configured.")
        
    try:
        sessions_res = supabase_client.table("sessions").select("*").execute()
        labels_res = supabase_client.table("clinician_labels").select("*").execute()
        
        sessions = sessions_res.data
        labels = labels_res.data
        
        total_sessions = len(sessions)
        labelled_sessions = len(labels)
        
        # Build map for fast lookup
        sess_map = {s['id']: s for s in sessions}
        
        # Aggregation metrics
        cbir_tp = cbir_fp = cbir_tn = cbir_fn = 0
        quest_tp = quest_fp = quest_tn = quest_fn = 0
        
        by_gender = {}
        by_age = {"<40": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                  "40-60": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                  ">60": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}}
        by_occupation = {}
        by_index = {}
        by_rule = {}
        
        for lbl in labels:
            sid = lbl['session_id']
            if sid not in sess_map: continue
            sess = sess_map[sid]
            
            true_dx = lbl['actual_diagnosis'].lower()
            if true_dx not in ['malignant', 'benign']:
                continue
                
            cbir_pred = sess.get('cbir_predicted_label', '')
            if cbir_pred:
                pred = "malignant" if "malignant" in cbir_pred.lower() else ("benign" if "benign" in cbir_pred.lower() else "inconclusive")
                if pred != "inconclusive":
                    is_tp = true_dx == 'malignant' and pred == 'malignant'
                    is_fp = true_dx == 'benign' and pred == 'malignant'
                    is_tn = true_dx == 'benign' and pred == 'benign'
                    is_fn = true_dx == 'malignant' and pred == 'benign'
                    
                    if is_tp: cbir_tp += 1
                    if is_fp: cbir_fp += 1
                    if is_tn: cbir_tn += 1
                    if is_fn: cbir_fn += 1

            quest_pred = sess.get('questionnaire_risk_label', '')
            if quest_pred:
                q_pred = "malignant" if "malignant" in quest_pred.lower() or "high" in quest_pred.lower() else ("benign" if "benign" in quest_pred.lower() or "low" in quest_pred.lower() else "inconclusive")
                if q_pred != "inconclusive":
                    is_q_tp = true_dx == 'malignant' and q_pred == 'malignant'
                    is_q_fp = true_dx == 'benign' and q_pred == 'malignant'
                    is_q_tn = true_dx == 'benign' and q_pred == 'benign'
                    is_q_fn = true_dx == 'malignant' and q_pred == 'benign'
                    
                    if is_q_tp: quest_tp += 1
                    if is_q_fp: quest_fp += 1
                    if is_q_tn: quest_tn += 1
                    if is_q_fn: quest_fn += 1
                    
                    # Subgroup Age
                    age = sess.get('age')
                    if age is not None:
                        band = "<40" if age < 40 else (">60" if age > 60 else "40-60")
                        if is_q_tp: by_age[band]["TP"] += 1
                        if is_q_fp: by_age[band]["FP"] += 1
                        if is_q_tn: by_age[band]["TN"] += 1
                        if is_q_fn: by_age[band]["FN"] += 1

                    # Subgroup Gender
                    gender = sess.get('gender')
                    if gender:
                        g = gender.strip().capitalize()
                        if "/" in g:
                            g = "/".join(part.capitalize() for part in g.split("/"))
                        if g not in by_gender: by_gender[g] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_q_tp: by_gender[g]["TP"] += 1
                        if is_q_fp: by_gender[g]["FP"] += 1
                        if is_q_tn: by_gender[g]["TN"] += 1
                        if is_q_fn: by_gender[g]["FN"] += 1

                    # Subgroup Occupation
                    occupation = sess.get('occupation')
                    if occupation:
                        o = occupation.title()
                        if o not in by_occupation: by_occupation[o] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_q_tp: by_occupation[o]["TP"] += 1
                        if is_q_fp: by_occupation[o]["FP"] += 1
                        if is_q_tn: by_occupation[o]["TN"] += 1
                        if is_q_fn: by_occupation[o]["FN"] += 1

                    # Versioning
                    iv = sess.get('index_version')
                    if iv:
                        if iv not in by_index: by_index[iv] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_q_tp: by_index[iv]["TP"] += 1
                        if is_q_fp: by_index[iv]["FP"] += 1
                        if is_q_tn: by_index[iv]["TN"] += 1
                        if is_q_fn: by_index[iv]["FN"] += 1

                    drv = sess.get('decision_rule_version')
                    if drv:
                        if drv not in by_rule: by_rule[drv] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_q_tp: by_rule[drv]["TP"] += 1
                        if is_q_fp: by_rule[drv]["FP"] += 1
                        if is_q_tn: by_rule[drv]["TN"] += 1
                        if is_q_fn: by_rule[drv]["FN"] += 1
                        
        def calc_metrics(tp, fp, tn, fn):
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            def wilson(p, n, z=1.96):
                if n == 0: return 0.0, 0.0
                den = 1 + z**2/n
                cap = p + z**2 / (2*n)
                std = z * math.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
                return max(0.0, (cap - std)/den), min(1.0, (cap + std)/den)
                
            sens_l, sens_u = wilson(sens, tp + fn)
            spec_l, spec_u = wilson(spec, tn + fp)
            
            return {
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "accuracy": acc,
                "sensitivity_ci_lower": sens_l,
                "sensitivity_ci_upper": sens_u,
                "specificity_ci_lower": spec_l,
                "specificity_ci_upper": spec_u,
                "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
            }
            
        def clean_subgroup(d):
            res = {}
            for k, v in d.items():
                res[k] = calc_metrics(v["TP"], v["FP"], v["TN"], v["FN"])
            return res

        return {
            "total_sessions": total_sessions,
            "labelled_sessions": labelled_sessions,
            "image_metrics": calc_metrics(cbir_tp, cbir_fp, cbir_tn, cbir_fn),
            "questionnaire_metrics": calc_metrics(quest_tp, quest_fp, quest_tn, quest_fn),
            "subgroup_breakdown": {
                "by_gender": clean_subgroup(by_gender),
                "by_age_band": clean_subgroup(by_age),
                "by_occupation": clean_subgroup(by_occupation)
            },
            "by_index_version": clean_subgroup(by_index),
            "by_decision_rule_version": clean_subgroup(by_rule)
        }
    except Exception as e:
        logger.error(f"/metrics failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error retrieving metrics.")

@app.post("/clinician-label")
def clinician_label(req: ClinicianLabelRequest):
    """
    Priority 9: Ground truth capture mechanism for clinician mode.
    """
    payload = {
        "session_id": req.session_id,
        "clinician_id": req.clinician_id,
        "actual_diagnosis": req.actual_diagnosis,
        "notes": req.notes
    }
    row_id = log_to_supabase("clinician_labels", payload)
    if not row_id:
        raise HTTPException(status_code=500, detail="Failed to log label to Supabase.")
    return {"message": "Diagnosis recorded. Thank you.", "id": row_id}