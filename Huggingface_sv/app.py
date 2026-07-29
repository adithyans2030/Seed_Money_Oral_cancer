"""
OralGuard FastAPI Backend
=========================
Endpoints:
  GET  /                  — health check
  POST /search            — CBIR image search + decision rule + Supabase logging  (P3)
  POST /predict-risk      — XGBoost questionnaire risk scoring + Supabase logging  (P4)
  POST /combined-risk     — Fusion of both tools into a single triage output       (P5)
"""

import hashlib
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Literal

import joblib
import numpy as np
import math
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from scripts.preprocess import preprocess_image

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oralguard")

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
load_dotenv(override=True)  # loads .env if present locally; env vars take precedence on HF Spaces

INDEX_PATH     = "cbir_index.npz"
RISK_MODEL_PATH = "risk_model.pkl"
IMG_SIZE       = 224
TOP_K          = 6        # matches to return per class
INCONCL_THRESH = 0.765    # Tuned threshold for MobileNetV2 cosine similarity mean of top k

# Validation Versions
INDEX_VERSION = "index_v2_20250723"
DECISION_RULE_VERSION = "rule_v1_k5_t0.08"

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
    """Update an existing row. Returns True on success. Never raises."""
    if supabase_client is None:
        return False
    try:
        supabase_client.table(table).update(payload).eq("id", row_id).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase update on '{table}' id={row_id} failed: {e}")
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
device = torch.device("cpu")

backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model = nn.Sequential(
    backbone.features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
).to(device)
model.eval()

logger.info("MobileNetV2 feature extractor loaded.")

# ──────────────────────────────────────────────────────────
# LOAD XGBOOST RISK MODEL (runs once at startup — optional)
# ──────────────────────────────────────────────────────────
risk_model = None
if os.path.exists(RISK_MODEL_PATH):
    try:
        risk_model = joblib.load(RISK_MODEL_PATH)
        logger.info(f"XGBoost risk model loaded from {RISK_MODEL_PATH}.")
    except Exception as e:
        logger.error(f"Failed to load risk model: {e}. /predict-risk will be unavailable.")
else:
    logger.warning(
        f"{RISK_MODEL_PATH} not found. "
        "Run scripts/train_risk_model.py locally and commit the output file."
    )

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(image: Image.Image) -> np.ndarray:
    """Extract L2-normalised MobileNetV2 feature vector."""
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

app.mount("/images", StaticFiles(directory=r"c:\Users\adith\OneDrive\Desktop\seed_money_project\Oral Cancer\Oral Cancer Dataset"), name="images")


# ──────────────────────────────────────────────────────────
# ENDPOINT: GET /
# ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":    "OralGuard API is running.",
        "version":    "2.0.0",
        "index_size": len(index["paths"]),
        "risk_model": "loaded" if risk_model is not None else "unavailable",
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
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
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
            label_raw     = str(index["labels"][idx_i])
            label         = "malignant" if label_raw == "1" else "benign"

            if original_path.startswith("new_data/"):
                local_folder = "CANCER" if label == "malignant" else "NON CANCER"
                public_url = f"http://127.0.0.1:8000/images/{local_folder}/{filename}"
            else:
                public_url = (
                    f"https://huggingface.co/datasets/GPrabhanjana/oral-images"
                    f"/resolve/main/{class_folder}/{filename}"
                )
            result = {
                "image_path": public_url,
                "label":      label,
                "similarity": float(sim),
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
            "decision":   decision,
            "session_id": session_id,
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
    Run XGBoost oral cancer risk prediction from questionnaire answers.

    Request body: { "answers": { ...15 features... } }
    Returns: { risk_label, risk_score, top_features, session_id }
    """
    if risk_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk prediction model is not available. "
                "Please ensure risk_model.pkl is present in the container."
            ),
        )

    try:
        ans = request.answers

        # ── Build feature vector (must match training column order) ──
        # Order: age, gender, tobacco_use, alcohol_consumption, hpv_infection,
        #        betel_quid_use, chronic_sun_exposure, poor_oral_hygiene, diet,
        #        family_history, compromised_immune, oral_lesions,
        #        unexplained_bleeding, difficulty_swallowing, white_red_patches
        feature_names = [
            "age", "gender", "tobacco_use", "alcohol_consumption",
            "hpv_infection", "betel_quid_use", "chronic_sun_exposure",
            "poor_oral_hygiene", "diet", "family_history",
            "compromised_immune", "oral_lesions", "unexplained_bleeding",
            "difficulty_swallowing", "white_red_patches",
        ]
        feature_vector = np.array([[
            ans.age,
            ans.gender,
            ans.tobacco_use,
            ans.alcohol_consumption,
            ans.hpv_infection,
            ans.betel_quid_use,
            ans.chronic_sun_exposure,
            ans.poor_oral_hygiene,
            ans.diet,
            ans.family_history,
            ans.compromised_immune,
            ans.oral_lesions,
            ans.unexplained_bleeding,
            ans.difficulty_swallowing,
            ans.white_red_patches,
        ]], dtype=float)

        # ── Inference ─────────────────────────────────────
        risk_score = float(risk_model.predict_proba(feature_vector)[0][1])
        risk_label = score_to_label(risk_score)

        # ── Top feature importances ───────────────────────
        global_importances = risk_model.feature_importances_
        # Only consider risk factors the user actually has (value > 0)
        # Age and Gender are always > 0 but we generally want to highlight modifiable risks
        user_features_present = feature_vector[0] > 0
        local_importances = global_importances * user_features_present

        ranked = sorted(zip(feature_names, local_importances), key=lambda x: x[1], reverse=True)
        
        # Filter out features with 0 local importance, but don't show Age/Gender as primary actionable risks unless needed
        top_features = [name for name, imp in ranked if imp > 0 and name not in ("age", "gender")][:3]
        if not top_features:
            top_features = [name for name, imp in ranked if imp > 0][:3]
        if not top_features:
            top_features = ["age"]
        
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
            "risk_label":   risk_label,
            "risk_score":   round(risk_score, 4),
            "top_features": top_features,
            "session_id":   session_id,
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
    expected = os.environ.get("X_ADMIN_KEY", "oralguard_admin_dev")
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
                        if is_tp: by_age[band]["TP"] += 1
                        if is_fp: by_age[band]["FP"] += 1
                        if is_tn: by_age[band]["TN"] += 1
                        if is_fn: by_age[band]["FN"] += 1
                        
                    # Subgroup Gender
                    gender = sess.get('gender')
                    if gender:
                        g = gender.capitalize()
                        if g not in by_gender: by_gender[g] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_tp: by_gender[g]["TP"] += 1
                        if is_fp: by_gender[g]["FP"] += 1
                        if is_tn: by_gender[g]["TN"] += 1
                        if is_fn: by_gender[g]["FN"] += 1
                        
                    # Subgroup Occupation
                    occupation = sess.get('occupation')
                    if occupation:
                        o = occupation.title()
                        if 'by_occupation' not in locals(): by_occupation = {}
                        if o not in by_occupation: by_occupation[o] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_tp: by_occupation[o]["TP"] += 1
                        if is_fp: by_occupation[o]["FP"] += 1
                        if is_tn: by_occupation[o]["TN"] += 1
                        if is_fn: by_occupation[o]["FN"] += 1
                        
                    # Versioning
                    iv = sess.get('index_version')
                    if iv:
                        if iv not in by_index: by_index[iv] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_tp: by_index[iv]["TP"] += 1
                        if is_fp: by_index[iv]["FP"] += 1
                        if is_tn: by_index[iv]["TN"] += 1
                        if is_fn: by_index[iv]["FN"] += 1
                        
                    drv = sess.get('decision_rule_version')
                    if drv:
                        if drv not in by_rule: by_rule[drv] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                        if is_tp: by_rule[drv]["TP"] += 1
                        if is_fp: by_rule[drv]["FP"] += 1
                        if is_tn: by_rule[drv]["TN"] += 1
                        if is_fn: by_rule[drv]["FN"] += 1
                        
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
                "by_occupation": clean_subgroup(locals().get('by_occupation', {}))
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