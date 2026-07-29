# OralGuard - Oral Cancer Detection System

A content-based image retrieval (CBIR) system and symptom risk screener for oral cancer diagnosis support.
Upload an oral lesion image to retrieve visually similar cases from a curated dataset, classified as benign or malignant. Complete the risk screener to receive a fused triage recommendation to assist clinical decision-making.

---

## Repository Structure

```
├── Documents/          # SRS, UI design, literature review, PPT, conference papers
├── Huggingface_sv/     # FastAPI backend with XGBoost and CBIR logic
├── website_react/      # React web frontend (Vite)
├── website/            # Legacy Web frontend
├── oralguard_flutter/  # Flutter mobile application (Android/iOS)
└── Output/             # Release APK ready for download
```

---

## System Components

### Backend — FastAPI
Located in `Huggingface_sv/`

- MobileNetV2 backbone extracts feature embeddings from uploaded images for CBIR.
- Cosine similarity search against a pre-built index (`cbir_index.npz`) returning top benign/malignant matches.
- XGBoost classification model evaluates patient risk factors from a questionnaire.
- Advanced Triage Fusion logic combines both the questionnaire and image similarity for a clinical recommendation.
- UUID session tracking and persistent storage in Supabase.

### Web Frontend — React
Located in `website_react/`

- Modern React/Vite application with responsive design.
- Image Matcher page with immediate clinical similarity search results.
- Questionnaire page with real-time dynamic risk calculation.
- Combined Risk view for a comprehensive holistic patient assessment.

### Mobile Application — Flutter
Located in `oralguard_flutter/`

- Cross-platform Flutter app (Android/iOS).
- Upload from gallery or capture directly with camera.

---

## Recent Updates & Bug Fixes
- **Demographics & Subgroup Analysis**: Added Occupation field (Manual Labour, Office, Other) across the entire stack (React, Flutter, FastAPI, and Postgres). The `/metrics` endpoint and local `subgroup_analysis.py` scripts now calculate sensitivity and specificity by Occupational subgroups to ensure clinical fairness.
- **Triage Fusion Engine**: Updated `/combined-risk` to robustly handle partial data gracefully. If a patient only uses one of the two tools, the system provides an immediate fallback recommendation instead of failing.
- **XGBoost Feature Importance**: Local feature importance logic was implemented. The system now accurately lists the exact, specific risk factors that contributed most heavily to *your* personal risk profile.
- **Decision Rule Tuning**: Implemented a k-NN (k=5) majority voting mechanism for image retrieval with an empirically tuned similarity threshold for improved specificity.
- **API Reliability**: Fixed Supabase Postgres row tracking. Refactored React promises to prevent race conditions during triage calculation. Cleaned up error boundaries.

---

## Running Locally

### Backend
Start the Python backend server using uvicorn:
```bash
cd Huggingface_sv
pip install -r requirements.txt
uvicorn app:app --reload
```
API available at `http://127.0.0.1:8000`

### Frontend
Start the React frontend development server:
```bash
cd website_react
npm install
npm run dev
```

---

## License

MIT
