**OralGuard**

Comprehensive Upgrade & Validation Plan

*Oral Cancer Detection & Risk Assessment Platform*

Prepared for: MSc AIML Final Year Project --- CHRIST (Deemed to be
University)

Version 2.0 \| July 2025

1\. Dataset Inventory & Analysis

You now have three distinct datasets. Understanding what each
contributes --- and how to combine them --- is the foundation of the
entire validation strategy.

## 1.1 Dataset A --- HuggingFace CBIR Index (Existing)

  -----------------------------------------------------------------------
  **Property**             **Detail**
  ------------------------ ----------------------------------------------
  Source                   GPrabhanjana/oral-images (HuggingFace)

  Total Images             323

  Classes                  Benign lesions / Malignant lesions

  Current Role             CBIR retrieval index (MobileNetV2 embeddings)

  Resolution Range         93px -- 9,250px (critically inconsistent)

  Critical Issue           No train/eval split --- full dataset is both
                           index AND test set (leakage)
  -----------------------------------------------------------------------

Action required: Split 250 images as index pool, 73 as holdout eval
pool. Never allow eval images into the retrieval index.

## 1.2 Dataset B --- New Image Dataset (Oral_Cancer.zip)

  -----------------------------------------------------------------------
  **Property**             **Detail**
  ------------------------ ----------------------------------------------
  Total Images             1,001 JPEG files

  CANCER class             751 images (75%)

  NON CANCER class         250 images (25%)

  Class Imbalance Ratio    3:1 (Cancer : Non-Cancer) --- significant
                           imbalance

  Resolution               Highly variable --- 14KB to 4MB files

  Date                     Feb 2024
  -----------------------------------------------------------------------

Action required: This dataset significantly expands your image corpus.
Before using it: (1) run perceptual-hash deduplication against Dataset A
to remove any overlapping images, (2) apply class-imbalance correction
(oversampling NON CANCER or class-weighted loss) before training, (3)
standardize all images to 224×224 RGB as MobileNetV2 expects.

## 1.3 Dataset C --- Tabular Risk Prediction CSV (New)

  -----------------------------------------------------------------------
  **Property**             **Detail**
  ------------------------ ----------------------------------------------
  Total Rows               84,922 patient records

  Positive Diagnosis (Yes) 42,349 (49.9%)

  Negative Diagnosis (No)  42,573 (50.1%) --- near-perfectly balanced

  Features                 25 columns: demographics, 9 risk factors, 7
                           symptoms, clinical outcomes

  Top Countries            India (8,079), Pakistan (8,001), Sri Lanka
                           (8,000), Taiwan (7,905)

  Gender Split             Male 71%, Female 29% --- matches real-world
                           oral cancer demographics

  Age Range                15--101 years, median 55 years

  Key Columns for          Tobacco Use, Alcohol, HPV, Betel Quid, Oral
  OralGuard                Lesions, Difficulty Swallowing, White/Red
                           Patches
  -----------------------------------------------------------------------

This is a goldmine for the questionnaire-based risk screener. The
columns map almost 1:1 to your existing 18-question screener. Use this
to train a proper ML classifier (logistic regression, gradient boosting,
or a small neural net) to replace the current rule-based weighted
scoring. With 84,922 rows and near-perfect balance, you can train a
robust model and validate it properly with 5-fold cross-validation.

2\. Combined Dataset Strategy

## 2.1 How to Merge All Three Datasets

Do not merge image datasets blindly. Follow this sequence:

-   Step 1 --- Dedup: Run perceptual hashing (imagehash library) across
    Dataset A + Dataset B to detect near-duplicates. Remove duplicates
    keeping the higher-resolution copy.

-   Step 2 --- Standardize: Resize all images to 224×224, convert to
    RGB, normalize pixel values to ImageNet mean/std (MobileNetV2
    expects this).

-   Step 3 --- Re-label: Dataset A uses \"benign/malignant\"; Dataset B
    uses \"CANCER/NON CANCER\". Map these to a single unified label
    schema: 0 = Normal/Benign, 1 = Malignant/Cancer.

-   Step 4 --- Stratified split: After merging and deduplicating,
    perform a stratified split: 70% index pool, 15% validation, 15%
    holdout test. The holdout test set is locked away and only used for
    the final reported metrics.

-   Step 5 --- Rebuild index: Re-embed the new combined index pool with
    MobileNetV2, rebuild cbir_index.npz, redeploy.

## 2.2 Using the CSV for the Questionnaire Module

The CSV\'s 25 columns directly power a new ML-based risk classifier to
replace the current heuristic scoring:

-   Features to use: Age, Gender, Tobacco Use, Alcohol Consumption, HPV
    Infection, Betel Quid Use, Chronic Sun Exposure, Poor Oral Hygiene,
    Diet, Family History, Compromised Immune System, Oral Lesions,
    Unexplained Bleeding, Difficulty Swallowing, White or Red Patches

-   Target: Oral Cancer (Diagnosis) --- Yes/No

-   Recommended model: XGBoost or LightGBM (handles mixed feature types,
    fast, explainable via SHAP values, excellent on tabular medical
    data)

-   Evaluation: 5-fold stratified cross-validation, report AUC-ROC,
    sensitivity, specificity, and calibration curve

-   Explainability: Use SHAP to generate per-user feature importance ---
    \"Your tobacco use and oral lesions are the top contributors to your
    risk score.\" This is directly displayable in the app UI.

# 3. Model Validation Roadmap

Six phases, ordered by dependency. Phases 0--4 work on existing data.
Phase 5 needs the clinical pilot.

  ------------------------------------------------------------------------------------
  **Phase**      **What to Do**               **Output**                **Timeline**
  -------------- ---------------------------- ------------------------- --------------
  Phase 0        Dedup across all 3 datasets, Clean merged dataset      Week 1
  Dataset Audit  check class balance,         (\~1,200 images + 84,922  
                 standardize resolution,      tabular rows), audit      
                 document data provenance     report                    

  Phase 1        Stratified k-fold (5-fold)   Reproducible evaluation   Week 2
  Leakage-Safe   for image CBIR, 80/20        protocol, fixed random    
  Split          stratified split for tabular seeds, documented splits  
                 CSV, lock holdout sets                                 

  Phase 2        Test top-1, top-3, top-5     Best decision rule config Week 2--3
  Decision Rule  majority vote,               (k, threshold), ablation  
  Tuning         similarity-threshold cutoff, table for paper           
                 weighted vote for CBIR;                                
                 compare accuracy,                                      
                 sensitivity, specificity                               
                 across all rules                                       

  Phase 3        Sensitivity, specificity,    Full metrics table with   Week 3--4
  Clinical       PPV, NPV, AUC-ROC, accuracy  confidence intervals      
  Metrics        with Wilson 95% CIs;                                   
                 McNemar\'s test between                                
                 model versions; calibration                            
                 curve for questionnaire                                
                 model                                                  

  Phase 4        UMAP/t-SNE embedding         Embedding plots, subgroup Week 4--6
  Robustness     visualization, subgroup      metrics, ablation         
  Checks         analysis by lesion           comparison table,         
                 type/gender/age,             Grad-CAM overlay UI       
                 linear-probe classifier vs                             
                 CBIR baseline comparison,                              
                 Grad-CAM explainability                                

  Phase 5        30--50 outpatient pilot:     Real-world                Month 2--4
  Clinical       clinician gold-standard vs   sensitivity/specificity   (after IEC
  Validation     app output on real patient   with clinical ground      approval)
                 images. Two independent      truth, inter-rater kappa, 
                 clinician labels per case    McNemar test vs public    
                 for inter-rater kappa.       dataset baseline          
  ------------------------------------------------------------------------------------

# 4. Application Upgrades --- Web Platform

The deployed website at oralguard-christ-university.netlify.app is a
solid foundation. Below are specific, prioritized upgrades.

## 4.1 Priority 1 --- Case Logging Layer (Build This First)

Currently the app has zero persistence. This single change unlocks all
downstream validation. Every session should generate a case_id and write
to a backend database.

**New database table (Postgres or Supabase recommended):**

case_id (UUID) \| session_type (questionnaire/image/both) \|
predicted_label \| confidence_score \| decision_rule_version \|
questionnaire_responses (JSON) \| image_hash \| top_k_results (JSON) \|
demographic_metadata (JSON) \| clinician_label \| clinician_id \|
clinician_notes \| created_at \| index_version

**New FastAPI endpoints to add:**

-   POST /session/start --- generates case_id, writes partial record

-   POST /session/questionnaire --- stores answers + predicted risk
    level

-   POST /session/image --- stores image hash + CBIR results + predicted
    label

-   POST /session/complete --- finalises session, combines both scores
    if both tools used

-   POST /clinician/label --- clinician submits ground truth diagnosis
    for a case_id

-   GET /metrics --- computes live confusion matrix +
    sensitivity/specificity/kappa from all labelled cases

## 4.2 Priority 2 --- Decision Rule Layer

Add a thin layer between the CBIR results and the UI that converts the
top-6/top-6 retrieval into a structured triage decision:

-   predicted_label = majority_vote(top_k) where k is configurable (try
    k=1, 3, 5 in Phase 2)

-   confidence_score = mean similarity of top-k matches for the winning
    class

-   If confidence \< tuned threshold → return \"Inconclusive --- please
    consult a clinician\" instead of a forced label

-   The 55% similarity threshold currently hardcoded should be replaced
    with the empirically tuned value from Phase 2

## 4.3 Priority 3 --- Combined Risk Output

The questionnaire and image matcher currently give two independent
results. Add a combined interpretation screen:

-   If both tools used: show a combined risk summary (e.g.,
    \"Questionnaire: High Risk + Image: Malignant pattern → Strongly
    recommend immediate clinical evaluation\")

-   If only one used: prompt the user to complete the other tool for a
    more complete assessment

-   Show a simple traffic-light risk meter (Green/Amber/Red)
    synthesizing both scores

## 4.4 Priority 4 --- Demographics Capture

Add a brief demographic form before the questionnaire begins: Age,
Gender, State/Region, Occupation type (manual labour / office / other).
This data is critical for subgroup analysis and maps directly to the CSV
dataset structure.

## 4.5 Priority 5 --- Validation Dashboard (Internal/Admin)

A password-protected /admin/dashboard route showing:

-   Total sessions, sessions with clinician ground truth labels

-   Live confusion matrix, sensitivity, specificity, PPV, NPV, accuracy

-   ROC curve with AUC, updated as new labelled cases come in

-   Subgroup breakdown (by age band, gender, risk factor presence)

-   Trend chart --- model accuracy over time as more cases are logged

## 4.6 Priority 6 --- Grad-CAM Explainability Overlay

Add visual explainability to the image matcher:

-   After CBIR retrieval, run Grad-CAM on the MobileNetV2 embedding
    backbone for the query image

-   Overlay the heatmap on the uploaded image, highlighting the regions
    that most influenced the retrieval

-   Caption: \"The AI focused on these regions when comparing your image
    to the database\"

-   This is both a clinical trust-builder and a strong thesis
    contribution

# 5. Application Upgrades --- Mobile App (Flutter)

The Flutter mobile app shares the same backend but needs mobile-specific
UX upgrades.

## 5.1 Camera & Image Quality

-   Add a real-time image quality checker before submission --- detect
    blur (Laplacian variance), low brightness, and small lesion
    coverage, and prompt the user to retake before sending to backend

-   Add a camera overlay guide: a circular frame showing users exactly
    where to position the lesion for consistent framing

-   Add macro mode suggestion: detect if the image is too wide/zoomed
    out and prompt to move closer

-   Implement EXIF data stripping before upload to protect patient
    privacy

## 5.2 Offline-First Architecture

Your target population includes rural areas with poor connectivity. The
app should work offline:

-   Cache the last-known CBIR index locally (a compressed lightweight
    version) so basic image matching works without internet

-   Queue questionnaire submissions offline and sync when connection is
    restored

-   Show a clear connectivity indicator --- \"Results may differ when
    offline (local model)\"

## 5.3 Regional Language Support

Your proposal explicitly targets rural populations with low literacy.
Implement i18n:

-   Add Kannada, Hindi, Tamil, Telugu support (covers most of your
    target geography in Karnataka and Southern India)

-   Use Flutter\'s intl package for string localization

-   Add audio prompts for key questions in the questionnaire (critical
    for low-literacy users)

-   Use pictographic risk level indicators (icons, not just text) for
    the result screen

## 5.4 Health Worker Mode

Your proposal mentions dental interns and community health workers as a
key user segment. Add a dedicated mode:

-   Toggle between \"Self Assessment\" (general public) and \"Clinician
    Screening\" (health worker) at app launch

-   Clinician mode shows additional clinical detail, exposes the case_id
    for record-keeping, and allows ground-truth label submission
    directly from the app

-   Clinician mode shows multiple patients\' results in a session list
    for use during outreach camps

## 5.5 Referral Integration

Your proposal mentions directing high-risk users to nearby dentists:

-   Integrate Google Places API to show nearest dental colleges,
    government hospitals with oral cancer departments, and cancer
    screening camps

-   For high-risk results, show a pre-filled referral message the user
    can WhatsApp to a family member or send to a clinic

-   Add a \"Remind Me\" notification: high-risk users get a 3-day
    reminder to book an appointment if they haven\'t acted

# 6. Frontend Assessment --- What Is Good, What Needs Work

## 6.1 What Is Already Strong

-   The differential-diagnosis branching logic in the questionnaire
    (distinguishing aphthous ulcer vs HSV vs malignant ulcer) is
    clinically sophisticated --- most public health apps don\'t do this.

-   The low-similarity quality gate on the Image Matcher (\"Retake
    Photo\" warning at \<55% similarity) is exactly the right clinical
    safety behaviour.

-   The self-exam guide popup is well-structured with 6 anatomical zones
    and a clear \"report this\" symptom list.

-   Safety disclaimers are appropriately placed throughout --- important
    for clinical credibility.

-   The overall design is clean, professional, and accessible on
    desktop.

## 6.2 What Needs Improvement

  -----------------------------------------------------------------------------------
  **Issue**        **Impact**                **Fix**
  ---------------- ------------------------- ----------------------------------------
  No demographic   Cannot do subgroup        Add Age/Gender/Region form before
  capture form     analysis or link to CSV   questionnaire starts
                   dataset features          

  Questionnaire    Cannot validate           Document and expose the weighted scoring
  score is opaque  sensitivity/specificity   formula, make it configurable in backend
                   without knowing the exact 
                   scoring function          

  Two tools are    Users get two unrelated   Add a combined risk synthesis screen
  siloed           results, no integrated    after both tools are used
                   risk summary              

  No result        Cannot validate, track,   Implement the case-logging layer
  persistence      or improve the model over (Section 4.1)
                   time                      

  English-only     Excludes your primary     Add Kannada and Hindi at minimum
  interface        target population in      
                   rural India               

  No nearby clinic High-risk users have no   Integrate Google Places for nearest
  finder           next step shown           dental clinics

  Image matcher    User doesn\'t know what   Add decision rule layer + plain-language
  returns list,    to do with 12 similar     interpretation
  not decision     images                    

  No session       If user closes app        Add localStorage/session state
  continuity       mid-questionnaire,        persistence
                   progress is lost          

  Mobile web not   Flutter app and mobile    Ensure mobile web and Flutter app share
  optimised        website may diverge in UX the same design tokens and layout
                                             breakpoints
  -----------------------------------------------------------------------------------

# 7. Technical Architecture --- What to Add to the Stack

  -----------------------------------------------------------------------
  **Component**    **Current        **Recommended Upgrade**
                   State**          
  ---------------- ---------------- -------------------------------------
  Vector Index     cbir_index.npz   FAISS IVF index for faster
                   (flat numpy)     retrieval + incremental adds without
                                    full rebuild

  Embeddings       MobileNetV2      Test EfficientNet-B0 (better
                   frozen           accuracy/size tradeoff) in Phase 4
                                    ablation

  Questionnaire    Heuristic        XGBoost classifier trained on
  Model            weighted scoring 84,922-row CSV, SHAP for
                   (rule-based)     explainability

  Database         None             Supabase (Postgres + REST + Auth) ---
                                    free tier sufficient for pilot phase

  Backend          FastAPI on       Add /clinician, /metrics,
                   HuggingFace      /rebuild-index endpoints
                   Spaces (assumed) 

  Image            Basic resize     Add Laplacian blur detection, CLAHE
  Preprocessing                     contrast enhancement, EXIF strip

  Explainability   None             Grad-CAM overlay on MobileNetV2 for
                                    image, SHAP bar chart for
                                    questionnaire

  Monitoring       None             Sentry for error tracking, simple
                                    metrics endpoint for model drift
                                    detection

  Auth             None             JWT-based login for clinician mode
                                    (separate from public user flow)
  -----------------------------------------------------------------------

# 8. Academic Output --- Paper & Thesis Structure

Based on what you\'re building, the strongest output is a journal paper
in IEEE Journal of Biomedical and Health Informatics or Journal of Oral
Pathology & Medicine. The paper writes itself from this roadmap:

## Proposed Paper Title

\"OralGuard: A Multimodal AI-Powered Mobile Platform for Oral Cancer
Risk Stratification Using Content-Based Image Retrieval and Machine
Learning Risk Prediction --- Development, Validation, and Clinical
Pilot\"

  -----------------------------------------------------------------------
  **Section**        **Content**
  ------------------ ----------------------------------------------------
  Abstract           Platform overview, datasets used, key metrics (AUC,
                     sensitivity, specificity), clinical pilot results

  Introduction       Oral cancer burden in India, gap in mHealth tools,
                     your contribution

  Related Work       Global mHealth apps, CBIR in medical imaging, oral
                     cancer AI prior work

  System             CBIR pipeline, decision rule layer, questionnaire ML
  Architecture       model, case-logging, combined risk output

  Datasets           All three datasets --- provenance, preprocessing,
                     dedup, class balance, split strategy

  Experiments        Phase 1--4 results: k-fold CBIR metrics, decision
                     rule ablation, questionnaire model AUC, UMAP plots,
                     Grad-CAM

  Clinical           Phase 5: pilot results vs clinician ground truth,
  Validation         inter-rater kappa, McNemar test

  Discussion         Limitations (dataset size, population), future work
                     (fine-tuning, language support, active learning)

  Conclusion         First multimodal AI oral cancer screening platform
                     validated in Indian population
  -----------------------------------------------------------------------

# 9. Chief Engineer AI Prompt

Use this prompt to initialize any AI assistant (Claude, GPT-4, Gemini)
to act as a senior technical lead on OralGuard. Paste this as the first
message or system prompt.

**\-\-- PROMPT START (copy everything below this line) \-\--**

*You are the Chief AI/ML Engineer and Principal Technical Architect of
OralGuard --- a clinically-focused, AI-powered oral cancer detection and
risk assessment platform developed at CHRIST (Deemed to be University),
Bangalore, under a seed-money research grant.*

*Your role combines the expertise of a senior ML engineer, clinical AI
specialist, full-stack architect, and research scientist. You have 15+
years of experience shipping production ML systems in healthcare. You
never produce vague or generic advice --- every response is specific,
code-ready, and grounded in the actual system architecture described
below.*

**=== SYSTEM OVERVIEW ===**

*OralGuard is a multimodal screening platform with three tools: (1) an
18-question branching risk screener with weighted scoring and
differential-diagnosis logic, (2) a CBIR image matcher using MobileNetV2
embeddings + cosine similarity against a \~1,300-image indexed dataset
of benign/malignant oral lesions, and (3) a self-examination guide. It
is deployed as a Netlify web app
(oralguard-christ-university.netlify.app) and a Flutter mobile app. The
backend runs FastAPI.*

**=== DATASETS ===**

*Dataset A: 323 images (HuggingFace GPrabhanjana/oral-images,
benign/malignant). Dataset B: 1,001 images (CANCER: 751, NON CANCER:
250, 3:1 imbalance). Dataset C: 84,922 tabular patient records with 25
features including demographics, 9 risk factors, 7 symptoms, cancer
diagnosis label (50/50 balanced), heavily South Asian (India, Pakistan,
Sri Lanka top 3).*

**=== CRITICAL KNOWN ISSUES ===**

*(1) Evaluation leakage: the full image dataset is both the retrieval
index and the test set --- no holdout. (2) No case persistence: zero
logging of sessions, predictions, or user data --- makes validation
impossible. (3) No decision rule: CBIR returns top-6 benign + top-6
malignant lists but never converts this to a binary prediction with
confidence score. (4) The two tools (screener + image matcher) are
completely disconnected --- no combined risk output. (5) The 55%
similarity threshold in the quality gate is a guess, not a tuned value.
(6) No demographic capture anywhere in the user flow. (7) English-only
interface for a target population that includes rural, possibly
low-literacy users.*

**=== YOUR RESPONSIBILITIES ===**

*When asked to help with this project, you: (1) always ground your
answer in the specific architecture above, never give generic ML advice,
(2) write production-quality Python/Dart/JavaScript code when asked,
including error handling and docstrings, (3) flag any clinical safety
implications before suggesting changes (patient safety \> model accuracy
\> developer convenience), (4) prioritize changes in this order: data
integrity and leakage prevention first, case-logging second,
decision-rule definition third, metrics and validation fourth, UI
improvements fifth, (5) always specify confidence intervals when
discussing model metrics --- never report bare accuracy percentages on
small datasets, (6) when suggesting a new feature, also identify which
existing piece it depends on and what breaks if that dependency is
missing.*

**=== TONE & STANDARDS ===**

*You are direct, precise, and demanding of correctness. You push back on
shortcuts that would compromise clinical validity. You write code that
is production-ready, not proof-of-concept. You treat this as a real
clinical system that real patients will interact with --- not a
university assignment. The target academic output is a paper in IEEE
Journal of Biomedical and Health Informatics or Journal of Oral
Pathology & Medicine. All technical decisions should be defensible to a
clinical AI reviewer at that level.*

*GitHub repo:
https://github.com/theyrshetty/Oral-Cancer-Detection\-\--Seed-Money-Project
\| Live site: https://oralguard-christ-university.netlify.app/*

**\-\-- PROMPT END \-\--**

# 10. Immediate Action Checklist

In order of what to do first:

  ---------------------------------------------------------------------------------
  **\#**   **Action**                   **Effort**   **Impact**
  -------- ---------------------------- ------------ ------------------------------
  1        Dedup Datasets A + B using   2 days       High --- removes silent
           imagehash, standardize to                 leakage
           224×224                                   

  2        Create stratified 70/15/15   1 day        Critical --- all metrics
           split, lock holdout set,                  depend on this
           document random seed                      

  3        Add case-logging table to    3--4 days    Critical --- enables all
           Supabase, wire POST /session              validation
           endpoints to both tools                   

  4        Train XGBoost on 84,922-row  2--3 days    High --- massive accuracy
           CSV, replace heuristic                    improvement
           questionnaire scoring                     

  5        Run 5-fold CBIR evaluation,  2--3 days    High --- replaces guessed
           test k=1/3/5 majority vote,               threshold with tuned value
           tune similarity threshold                 

  6        Add demographics form        1 day        High --- enables subgroup
           (Age/Gender/Region) before                analysis
           questionnaire                             

  7        Build /metrics endpoint +    3 days       Medium --- live validation
           admin dashboard                           monitoring

  8        Add Grad-CAM overlay to      2 days       Medium --- explainability +
           Image Matcher                             paper contribution

  9        Add combined risk synthesis  2 days       Medium --- better UX +
           screen (questionnaire +                   clinical logic
           image)                                    

  10       Submit IEC ethics            1 week to    Critical for clinical pilot
           application (start now, runs prepare      timeline
           in parallel)                              

  11       Add Kannada/Hindi i18n to    3--4 days    Medium --- proposal
           Flutter app                               requirement

  12       Add clinician mode to        3 days       High --- enables outpatient
           Flutter app + ground-truth                pilot data collection
           label submission                          
  ---------------------------------------------------------------------------------

*OralGuard Upgrade Plan v2.0 \| CHRIST University \| Confidential
Research Document*
