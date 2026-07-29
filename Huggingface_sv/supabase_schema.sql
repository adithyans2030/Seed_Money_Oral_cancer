-- ============================================================
-- OralGuard — Supabase Schema
-- Paste this entire file into the Supabase SQL Editor and run.
-- ============================================================

-- Enable the uuid extension (already enabled on most Supabase projects)
create extension if not exists "pgcrypto";

-- ──────────────────────────────────────────────────────────
-- sessions
-- One row per user session (questionnaire, image match, or both)
-- ──────────────────────────────────────────────────────────
create table if not exists sessions (
    id                       uuid primary key default gen_random_uuid(),
    created_at               timestamptz not null default now(),

    -- Optional demographics (collected in frontend mini-form)
    age                      integer,
    gender                   text,
    region                   text,
    occupation               text,

    -- Questionnaire branch
    questionnaire_answers    jsonb,           -- raw {c1:"yes", s3:"no", ...} map
    questionnaire_risk_label text,            -- "Low" | "Moderate" | "High"
    questionnaire_risk_score double precision, -- 0.0 – 1.0 from XGBoost

    -- CBIR branch
    image_hash               text,            -- SHA-256 hex of uploaded image bytes
    cbir_top_k               jsonb,           -- [{label, similarity, rank}, ...] for top-6 of each class
    cbir_predicted_label     text,            -- "Benign pattern" | "Malignant pattern" | "Inconclusive"
    cbir_confidence          double precision, -- 0.0 – 1.0

    -- Combined fusion output
    combined_risk_label      text,            -- "Low — Monitor" | "High — Urgent Referral" etc.

    -- Validation & Version Tracking (Task 1)
    index_version            text,
    decision_rule_version    text
);

-- Index for time-series queries and session lookups
create index if not exists idx_sessions_created_at on sessions (created_at desc);

-- ──────────────────────────────────────────────────────────
-- clinician_labels
-- Clinicians label sessions after the fact to enable validation
-- ──────────────────────────────────────────────────────────
create table if not exists clinician_labels (
    id                uuid primary key default gen_random_uuid(),
    session_id        uuid not null references sessions(id) on delete cascade,
    clinician_id      text not null,           -- opaque identifier, no PII
    actual_diagnosis  text not null,           -- "benign" | "malignant" | "indeterminate"
    notes             text,
    labelled_at       timestamptz not null default now()
);

create index if not exists idx_clinician_labels_session on clinician_labels (session_id);

-- ──────────────────────────────────────────────────────────
-- Row Level Security
-- Service role key bypasses RLS, so the FastAPI backend
-- (using the service role key) can read/write everything.
-- The anon key is never used from the backend.
-- ──────────────────────────────────────────────────────────
alter table sessions        enable row level security;
alter table clinician_labels enable row level security;

-- No public-facing policies — all access goes through the FastAPI service role key
-- (If you later add a clinician dashboard, add policies here)
