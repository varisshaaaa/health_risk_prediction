# Health Risk Prediction System - System Overview

## 1. High-Level Architecture
The project is a **Monorepo** containing two main deployable services:

1.  **Backend API (FastAPI)**: The brain of the operation. Handles logic, ML models, database interactions, and web scraping.
2.  **Frontend (Streamlit)**: The user interface. Sends data to the backend and displays results.

**Deployment**: Both services are containerized using Docker and deployed to **Railway** via **GitHub Actions**.

---

## 2. Detailed Backend Logic (`backend/`)

### A. Core Components
-   **API Entry Point**: `backend/main.py` (Startup/Lifespan) -> `backend/api/health.py` (Endpoints).
-   **Database**: PostgreSQL.
    -   *Tables*:
        -   `feature_store` (Logs every prediction).
        -   `precautions` (Stores scraped medical advice).
        -   `training_data` (The source of truth for the ML model).
-   **ML Models**: stored in `backend/ml_models/`.
    -   `health_impact_predictor.pkl`: Predicts environmental risk from AQI.
    -   `symptoms_and_disease.pkl`: Predicts disease from symptoms.

### B. The Prediction Pipeline (`POST /predict`)
When a user submits data, the backend executes this flow:

1.  **Environmental Analysis** (`services/health_features.py`):
    -   Fetches real-time weather/AQI from OpenWeatherMap.
    -   Passes data to `health_impact_predictor.pkl`.
    -   Result: **Health Impact Score (20% weight)**.

2.  **Demographic Analysis** (`services/demographic_risk.py`):
    -   Calculates risk based on Age and Gender.
    -   Result: **Demographic Risk Score (30% weight)**.

3.  **Symptom Processing** (`services/symptom_processor.py`):
    -   Inputs: Checked boxes + Free text.
    -   **NLP Cleaning**: Removes stop words, lemmatizes.
    -   **Fuzzy Matching**: Matches text against known model symptoms.
    -   **New Detection**: Identifies "unknown" symptoms for dynamic learning.

4.  **Disease Prediction** (`services/disease_prediction.py`):
    -   Converts matched symptoms to a vector (0s and 1s).
    -   Runs `symptoms_and_disease.pkl`.
    -   Result: **Predicted Disease, Probability, Severity**.
    -   *Logic*: **Symptom Risk Score (50% weight)** derived from probability + severity.

5.  **Final Risk Calculation**:
    -   `0.5 * SymptomRisk + 0.3 * DemographicRisk + 0.2 * ImpactScore`.

### C. Dynamic Learning (The "Smart" Part)
If an **unknown symptom** is detected during prediction:

1.  **Trigger**: `BackgroundTasks` in FastAPI calls `integrate_and_reload`.
2.  **Verification**: `services/web_scraper.py` checks search engines to confirm the symptom is real.
3.  **Data Scraping**: If valid, scrapes Healthline for associated diseases.
4.  **Database Update**:
    -   Adds new symptom column/data to **PostgreSQL `TrainingData` table**.
    -   Adds scraped precautions to `precautions` table.
5.  **Retraining** (`services/dynamic_learning.py`):
    -   Loads updated data from SQL.
    -   Retrains the Random Forest model.
    -   Overwrites `.pkl` files.
    -   Hot-reloads the model in memory.

---

## 3. Frontend Logic (`frontend/`)

-   **App**: `app.py` (Streamlit).
-   **Flow**:
    1.  Fetches known symptoms list from Backend (`GET /symptoms`).
    2.  User inputs data.
    3.  Sends `POST /predict`.
    4.  Displays Risk Meter, Disease Info, and Precautions.
-   **Configuration**: uses `API_URL` environment variable to look up the backend.

---

## 4. Orchestration & MLOps

### Current Active Orchestration
-   **Event-Driven**: The system uses **FastAPI BackgroundTasks** for immediate responsiveness.
-   *Why?* It allows the system to learn *instantly* when a new symptom is seen, without waiting for a scheduled job.

### Prefect (Optional/Legacy)
-   File: `prefect_flows.py`.
-   Status: Available for batch operations but not the primary driver. Can be used if you want to run nightly retraining jobs independent of user requests.

---

## 5. CI/CD Pipeline (`.github/workflows/deploy.yml`)

1.  **Trigger**: Push to `main`.
2.  **Job 1: Build & Test**:
    -   Installs backend dependencies.
    -   Runs `pytest backend/tests/`.
3.  **Job 2: Deploy**:
    -   Uses **Railway CLI**.
    -   Deploys **backend service** (Root context, `backend/Dockerfile`).
    -   Deploys **frontend service** (Root context, `frontend/Dockerfile`).

---

## 6. How to Verify It's Working?

1.  **Deployment**: Check Railway dashboard. Both services should represent "Active".
2.  **Smart Learning Test**:
    -   Go to Frontend.
    -   Enter a valid but "new" symptom (e.g., "polydipsia" if not trained).
    -   Submit.
    -   You should see a "🧠 Smart Learning Active" toast.
    -   Backend logs will show: `Scraping... Updating SQL... Retraining model...`.
