# Health System Guide

## Data Flow & Architecture

### 1. **User Request**
- User sends `POST /predict` to **`backend/main.py`** with `age`, `gender`, `city`, and `symptoms`.

### 2. **Processing (Backend Services)**
- **`health_features.py`**: Fetches weather/AQI from OpenWeatherMap -> uses `ml_models/health_impact_predictor.pkl` -> Returns `Health Impact Score`.
- **`demographic_risk.py`**: Calculates risk based on Age & Gender logic.
- **`symptom_processor.py`**:
  - Cleans user text (NLP).
  - Fuzzy matches against known symptoms in `disease_prediction.py` (which loads from SQL training data).
  - Identifies **NEW symptoms**.
- **`disease_prediction.py`**:
  - Converts matched symptoms to vehicle (0/1 array).
  - Uses `ml_models/symptoms_and_disease.pkl` to predict Disease.
  - Fetches precautions from SQL (`precautions` table).

### 3. **Dynamic Learning (Background Task)**
- If a **NEW symptom** is found:
  - **`dynamic_learning.py`** is triggered in background.
  - **Step A**: Verifies symptom validity via web search (`web_scraper.py`).
  - **Step B**: Scrapes associated diseases from Healthline (`web_scraper.py`).
  - **Step C**: Updates **SQL Database**:
    - **`TrainingData` table**: Updates JSON profile for relevant diseases with new symptom = 1.
    - **`Precaution` table**: Inserts any new precautions found.
  - **Step D**: **Retraining**:
    - Fetches all data from `TrainingData` table.
    - Retrains Random Forest model.
    - Saves new `.pkl` model to `ml_models/`.

### 4. **Storage (PostgreSQL)**
- **`feature_store`**: Logs every prediction request.
- **`training_data`**: The "source of truth" dataset for the ML model.
- **`precautions`**: Stores disease precautions.
- **`symptom_logs`**: Tracks new symptoms.

---

## File Structure Explanation

### `backend/`
- **`main.py`**: The API Server.
- **`database/`**:
  - `database.py`: Connects to `postgresql://`.
  - `models.py`: Defines SQL tables (`TrainingData`, `PredictionLog`, etc.).
  - `migration.py`: Helper to update tables if needed.
- **`services/`**:
  - `health_features.py`: Air quality & environmental risk.
  - `disease_prediction.py`: Loads ML model and predicts disease.
  - `dynamic_learning.py`: Handles **retraining** and **SQL updates**.
  - `web_scraper.py`: Helper to scrape Healthline/Google.
  - `symptom_processor.py`: Text cleaning & matching.
- **`ml_models/`**:
  - `symptoms_and_disease.pkl`: The trained disease model.
  - `label_encoder.pkl`: Encodes disease names to numbers.
  - `health_impact_predictor.pkl`: The trained AQI risk model.

## How to Train Manually?
If you want to force training from the database (e.g., after manually editing SQL):
```bash
python backend/services/dynamic_learning.py
```
This script acts as the **training trigger**. It loads data from SQL and overwrites the `.pkl` files.
