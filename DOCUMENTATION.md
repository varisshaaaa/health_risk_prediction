# Health Risk Assessment System - Project Documentation

## 1. System Architecture & Workflow

The Health Risk Assessment System is a multi-layered application designed to provide personalized health guidance by integrating user demographics, real-time environmental data, and symptoms-based disease prediction.

### High-Level Flow
1.  **User Input**: Users provide Age, Gender, City, and Symptoms via the Streamlit Frontend.
2.  **frontend logic**: The frontend captures inputs and visualizes real-time feedback.
3.  **FastAPI Backend**:
    *   **Orchestrator**: Validates inputs and triggers parallel risk assessments.
    *   **In-Memory/Feature Store**: Logs inputs for future retraining.
4.  **Risk Assessment Engine**:
    *   **Environmental Module**: Fetches live AQI and uses a dedicated **ML Health Impact Predictor** (Random Forest/XGBoost) to calculate specific health risks beyond simple heuristics.
    *   **Demographic Module**: Calculates baseline risk based on Age and Gender.
    *   **ML Prediction Module**: Uses a trained **CatBoost Classifier** to predict disease probability based on symptoms.
5.  **Weighted Aggregation**: Combines the three risk signals into a single "Total Health Risk Score".
6.  **Advisory Generation**:
    *   **Precautions DB**: Queries a PostgreSQL database for disease-specific precautions (scraped from WHO, CDC).
    *   **Filtering**: Selects precautions based on the calculated Risk Level (Low/Moderate/High/Critical).
7.  **Output**: Returns a structured JSON response with the Predicted Disease, Risk Score, and Natural Language Guidance.

## 2. Machine Learning vs. Rule-Based Logic

The system employs a **Hybrid Intelligence** approach:
*   **Machine Learning (CatBoost)**: Used for the complex task of pattern matching symptoms to diseases. This allows the system to handle non-linear relationships and rare symptom combinations that rule-based systems might miss.
*   **Rule-Based Logic (Heuristics)**: Used for:
    *   **Risk Weighting**: Strict business rules (50/30/20) ensure that symptoms are the primary driver, while environmental context acts as a modifier.
    *   **Precautions Filtering**: Deterministic logic guarantees that high-risk users receive "Urgent" advice, eliminating the randomness of generative AI for safety-critical instructions.

## 3. Risk Weighting Rationale

The Total Health Risk Score (0-100) is calculated using the following weighted formula:

| Factor | Weight | Rationale |
| :--- | :--- | :--- |
| **Symptoms-Based Prediction** | **50%** | The most direct indicator of immediate illness. |
| **Demographics (Age/Gender)** | **30%** | Baseline biological vulnerability (e.g., infants and elderly are at higher risk). |
| **Environmental Impact (Air Quality)** | **20%** | Contextual exacerbating factor (e.g., pollution worsens respiratory conditions). |

**Formula:**
`Total Score = (SymptomConfidence * 0.5) + (DemographicRisk * 0.3) + (EnvironmentalRisk * 0.2)`

## 4. Data Sources & Web Scraping

To ensure the advice is medically sound, the system utilizes a **Offline Web Scraping** pipeline.

*   **Trusted Sources**: WHO, CDC, and Mayo Clinic.
*   **Offline Approach**: Scraping is performed as a scheduled background task (Offline) rather than live.
    *   **Safety**: Prevents live scraping errors (broken links, timeouts) from crashing the user experience.
    *   **Consistency**: Allows for human-in-the-loop verification of scraped content before it enters the database.
    *   **Performance**: Database lookups (SQL) are milliseconds, whereas live scraping takes seconds.

## 5. Ethical Considerations & Disclaimers

*   **Non-Diagnostic**: The system explicitly states it provides "Health Guidance" and not a medical diagnosis.
*   **Transparency**: Users are shown exactly why their risk score is high (e.g., "Air Quality is contributing 15% to your risk").
*   **Data Privacy**: Prediction logs are stored for model monitoring (Drift Detection) but are anonymized where possible.

## 6. Deployment & Operations

*   **Database**: PostgreSQL (Railway Managed) for persistent storage of Precautions and Logs.
*   **Continuous Learning**: An `APScheduler` task runs every 5 hours to retrain the CatBoost model on new symptom logs, ensuring the system adapts to emerging trends.
*   **Containerization**: Docker and Docker Compose ensure consistent environments across Development and Production.
