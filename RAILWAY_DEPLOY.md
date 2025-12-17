# Railway Deployment Guide

This project is configured as a monorepo with two services: **Backend** (FastAPI) and **Frontend** (Streamlit).

## 1. Project Structure on Railway

You should create **two services** in your Railway project, pointing to the same GitHub repo.

### Service 1: Backend
- **Root Directory**: `backend` (or leave empty if using specific Dockerfile path)
- **Dockerfile Path**: `backend/Dockerfile`
- **Variables**:
  - `PORT`: `8000`
  - `DATABASE_URL`: (Add PostgreSQL plugin variable here)
  - `WEATHERMAP_API_KEY`: (Your API Key)

### Service 2: Frontend
- **Root Directory**: `frontend`
- **Dockerfile Path**: `frontend/Dockerfile`
- **Variables**:
  - `PORT`: `8501`
  - `API_URL`: `https://<your-backend-service-url>.up.railway.app` (Streamlit needs this to talk to backend)

## 2. Updated Dockerfiles
I have updated both Dockerfiles to be standalone:
- **Backend**: Uses `uvicorn backend.main:app` and installs from `backend/requirements.txt`.
- **Frontend**: Uses `streamlit run frontend/app.py` and installs from `frontend/requirements.txt`.

## 3. CI/CD
The `.github/workflows/deploy.yml` is updated to trigger `railway up` for both services (`backend` and `frontend`). Ensure your Railway service names match these exactly.
