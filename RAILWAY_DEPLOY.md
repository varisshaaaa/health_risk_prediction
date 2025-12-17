# Railway Deployment Guide

This project is configured as a monorepo with two services: **Backend** (FastAPI) and **Frontend** (Streamlit).

## Auto-Deployment Setup

This project uses GitHub Actions for CI/CD. Every push to the `main` or `master` branch will:
1. Run automated tests
2. Deploy to Railway (if tests pass)

### Step 1: Get Your Railway Token

1. Go to [Railway Dashboard](https://railway.app/account/tokens)
2. Click **"Create Token"**
3. Name it something like `github-deploy`
4. Copy the token (you won't see it again!)

### Step 2: Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** > **Secrets and variables** > **Actions**
3. Click **"New repository secret"**
4. Name: `RAILWAY_TOKEN`
5. Value: Paste your Railway token
6. Click **"Add secret"**

### Step 3: Push to Deploy

Now whenever you push to `main` or `master`:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

The GitHub Actions workflow will automatically:
- Run tests
- Deploy to Railway if tests pass

---

## Manual Railway Setup

### 1. Project Structure on Railway

Create **two services** in your Railway project, pointing to the same GitHub repo.

### Service 1: Backend
- **Root Directory**: Leave empty (uses project root)
- **Dockerfile Path**: `backend/Dockerfile`
- **Build Command**: Uses Dockerfile
- **Variables**:
  - `PORT`: `8000` (Railway sets this automatically)
  - `DATABASE_URL`: Add PostgreSQL plugin, Railway auto-injects this
  - `WEATHERMAP_API_KEY`: Your OpenWeatherMap API Key

### Service 2: Frontend
- **Root Directory**: Leave empty
- **Dockerfile Path**: `frontend/Dockerfile`
- **Variables**:
  - `PORT`: `8501` (Railway sets this automatically)
  - `API_URL`: `https://<your-backend-service-url>.up.railway.app`

### 2. Add PostgreSQL Database

1. In Railway project, click **"+ New"** > **"Database"** > **"PostgreSQL"**
2. Railway automatically injects `DATABASE_URL` to your backend service
3. The backend will auto-create all tables on first startup

---

## Environment Variables Reference

### Backend Required
| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Auto-injected by Railway |
| `PORT` | Server port | 8000 (auto-set) |

### Backend Optional
| Variable | Description | Example |
|----------|-------------|---------|
| `WEATHERMAP_API_KEY` | OpenWeatherMap API key | `abc123...` |

### Frontend Required
| Variable | Description | Example |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `https://backend-xxx.up.railway.app` |
| `PORT` | Server port | 8501 (auto-set) |

---

## Troubleshooting

### Deployment Fails
1. Check GitHub Actions logs: Go to **Actions** tab in your repo
2. Ensure `RAILWAY_TOKEN` secret is set correctly
3. Check Railway dashboard for deployment logs

### Database Not Connecting
1. Ensure PostgreSQL addon is attached to backend service
2. Check that `DATABASE_URL` environment variable exists in Railway
3. Backend logs should show "Database connection successful"

### Frontend Can't Reach Backend
1. Ensure `API_URL` is set correctly in frontend service
2. The URL should be the public Railway URL of your backend
3. Check that backend is running and `/health` endpoint responds

---

## Security Note

**IMPORTANT**: Never commit tokens or API keys to your repository!

- Use GitHub Secrets for `RAILWAY_TOKEN`
- Use Railway environment variables for `DATABASE_URL` and `WEATHERMAP_API_KEY`
- The old hardcoded token in workflow files has been removed for security
