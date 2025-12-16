# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install NLTK dependencies and others
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build to save runtime startup
RUN python -m nltk.downloader punkt stopwords

# Copy Dependencies
COPY backend ./backend
COPY webscraping ./webscraping
COPY symptoms_and_disease.csv .
COPY requirements.txt .

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose Port
EXPOSE 8000

# Start Command
# 1. We rely on the app's internal startup event to train the model if missing.
# 2. We just start uvicorn.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
