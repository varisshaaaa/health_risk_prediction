FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code (current folder → /app)
COPY . .

# Make /app importable
ENV PYTHONPATH=/app

# Pre-download nltk data
RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4

ENV PORT=8000
EXPOSE 8000

# IMPORTANT: no "backend." prefix now
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

