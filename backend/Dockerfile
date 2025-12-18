FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev gcc netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data during build (prevents timeout on startup)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"

# Copy backend source code
COPY backend ./backend

# Make backend importable
ENV PYTHONPATH=/app

# Railway provides PORT dynamically
ENV PORT=8000
EXPOSE $PORT

# Use shell form to expand $PORT variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
