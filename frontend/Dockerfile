# Frontend Dockerfile (Streamlit)
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy frontend requirements first (for better caching)
COPY frontend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Verify streamlit is installed and accessible
RUN which streamlit && streamlit --version

# Copy frontend code
COPY frontend /app/frontend

# Set working directory to frontend
WORKDIR /app/frontend

# Railway provides PORT dynamically
ENV PORT=8501
EXPOSE $PORT

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/_stcore/health')" || exit 1

# Use shell form to expand PORT variable
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
