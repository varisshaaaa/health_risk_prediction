# Frontend Dockerfile (Streamlit)
FROM python:3.9-slim

WORKDIR /app

# Copy frontend requirements
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend ./frontend

ENV PORT=8501
EXPOSE $PORT

# Run Streamlit
# Note: --server.port must match the PORT env var for Railway
CMD ["sh", "-c", "streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0"]
