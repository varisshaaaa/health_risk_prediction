# Frontend Dockerfile (Streamlit)
FROM python:3.9-slim

WORKDIR /app

# Copy frontend requirements
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend ./frontend

# Change workdir to frontend
WORKDIR /app/frontend

# Railway provides PORT dynamically
ENV PORT=8501
EXPOSE $PORT

# Streamlit config for production
CMD sh -c "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"
