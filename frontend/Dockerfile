# Frontend Dockerfile (Streamlit)
FROM python:3.9-slim

WORKDIR /app

# Copy frontend requirements
# Assumes build context is REPO ROOT
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend ./frontend

# CHANGE WORKDIR TO FRONTEND SO 'app.py' IS LOCAL
WORKDIR /app/frontend

ENV PORT=8501
EXPOSE $PORT

# Command now runs 'app.py' directly because we are in /app/frontend
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
