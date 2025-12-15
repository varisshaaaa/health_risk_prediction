FROM python:3.9-slim

WORKDIR /app

# Copy requirements from root (assuming build context is root or we copy specifically)
# Best practice for monorepo in Docker compose: contextual build
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project or specific folders
# We need backend AND disease_catboost_symptoms (for the model/utils)
COPY backend ./backend
COPY disease_catboost_symptoms ./disease_catboost_symptoms
COPY symptoms_and_disease.csv .

COPY start_backend.py .

# Expose port
EXPOSE 8000

# Run startup script (Training + Server)
CMD ["python", "start_backend.py"]
