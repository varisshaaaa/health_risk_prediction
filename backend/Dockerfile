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

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
