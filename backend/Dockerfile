FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PostageSQL and other libs
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory into /app/backend
# We assume the build context is the 'backend' folder itself if using 'COPY . /app/backend'
# OR if context is root, we copy backend.
# Here we assume Dockerfile is inside backend/ and we build from inside backend/ OR 
# we build from root and specify context. 
# Standard pattern: COPY . /app/backend
COPY . /app/backend

# Set working directory to /app so 'backend' is a module
WORKDIR /app

# Download NLTK data (optional here, mostly handled in code but good to pre-cache)
RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4

ENV PORT=8000
EXPOSE $PORT

# Command: Run the app as a module
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
