FROM python:3.9-slim

WORKDIR /app

COPY webscraping/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code. 
# Note: we copy from 'webscraping' folder in build context to 'app' in container
# If docker-compose context is root, and dockerfile is in webscraping/, usually we want:
# COPY . . (if context is webscraping dir)
# OR COPY webscraping/ . (if context is root)
# Based on docker-compose: context: .
# So we need to copy webscraping folder contents to /app/webscraping or /app in a way that imports work.
# The code does `from app.services...`
# So inside container we need `/app/app/services...` if we run from /app.
# webscraping folder structure:
# webscraping/app/
# So we copy webscraping/ to /app.
COPY webscraping/ .

# We need to run from /app so that "app" package is found.
# If we copy webscraping/app to /app/app, then `uvicorn app.main:app` works from /app.

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
