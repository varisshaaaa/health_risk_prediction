# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy Source Code
COPY frontend ./frontend

# Build (adjust if your build script outputs to a different folder than 'build' or 'dist')
# Streamlit doesn't need a build step like React. 
# WAIT - The user has a Streamlit frontend now (app.py), NOT a React frontend.
# The user's prompt mentioned "React/Vite" in the template example, but the actual file I viewed earlier was `frontend/app.py` (Streamlit).
# I must create a Dockerfile for STREAMLIT, not Node.js.

# ... Checking back logs ...
# Step 82 verified frontend/app.py is Streamlit.
# Step 81 verified frontend/package.json exists and lists React/MUI.
# This means the user HAS a React project but possibly abandoned it for Streamlit?
# OR they have both.
# The user said "I have already deployed backend and frontend".
# And "my frontend it wil show the user all the symptoms".
# The user interacted with the Streamlit app code in Step 92.
# So I should deploy the Streamlit app as the primary frontend for this "AI Health" project.

# If the user wants the React app, I will provide that too, but the Streamlit one matches the Python logic I wrote.
# Let's provide a Python-based Frontend Dockerfile for Streamlit.

FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .
# Add streamlit if not in requirements (it is in my updated one)
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend
COPY backend ./backend 
# Streamlit might need backend utils if importing directly, but usually it talks via HTTP.
# In `frontend/app.py`, it uses `requests.get(API_URL)`. So it's decoupled.

# Expose Streamlit Port
EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
