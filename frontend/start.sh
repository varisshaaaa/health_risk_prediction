#!/bin/bash
# Start script for Streamlit frontend on Railway

# Set default PORT if not provided
export PORT=${PORT:-8501}

# Run Streamlit
exec streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false


