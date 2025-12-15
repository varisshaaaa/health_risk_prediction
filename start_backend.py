import os
import subprocess
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from disease_catboost_symptoms.utils.train_model import train_model
except ImportError as e:
    print(f"Error importing train_model: {e}")
    sys.exit(1)

def main():
    # 1. Train the model
    print("--- STARTING MODEL TRAINING ---")
    try:
        train_model()
        print("--- MODEL TRAINING COMPLETE ---")
    except Exception as e:
        print(f"--- TRAINING FAILED: {e} ---")
        # Decide if we want to exit or try starting app anyway. 
        # Usually better to fail if model is critical.
        sys.exit(1)

    # 2. Start the Backend
    print("--- STARTING UVICORN SERVER ---")
    port = os.getenv("PORT", "8000")
    
    # Using sys.executable to ensure we use the same python interpreter
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", port]
    
    # subprocess.run will wait for the process to complete (which is what we want for a server)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
