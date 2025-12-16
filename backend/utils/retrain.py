import sys
import os
import logging

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from disease_catboost_symptoms.utils.train_model import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_retraining_job():
    """
    Wrapper function to be called by the scheduler.
    """
    logger.info("Starting scheduled model retraining job...")
    try:
        train_model()
        logger.info("Retraining job completed successfully.")
    except Exception as e:
        logger.error(f"Retraining job failed: {e}")

if __name__ == "__main__":
    run_retraining_job()
