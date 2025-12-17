"""
Centralized Logging Module
Provides consistent logging configuration across the application.
"""

import logging
import sys
import os
from datetime import datetime

# Log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log level from environment (default: INFO)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Creates and returns a configured logger.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
    
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level or LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    
    # File Handler (daily rotating)
    try:
        log_file = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        # File logging optional - don't fail if we can't write
        print(f"Warning: Could not set up file logging: {e}")
    
    return logger


def log_prediction(logger: logging.Logger, request_data: dict, result: dict):
    """
    Structured logging for prediction requests.
    
    Args:
        logger: Logger instance
        request_data: Input data from the request
        result: Prediction result
    """
    logger.info(
        f"PREDICTION | "
        f"City: {request_data.get('city', 'N/A')} | "
        f"Age: {request_data.get('age', 'N/A')} | "
        f"Disease: {result.get('predicted_disease', 'N/A')} | "
        f"Risk: {result.get('overall_health_risk', 0):.1f}%"
    )


def log_retraining(logger: logging.Logger, accuracy: float, n_samples: int, n_features: int):
    """
    Structured logging for model retraining events.
    
    Args:
        logger: Logger instance
        accuracy: Model accuracy
        n_samples: Number of training samples
        n_features: Number of features
    """
    logger.info(
        f"RETRAIN | "
        f"Accuracy: {accuracy:.4f} | "
        f"Samples: {n_samples} | "
        f"Features: {n_features}"
    )


def log_dynamic_learning(logger: logging.Logger, symptom: str, diseases: list, success: bool):
    """
    Structured logging for dynamic learning events.
    
    Args:
        logger: Logger instance
        symptom: New symptom detected
        diseases: Associated diseases found
        success: Whether integration was successful
    """
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"DYNAMIC_LEARNING | "
        f"Symptom: {symptom} | "
        f"Diseases: {len(diseases)} | "
        f"Status: {status}"
    )


# Application-wide logger instance
app_logger = get_logger('health_risk_api')

