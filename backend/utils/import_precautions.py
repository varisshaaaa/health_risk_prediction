"""
Utility script to import precautions from JSON files into the database.
"""

import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.database import SessionLocal
from backend.database.models import Precaution

try:
    from backend.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def import_precautions(json_path):
    """
    Imports precautions from a JSON Lines file or JSON array file into the database.
    
    Args:
        json_path: Path to JSON file containing precautions
    """
    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return

    session = SessionLocal()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error("JSON content is not a list.")
                    return
            except json.JSONDecodeError:
                # Try reading as JSON Lines
                f.seek(0)
                data = [json.loads(line) for line in f]

        count = 0
        for item in data:
            # Check for duplicates based on content and disease
            exists = session.query(Precaution).filter_by(
                disease=item.get('disease'),
                content=item.get('text')
            ).first()
            
            if not exists:
                p = Precaution(
                    disease=item.get('disease'),
                    content=item.get('text'),
                    severity_level=item.get('severity_label', 'BASIC'),
                    source=item.get('source')
                )
                session.add(p)
                count += 1
        
        session.commit()
        logger.info(f"Successfully imported {count} new precautions.")
        
    except Exception as e:
        logger.error(f"Error importing precautions: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    # Default path assumption
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_path = os.path.join(base_dir, 'webscraping', 'precautions.json')
    import_precautions(default_path)
