from sqlalchemy import text, inspect
import sys
import os

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(BASE_DIR))

from backend.database.database import engine

def check_and_migrate_tables():
    """
    Simple migration script to add missing columns if they don't exist.
    """
    try:
        insp = inspect(engine)
        if "feature_store" in insp.get_table_names():
            columns = [c['name'] for c in insp.get_columns("feature_store")]
            
            # Columns to check and add (Risk scores)
            new_cols = {
                "symptom_risk": "FLOAT",
                "demographic_risk": "FLOAT",
                "air_quality_risk": "FLOAT"
            }
            
            with engine.connect() as conn:
                for col, dtype in new_cols.items():
                    if col not in columns:
                        print(f"Migrating: Adding column '{col}' to feature_store...")
                        conn.execute(text(f"ALTER TABLE feature_store ADD COLUMN {col} {dtype}"))
                        conn.commit()
        else:
            print("Table feature_store does not exist yet. It will be created by models.Base.metadata.create_all.")
            
    except Exception as e:
        print(f"Migration Warning: {e}")

if __name__ == "__main__":
    check_and_migrate_tables()
