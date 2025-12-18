from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Railway provides DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback to local postgres if not set
    # Defaulting to widely used local dev credentials: postgres/postgres
    print("WARNING: DATABASE_URL not set. Defaulting to local postgres user 'postgres'.")
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/health_db"

# Fix for some postgres providers using postgres:// instead of postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

from sqlalchemy import text, inspect

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_and_migrate_tables():
    """
    Simple migration script to add missing columns if they don't exist.
    """
    try:
        insp = inspect(engine)
        if "feature_store" in insp.get_table_names():
            columns = [c['name'] for c in insp.get_columns("feature_store")]
            
            # Columns to check and add
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
    except Exception as e:
        print(f"Migration Warning: {e}")

