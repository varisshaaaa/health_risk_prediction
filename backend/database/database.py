from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
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

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    poolclass=QueuePool if "postgresql" in DATABASE_URL else None,
    pool_size=5 if "postgresql" in DATABASE_URL else None,
    max_overflow=10 if "postgresql" in DATABASE_URL else None,
    pool_pre_ping=True  # Verify connection before use
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

from sqlalchemy import text, inspect

# Connection status tracking
_database_available = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_database_connection() -> bool:
    """
    Tests the database connection and returns True if successful.
    Results are cached to avoid repeated connection attempts.
    """
    global _database_available
    
    if _database_available is not None:
        return _database_available
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            _database_available = True
            print("✅ Database connection successful")
            return True
    except Exception as e:
        _database_available = False
        print(f"❌ Database connection failed: {e}")
        return False


def reset_connection_status():
    """Resets the cached connection status to force a new test."""
    global _database_available
    _database_available = None


def init_database():
    """
    Initializes the database by creating all tables if they don't exist.
    Also tests the connection and runs migrations.
    """
    if not test_database_connection():
        print("⚠️ Database unavailable. Running in CSV-only mode.")
        return False
    
    try:
        # Import models to register them with Base
        from . import models  # noqa: F401
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
        
        # Run migrations
        check_and_migrate_tables()
        
        # Seed precautions if table is empty
        seed_precautions_from_csv()
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def seed_precautions_from_csv():
    """
    Seeds the precautions table from the CSV file if the table is empty.
    """
    import pandas as pd
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), 'Precautions.csv')
    
    if not os.path.exists(csv_path):
        print("⚠️ Precautions.csv not found. Skipping seeding.")
        return
    
    try:
        from .models import Precaution
        
        db = SessionLocal()
        
        # Check if table has any data
        count = db.query(Precaution).count()
        if count > 0:
            print(f"ℹ️ Precautions table already has {count} entries. Skipping seeding.")
            db.close()
            return
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print("⚠️ Precautions.csv is empty. Skipping seeding.")
            db.close()
            return
        
        # Check for required columns
        if 'Disease' not in df.columns or 'Precaution' not in df.columns:
            print("⚠️ Precautions.csv has incorrect format. Skipping seeding.")
            db.close()
            return
        
        # Add entries to database
        added = 0
        for _, row in df.iterrows():
            disease = row.get('Disease', '').strip()
            precaution = row.get('Precaution', '').strip()
            source = row.get('Source', 'csv_seed').strip()
            
            if disease and precaution and len(precaution) > 5:
                db.add(Precaution(
                    disease=disease,
                    content=precaution,
                    severity_level='BASIC',
                    source=source
                ))
                added += 1
        
        db.commit()
        print(f"✅ Seeded {added} precautions from CSV to database")
        db.close()
        
    except Exception as e:
        print(f"⚠️ Error seeding precautions: {e}")


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

