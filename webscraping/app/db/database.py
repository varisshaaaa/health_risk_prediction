import os
import sqlite3
import psycopg2
from urllib.parse import urlparse

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./webscraping.db")

def get_db_connection():
    if DATABASE_URL.startswith("sqlite"):
        # For local dev / file based
        db_path = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    else:
        # PostgreSQL logic
        result = urlparse(DATABASE_URL)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port
        
        conn = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Create tables if not exist (Mirrors the schema expected by crud)
    # Note: Backend might manage the main schema, but this service needs to read/write its view.
    # For simplicity, we assume the backend manages the detailed schema, 
    # but we need 'symptoms_and_disease' and 'precautions' tables here if we use SQL.
    
    # However, if we are using CSVs as the source of truth for the model,
    # these tables might be temporary or a cache.
    # Given the user's files, we'll create them.
    
    if DATABASE_URL.startswith("sqlite"):
        c.execute('''CREATE TABLE IF NOT EXISTS symptoms_and_disease
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, symptom TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS precautions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, symptom TEXT, disease TEXT, precaution TEXT)''')
    else:
        # Postgres syntax
        c.execute('''CREATE TABLE IF NOT EXISTS symptoms_and_disease
                     (id SERIAL PRIMARY KEY, symptom TEXT UNIQUE)''')
        c.execute('''CREATE TABLE IF NOT EXISTS precautions
                     (id SERIAL PRIMARY KEY, symptom TEXT, disease TEXT, precaution TEXT)''')
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
