def precautions_exist(db, symptom):
    q = "SELECT 1 FROM precautions WHERE symptom=%s LIMIT 1"
    return db.execute(q, (symptom,)).fetchone()

def insert_precautions(db, symptom, disease, precautions):
    for p in precautions:
        db.execute("""
            INSERT INTO precautions (symptom, disease, precaution)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (symptom, disease, p))
    db.commit()
