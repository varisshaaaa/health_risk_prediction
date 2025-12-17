def add_symptom(db, symptom):
    db.execute(
        "INSERT INTO symptoms_and_disease (symptom) VALUES (%s) ON CONFLICT DO NOTHING",
        (symptom,)
    )
    db.commit()
