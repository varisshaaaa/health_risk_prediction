from app.models.ml_model import retrain_model

def trigger_retrain(new_symptoms):
    if new_symptoms:
        retrain_model()
