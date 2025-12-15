import pandas as pd
from catboost import CatBoostClassifier
import os

def train_model():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_symptoms.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'catboost_model.cbm')

    # Create model directory if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('disease', axis=1)
    y = df['disease']

    # Train Model
    print("Training CatBoost Classifier...")
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False)
    model.fit(X, y)

    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    print("Model training completed successfully.")

if __name__ == "__main__":
    train_model()
