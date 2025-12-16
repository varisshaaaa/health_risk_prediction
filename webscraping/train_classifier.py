# train_classifier.py
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def prepare_training_data():
    """Return list of (sentence, severity) pairs."""
    # Expanded training dataset
    return [
        ("Wash your hands regularly with soap and water.", 1),
        ("Cover your mouth when you cough or sneeze.", 1),
        ("Get vaccinated annually.", 2),
        ("Avoid close contact with sick people.", 2),
        ("Stay home when you are sick.", 3),
        ("Wear a mask in crowded places.", 3),
        ("Seek immediate medical attention if you have difficulty breathing.", 4),
        ("Go to the emergency room if you experience chest pain.", 4),
        ("Clean and disinfect frequently touched surfaces.", 1),
        ("Maintain a healthy diet and exercise regularly.", 1),
        ("Use hand sanitizer when soap is not available.", 1),
        ("Practice social distancing.", 2),
        ("Monitor your symptoms daily.", 2),
        ("Take prescribed medications as directed.", 3),
        ("Isolate yourself from others if you test positive.", 3),
        ("Call emergency services if you have severe symptoms.", 4),
        ("Avoid touching your face with unwashed hands.", 1),
        ("Get adequate sleep to boost your immune system.", 1),
        ("Avoid sharing personal items like towels or utensils.", 2),
        ("Ensure proper ventilation in indoor spaces.", 2),
        ("Use a humidifier to ease breathing.", 3),
        ("Take over-the-counter fever reducers if needed.", 3),
        ("Seek urgent care for high fever that doesn't break.", 4),
        ("Go to the hospital if you experience confusion or dizziness.", 4),
        ("Drink plenty of fluids.", 1),
        ("Eat fruits and vegetables rich in vitamins.", 1),
        ("Avoid crowded events and gatherings.", 2),
        ("Work from home if possible.", 2),
        ("Check your temperature twice a day.", 3),
        ("Keep a symptom diary.", 3),
        ("Immediately report to health authorities if diagnosed.", 4),
        ("Follow quarantine protocols strictly.", 4),
        ("Wash fruits and vegetables before eating.", 1),
        ("Cook meat thoroughly.", 1),
        ("Avoid contact with animals that may carry disease.", 2),
        ("Use insect repellent in affected areas.", 2),
        ("Take antiviral medication as prescribed.", 3),
        ("Use a separate bathroom if possible.", 3),
        ("Call your doctor immediately if symptoms worsen.", 4),
        ("Go to the ER for persistent vomiting or diarrhea.", 4),
        ("Practice good respiratory hygiene.", 1),
        ("Dispose of tissues properly after use.", 1),
        ("Wear gloves when caring for sick individuals.", 2),
        ("Sanitize your phone and other devices regularly.", 2),
        ("Keep emergency contacts handy.", 3),
        ("Have a supply of essential medications.", 3),
        ("Evacuate immediately if advised by authorities.", 4),
        ("Follow all public health advisories without delay.", 4),
        ("Wear protective clothing when handling contaminated items.", 2),
        ("Boil drinking water if contamination is suspected.", 3),
        ("Report any suspected cases to local health departments.", 3),
        ("Avoid travel to outbreak areas.", 2),
        ("Complete the full course of antibiotics even if you feel better.", 3),
        ("Use a tissue or your elbow when coughing or sneezing.", 1),
        ("Keep a distance of at least 6 feet from others.", 2),
        ("Wash your hands for at least 20 seconds.", 1),
        ("Avoid shaking hands or hugging.", 2),
        ("Clean your hands before touching your eyes, nose, or mouth.", 1),
        ("Stay in a separate room from other household members if sick.", 3),
        ("Use a separate bathroom if available.", 3),
        ("Avoid using public transportation when sick.", 3),
        ("Seek medical care immediately if you have trouble breathing.", 4),
        ("Call ahead before visiting your doctor or hospital.", 3),
        ("Wear a face covering when around others.", 2),
        ("Cover all surfaces of your hands and rub them together until dry.", 1),
        ("Avoid contact with people who are sick.", 2),
        ("Stay home except to get medical care.", 3),
        ("Monitor your health for symptoms.", 2),
        ("Take your temperature regularly.", 2),
        ("Watch for emergency warning signs.", 3),
        ("Get tested if you have symptoms.", 3),
        ("Follow guidance from your local health department.", 2),
        ("Clean your hands often.", 1),
        ("Avoid crowded places.", 2),
        ("Limit close contact with others.", 2),
        ("Stay at least 6 feet away from others.", 2),
        ("Wear a cloth face covering in public.", 2),
        ("Cover coughs and sneezes.", 1),
        ("Clean and disinfect daily.", 1),
        ("Stay home if you are sick.", 3),
        ("Get medical care immediately if needed.", 4),
    ]

class PrecautionsClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial'))
        ])
    
    def train(self, texts, labels):
        """Train the classifier on given texts and labels."""
        self.pipeline.fit(texts, labels)
        
        # Check accuracy on training data
        predictions = self.pipeline.predict(texts)
        accuracy = np.mean(predictions == labels)
        print(f"Training accuracy: {accuracy:.2%}")
        
        # Show a few examples
        print("\nSample predictions:")
        for i in range(min(5, len(texts))):
            pred = self.pipeline.predict([texts[i]])[0]
            print(f"  Text: {texts[i][:50]}...")
            print(f"  True: {labels[i]}, Predicted: {pred}")
            print()
    
    def predict(self, texts):
        """Predict severity for given texts."""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            raise ValueError("Classifier not trained. Call train() first.")
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        """Get probability estimates for predictions."""
        return self.pipeline.predict_proba(texts)

def train_and_save():
    """Train classifier and save to model.pkl"""
    print("Preparing training data...")
    training_data = prepare_training_data()
    
    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]
    
    print(f"Training on {len(training_data)} labeled sentences...")
    
    # Create and train classifier
    classifier = PrecautionsClassifier()
    classifier.train(texts, labels)
    
    # Save the trained classifier
    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"\nModel saved as 'model.pkl'")
    print(f"Training completed successfully!")
    
    # Optional: Test with some new sentences
    test_sentences = [
        "Wash hands frequently",
        "Go to hospital immediately",
        "Take your medicine",
        "Emergency room visit required"
    ]
    
    print("\nTesting with new sentences:")
    for sentence in test_sentences:
        prediction = classifier.predict([sentence])[0]
        severity_labels = {1: "BASIC", 2: "MODERATE", 3: "IMPORTANT", 4: "URGENT"}
        print(f"  '{sentence}' → Severity {prediction} ({severity_labels[prediction]})")

if __name__ == "__main__":
    train_and_save()