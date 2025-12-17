import re
import pandas as pd
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data downloaded.")

class SymptomProcessor:
    def __init__(self, known_symptoms_list=None):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.known_symptoms = known_symptoms_list if known_symptoms_list else []

    def preprocess_text(self, text):
        """
        Cleans and normalizes text.
        """
        # Basic clean
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
        
        # Tokenize & Stopwords & Lemmatize
        tokens = text.split()
        cleaned = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return " ".join(cleaned)

    def extract_symptoms(self, checked_symptoms, free_text):
        """
        Combines checked symptoms and processed free text.
        Returns: matching_symptoms (list), new_candidate_symptoms (list)
        """
        final_symptoms = set(checked_symptoms)
        new_candidates = []

        if free_text:
            # Parse commas if user typed list
            parts = [p.strip() for p in free_text.split(',')]
            
            for part in parts:
                clean = self.preprocess_text(part)
                if not clean: continue
                
                # Fuzzy match against known
                if self.known_symptoms:
                    match = process.extractOne(clean, self.known_symptoms, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 85: # High threshold
                        final_symptoms.add(match[0])
                    else:
                        new_candidates.append(clean)
                else:
                    new_candidates.append(clean)

        return list(final_symptoms), new_candidates
