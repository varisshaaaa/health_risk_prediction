import nltk
from nltk.corpus import stopwords
from difflib import get_close_matches
import re

# Ensure NLTK data is downloaded (using quiet to avoid output clutter if already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_and_extract_smart(user_input, existing_symptoms):
    """
    Extract meaningful symptoms from free-text input.
    Preserves multi-word symptoms and avoids breaking them.
    Logic provided by user:
    1. Replace punctuation.
    2. Split by comma/and/or.
    3. Remove stopwords.
    4. Fuzzy match against existing symptoms.
    5. Return matched and new symptoms.
    """
    text = user_input.lower()
    
    # Replace punctuation with commas
    for p in ['.', ';', '!', '?']:
        text = text.replace(p, ',')
    
    # Split by comma or 'and'/'or' connectors
    phrases = re.split(r',|\band\b|\bor\b', text)
    phrases = [p.strip() for p in phrases if p.strip()]

    matched = set()
    new_symptoms = set()

    for phrase in phrases:
        # Remove leading/trailing stopwords
        words = [w for w in nltk.word_tokenize(phrase) if w.isalpha() and w not in stop_words]
        if not words:
            continue
        
        # Recombine words into one phrase
        cleaned_phrase = " ".join(words)

        # Fuzzy match full phrase first
        close_phrase = get_close_matches(cleaned_phrase, existing_symptoms, n=1, cutoff=0.7)
        if close_phrase:
            matched.add(close_phrase[0])
            continue
        
        # Optional: check concatenated words for merged symptoms (e.g. "head ache" -> "headache")
        concatenated = "".join(words) 
        close_concat = get_close_matches(concatenated, existing_symptoms, n=1, cutoff=0.7)
        if close_concat:
            matched.add(close_concat[0])
            continue
        
        # If still unmatched, consider it a new symptom
        # We store it for potential learning
        new_symptoms.add(cleaned_phrase)

    return list(matched), list(new_symptoms)
