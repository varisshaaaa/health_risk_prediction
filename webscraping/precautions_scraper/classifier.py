"""
classifier.py

Provides:
- extract_precaution_sentences_from_html(html): heuristics to find prevention/precaution sections and split into sentences
- MLClassifier: loads model.pkl if present and predicts severity; otherwise falls back to rule-based classify_sentence()
"""

import re
from bs4 import BeautifulSoup
from typing import List
import os

# try to import joblib/sklearn; not required if you only want rule-based
try:
    from joblib import load as joblib_load
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Keywords for finding prevention sections
HEADING_KEYWORDS = [
    "precaution", "precautions", "prevention", "prevent", "how to prevent",
    "protect yourself", "preventive measures", "prevention and control",
    "avoid", "how to avoid", "protecting", "prevention & control", "prevention tips"
]
KEYWORDS_RE = re.compile("|".join(re.escape(k) for k in HEADING_KEYWORDS), re.I)

# classification keywords for rule-based fallback: severity 1..4
CLASS_KEYWORDS = {
    4: ["isolate", "quarantine", "seek medical", "hospital", "urgent", "call your doctor", "emergency", "immediately seek care"],
    3: ["vaccin", "immuniz", "immunization", "get vaccinated", "mask", "wear a mask", "ppe", "respirator"],
    2: ["avoid contact", "avoid crowds", "stay home", "social distancing", "avoid travel", "avoid close contact", "limit visitors"],
    1: ["wash hands", "hand hygiene", "soap", "clean", "cover cough", "cover your cough", "hand sanitizer", "stay home if sick", "rest", "drink fluids"]
}
CLASS_KEYWORDS_RE = {sev: re.compile("|".join(re.escape(k) for k in keys), re.I) for sev, keys in CLASS_KEYWORDS.items()}

SEVERITY_LABEL = {1: "BASIC", 2: "MODERATE", 3: "IMPORTANT", 4: "URGENT"}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")  # default location (project root's model.pkl)


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def split_into_sentences(text: str) -> List[str]:
    # naive sentence splitter: split on . ? ! ; or line breaks, keep chunks > 20 chars
    parts = re.split(r'(?<=[\.\?\!;])\s+|\n+', text)
    cleaned = [clean_text(p) for p in parts if p and len(clean_text(p)) > 20]
    return cleaned


def extract_precaution_sentences_from_html(html: str) -> List[str]:
    """
    Heuristics:
     - find headings that match prevention keywords and gather sibling paragraphs/lists
     - find elements with id/class that contain keyword
     - fallback: paragraphs that contain keyword
    Returns a deduplicated list of sentences (strings).
    """
    soup = BeautifulSoup(html, "lxml")
    results = []

    # 1) headings h1..h6
    for heading in soup.find_all(re.compile(r'^h[1-6]$', re.I)):
        htext = heading.get_text(" ", strip=True)
        if KEYWORDS_RE.search(htext):
            collected = []
            for sib in heading.next_siblings:
                if getattr(sib, "name", None) and re.match(r"^h[1-6]$", sib.name, re.I):
                    break
                if getattr(sib, "name", None) in ("p", "div", "ul", "ol", "li"):
                    t = sib.get_text(" ", strip=True)
                    if t:
                        collected.append(clean_text(t))
            if collected:
                for s in split_into_sentences(" ".join(collected)):
                    results.append(s)

    # 2) elements with id/class containing keyword
    for el in soup.find_all(True, attrs={"id": True}):
        if KEYWORDS_RE.search(el.get("id", "")):
            text = el.get_text(" ", strip=True)
            for s in split_into_sentences(text):
                results.append(s)
    for el in soup.find_all(True, attrs={"class": True}):
        classes = " ".join(el.get("class"))
        if KEYWORDS_RE.search(classes):
            text = el.get_text(" ", strip=True)
            for s in split_into_sentences(text):
                results.append(s)

    # 3) paragraphs containing keyword
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if KEYWORDS_RE.search(txt):
            for s in split_into_sentences(txt):
                results.append(s)

    # fallback: search within lists as well if nothing found
    if not results:
        for li in soup.find_all("li"):
            txt = li.get_text(" ", strip=True)
            if KEYWORDS_RE.search(txt) or len(txt) > 40:
                for s in split_into_sentences(txt):
                    results.append(s)

    # deduplicate preserving order
    seen = set()
    dedup = []
    for r in results:
        if r not in seen:
            dedup.append(r)
            seen.add(r)
    return dedup


class MLClassifier:
    """
    Loads a model.pkl if available and exposes predict(sentence) -> severity int.
    If model not available, uses rule-based classifier.
    model.pkl should be a sklearn pipeline that accepts list[str] and outputs integer labels 1..4.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            path = model_path
        else:
            # look for model in project root
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model.pkl"))
        if SKLEARN_AVAILABLE and os.path.exists(path):
            try:
                self.model = joblib_load(path)
            except Exception:
                self.model = None

    def predict(self, sentence: str) -> int:
        sentence = clean_text(sentence)
        if self.model:
            try:
                pred = self.model.predict([sentence])[0]
                # ensure int in 1..4
                try:
                    return int(pred)
                except Exception:
                    return self.rule_based(sentence)
            except Exception:
                return self.rule_based(sentence)
        else:
            return self.rule_based(sentence)

    @staticmethod
    def rule_based(sentence: str) -> int:
        # highest severity match wins
        for sev in sorted(CLASS_KEYWORDS_RE.keys(), reverse=True):
            if CLASS_KEYWORDS_RE[sev].search(sentence):
                return sev
        return 1

    @staticmethod
    def severity_label(severity: int) -> str:
        return SEVERITY_LABEL.get(severity, "BASIC")