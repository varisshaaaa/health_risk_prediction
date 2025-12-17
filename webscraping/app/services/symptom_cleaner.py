from rapidfuzz import process, fuzz

def parse_free_text(text):
    return [t.strip().lower() for t in text.split(",") if t.strip()]

def match_symptom(symptom, known, threshold=80):
    match = process.extractOne(symptom, known, scorer=fuzz.token_sort_ratio)
    return match[0] if match and match[1] >= threshold else None

def extract_symptoms(checked, free_text, known):
    final = set(checked)
    new = []

    for s in parse_free_text(free_text):
        m = match_symptom(s, known)
        if m:
            final.add(m)
        else:
            new.append(s)

    return list(final), new
