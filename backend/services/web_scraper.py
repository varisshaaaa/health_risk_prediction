import requests
import re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def verify_symptom(symptom):
    """
    Verifies if a symptom exists using a search engine query check.
    """
    try:
        url = f"https://duckduckgo.com/html/?q={symptom}+symptom+disease"
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return len(soup.find_all("a", class_="result__a")) > 0
    except Exception as e:
        print(f"Verification Check Failed: {e}")
        return False

def scrape_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
    except:
        pass
    return None

def parse_healthline_page(soup):
    result = {"diseases": [], "precautions": []}
    if not soup:
        return result

    for h in soup.find_all(["h2", "h3"]):
        text = h.text.lower()

        # Diseases extraction (Causes)
        if "cause" in text:
            ul = h.find_next("ul")
            if ul:
                result["diseases"].extend([
                    re.sub(r"\[.*?\]", "", li.text.strip())
                    for li in ul.find_all("li")
                ])

        # Precautions extraction (Treatment/Prevention)
        if any(w in text for w in ["treatment", "prevention", "management", "remedies", "lifestyle"]):
            ul = h.find_next("ul")
            if ul:
                result["precautions"].extend([
                    re.sub(r"\[.*?\]", "", li.text.strip())
                    for li in ul.find_all("li")
                ])
    
    return result

def scrape_disease_and_precautions(query):
    """
    Scrapes diseases and precautions for a given symptom query.
    """
    query_slug = query.replace(' ', '-').lower()
    
    # Strategy 1: Symptom URL
    url1 = f"https://www.healthline.com/symptom/{query_slug}"
    soup1 = scrape_url(url1)
    res1 = parse_healthline_page(soup1)
    
    # Strategy 2: Health/Condition URL
    if not res1["precautions"] and not res1["diseases"]:
        url2 = f"https://www.healthline.com/health/{query_slug}"
        soup2 = scrape_url(url2)
        res2 = parse_healthline_page(soup2)
        
        if res2["diseases"] or res2["precautions"]:
            return res2
            
    return res1
