import requests, re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}

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
            # Sometimes parsing <p> tags is needed if no <ul>, but keeping it simple for now
    
    return result

def scrape_disease_and_precautions(query):
    query_slug = query.replace(' ', '-').lower()
    
    # Try strategy 1: Symptom URL
    url1 = f"https://www.healthline.com/symptom/{query_slug}"
    soup1 = scrape_url(url1)
    res1 = parse_healthline_page(soup1)
    
    # If we got nothing useful, try strategy 2: Condition/Health URL
    if not res1["precautions"] and not res1["diseases"]:
        url2 = f"https://www.healthline.com/health/{query_slug}"
        soup2 = scrape_url(url2)
        res2 = parse_healthline_page(soup2)
        
        if res2["diseases"] or res2["precautions"]:
            return res2
            
    return res1
