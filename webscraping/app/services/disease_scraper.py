import requests, re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape_disease_and_precautions(symptom):
    url = f"https://www.healthline.com/symptom/{symptom.replace(' ', '-')}"
    r = requests.get(url, headers=HEADERS, timeout=10)

    result = {"diseases": [], "precautions": []}
    if r.status_code != 200:
        return result

    soup = BeautifulSoup(r.text, "html.parser")

    for h in soup.find_all("h2"):
        text = h.text.lower()

        if "cause" in text:
            ul = h.find_next("ul")
            if ul:
                result["diseases"] = [
                    re.sub(r"\[.*?\]", "", li.text.strip())
                    for li in ul.find_all("li")
                ]

        if any(w in text for w in ["treatment", "prevention", "management"]):
            ul = h.find_next("ul")
            if ul:
                result["precautions"] = [
                    re.sub(r"\[.*?\]", "", li.text.strip())
                    for li in ul.find_all("li")
                ]

    return result
