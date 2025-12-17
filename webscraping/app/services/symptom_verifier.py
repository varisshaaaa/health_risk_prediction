import requests
from bs4 import BeautifulSoup

def verify_symptom(symptom):
    url = f"https://duckduckgo.com/html/?q={symptom}+symptom+disease"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    return len(soup.find_all("a", class_="result__a")) > 0
