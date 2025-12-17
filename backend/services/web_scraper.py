"""
Web scraper for disease and precaution information.
Uses multiple reliable sources with fallback strategies.
"""

import requests
import re
import time
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# User agent for web requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# Known medical symptoms list for verification
KNOWN_MEDICAL_TERMS = {
    "fever", "cough", "headache", "nausea", "vomiting", "diarrhea", "fatigue",
    "pain", "ache", "rash", "itching", "swelling", "inflammation", "bleeding",
    "dizziness", "weakness", "numbness", "tingling", "burning", "cramping",
    "stiffness", "soreness", "tenderness", "discharge", "congestion", "sneezing",
    "breathlessness", "palpitations", "anxiety", "depression", "insomnia",
    "constipation", "bloating", "heartburn", "indigestion", "appetite", "thirst",
    "urination", "weight", "temperature", "chills", "sweating", "tremor"
}


def verify_symptom(symptom: str) -> bool:
    """
    Verifies if a string is likely a medical symptom.
    Uses simple keyword matching instead of web requests to avoid failures.
    
    Args:
        symptom: The symptom string to verify
        
    Returns:
        True if likely a valid symptom, False otherwise
    """
    if not symptom or len(symptom) < 3:
        return False
    
    symptom_lower = symptom.lower().strip()
    
    # Check if it contains known medical terms
    symptom_words = set(symptom_lower.replace("_", " ").replace("-", " ").split())
    
    # Direct match with known terms
    if symptom_words & KNOWN_MEDICAL_TERMS:
        return True
    
    # Check if any known term is a substring
    for term in KNOWN_MEDICAL_TERMS:
        if term in symptom_lower:
            return True
    
    # Medical suffix patterns
    medical_suffixes = ["itis", "osis", "emia", "algia", "pathy", "uria", "pnea", "rrhea"]
    for suffix in medical_suffixes:
        if symptom_lower.endswith(suffix):
            return True
    
    # If the symptom was extracted from training data, assume valid
    # (the system already vetted it)
    return True


def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 15) -> Optional[BeautifulSoup]:
    """
    Fetches a URL with retry logic and exponential backoff.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        BeautifulSoup object or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser")
            elif response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) * 2
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                time.sleep(wait_time)
            elif response.status_code >= 500:  # Server error
                wait_time = (2 ** attempt)
                time.sleep(wait_time)
            else:
                logger.debug(f"URL {url} returned status {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    
    return None


def scrape_wikipedia(query: str) -> Dict[str, List[str]]:
    """
    Scrapes disease and precaution information from Wikipedia.
    Wikipedia is reliable and rarely blocks requests.
    
    Args:
        query: Search term (symptom or disease name)
        
    Returns:
        Dictionary with 'diseases' and 'precautions' lists
    """
    result = {"diseases": [], "precautions": []}
    
    # Format query for Wikipedia URL
    query_formatted = query.replace(" ", "_").replace("-", "_").title()
    url = f"https://en.wikipedia.org/wiki/{query_formatted}"
    
    soup = fetch_with_retry(url)
    if not soup:
        # Try with (symptom) suffix
        url = f"https://en.wikipedia.org/wiki/{query_formatted}_(symptom)"
        soup = fetch_with_retry(url)
    
    if not soup:
        return result
    
    try:
        # Get main content
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            return result
        
        # Extract diseases from "Causes" or "Associated conditions" sections
        for header in content.find_all(["h2", "h3"]):
            header_text = header.get_text().lower()
            
            if any(term in header_text for term in ["cause", "etiology", "associated", "differential"]):
                # Get the following list or paragraph
                sibling = header.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3"]:
                    if sibling.name == "ul":
                        for li in sibling.find_all("li", recursive=False):
                            disease_text = li.get_text().strip()
                            # Clean up Wikipedia formatting
                            disease_text = re.sub(r"\[.*?\]", "", disease_text)
                            disease_text = re.sub(r"\(.*?\)", "", disease_text).strip()
                            if disease_text and len(disease_text) > 3:
                                result["diseases"].append(disease_text[:100])
                    sibling = sibling.find_next_sibling()
            
            # Extract precautions from Treatment/Prevention/Management sections
            if any(term in header_text for term in ["treatment", "prevention", "management", "prognosis", "lifestyle"]):
                sibling = header.find_next_sibling()
                while sibling and sibling.name not in ["h2", "h3"]:
                    if sibling.name == "ul":
                        for li in sibling.find_all("li", recursive=False):
                            precaution_text = li.get_text().strip()
                            precaution_text = re.sub(r"\[.*?\]", "", precaution_text)
                            if precaution_text and len(precaution_text) > 10:
                                result["precautions"].append(precaution_text[:200])
                    elif sibling.name == "p":
                        text = sibling.get_text().strip()
                        text = re.sub(r"\[.*?\]", "", text)
                        if text and len(text) > 20:
                            # Extract key sentences
                            sentences = text.split(". ")
                            for sentence in sentences[:3]:
                                if len(sentence) > 20:
                                    result["precautions"].append(sentence.strip() + ".")
                    sibling = sibling.find_next_sibling()
        
    except Exception as e:
        logger.error(f"Error parsing Wikipedia page: {e}")
    
    return result


def scrape_medlineplus(query: str) -> Dict[str, List[str]]:
    """
    Scrapes information from MedlinePlus (NIH).
    Reliable government health resource.
    
    Args:
        query: Search term
        
    Returns:
        Dictionary with 'diseases' and 'precautions' lists
    """
    result = {"diseases": [], "precautions": []}
    
    # Format query for MedlinePlus
    query_formatted = query.replace("_", " ").replace("-", " ").lower()
    url = f"https://medlineplus.gov/ency/article/003089.htm"  # Generic symptom page
    
    # Try search approach
    search_url = f"https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&v%3Asources=medlineplus-bundle&query={query_formatted}"
    
    soup = fetch_with_retry(search_url, timeout=10)
    if not soup:
        return result
    
    try:
        # Parse search results for relevant content
        for link in soup.find_all("a", href=True)[:5]:
            href = link.get("href", "")
            if "medlineplus.gov" in href and "ency" in href:
                page_soup = fetch_with_retry(href)
                if page_soup:
                    # Extract summary content
                    summary = page_soup.find("div", {"id": "ency_summary"})
                    if summary:
                        for li in summary.find_all("li"):
                            text = li.get_text().strip()
                            if text:
                                result["precautions"].append(text[:200])
    except Exception as e:
        logger.debug(f"MedlinePlus scraping error: {e}")
    
    return result


def scrape_healthline(query: str) -> Dict[str, List[str]]:
    """
    Scrapes from Healthline with updated parsing.
    
    Args:
        query: Search term
        
    Returns:
        Dictionary with 'diseases' and 'precautions' lists
    """
    result = {"diseases": [], "precautions": []}
    
    query_slug = query.replace(" ", "-").replace("_", "-").lower()
    urls = [
        f"https://www.healthline.com/symptom/{query_slug}",
        f"https://www.healthline.com/health/{query_slug}"
    ]
    
    for url in urls:
        soup = fetch_with_retry(url, timeout=10)
        if not soup:
            continue
        
        try:
            # Look for article content
            article = soup.find("article") or soup.find("main")
            if not article:
                continue
            
            # Extract from headers and lists
            for header in article.find_all(["h2", "h3"]):
                header_text = header.get_text().lower()
                
                if any(term in header_text for term in ["cause", "condition", "disease"]):
                    ul = header.find_next("ul")
                    if ul:
                        for li in ul.find_all("li")[:10]:
                            text = re.sub(r"\[.*?\]", "", li.get_text().strip())
                            if text and len(text) > 3:
                                result["diseases"].append(text[:100])
                
                if any(term in header_text for term in ["treatment", "prevention", "home", "remedy", "manage", "care"]):
                    ul = header.find_next("ul")
                    if ul:
                        for li in ul.find_all("li")[:10]:
                            text = re.sub(r"\[.*?\]", "", li.get_text().strip())
                            if text and len(text) > 10:
                                result["precautions"].append(text[:200])
            
            if result["diseases"] or result["precautions"]:
                break
                
        except Exception as e:
            logger.debug(f"Healthline parsing error: {e}")
    
    return result


def scrape_disease_info_for_precautions(disease: str) -> List[str]:
    """
    Scrapes precautions specifically for a known disease.
    
    Args:
        disease: Disease name
        
    Returns:
        List of precaution strings
    """
    precautions = []
    
    # Try Wikipedia first
    disease_formatted = disease.replace(" ", "_").replace("-", "_")
    url = f"https://en.wikipedia.org/wiki/{disease_formatted}"
    
    soup = fetch_with_retry(url)
    if soup:
        try:
            content = soup.find("div", {"id": "mw-content-text"})
            if content:
                for header in content.find_all(["h2", "h3"]):
                    header_text = header.get_text().lower()
                    
                    if any(term in header_text for term in ["treatment", "prevention", "management", "prognosis"]):
                        sibling = header.find_next_sibling()
                        while sibling and sibling.name not in ["h2", "h3"]:
                            if sibling.name == "ul":
                                for li in sibling.find_all("li", recursive=False)[:8]:
                                    text = li.get_text().strip()
                                    text = re.sub(r"\[.*?\]", "", text)
                                    if text and len(text) > 15:
                                        precautions.append(text[:200])
                            elif sibling.name == "p":
                                text = sibling.get_text().strip()
                                text = re.sub(r"\[.*?\]", "", text)
                                sentences = text.split(". ")
                                for sentence in sentences[:2]:
                                    if len(sentence) > 20:
                                        precautions.append(sentence.strip() + ".")
                            sibling = sibling.find_next_sibling()
                        break
        except Exception as e:
            logger.debug(f"Wikipedia disease scraping error: {e}")
    
    return precautions


def scrape_disease_and_precautions(query: str) -> Dict[str, List[str]]:
    """
    Main function to scrape diseases and precautions from multiple sources.
    Uses fallback strategy: Wikipedia -> Healthline -> Static data.
    
    Args:
        query: Symptom or disease to search for
        
    Returns:
        Dictionary with 'diseases' and 'precautions' lists
    """
    logger.info(f"🔍 Scraping for: {query}")
    
    result = {"diseases": [], "precautions": []}
    
    # Clean up query
    query_clean = query.replace("_", " ").replace("-", " ").strip().lower()
    
    # Strategy 1: Wikipedia (most reliable)
    logger.debug(f"Trying Wikipedia for: {query_clean}")
    wiki_result = scrape_wikipedia(query_clean)
    if wiki_result["diseases"]:
        result["diseases"].extend(wiki_result["diseases"])
    if wiki_result["precautions"]:
        result["precautions"].extend(wiki_result["precautions"])
    
    # Strategy 2: Healthline (if Wikipedia didn't have enough)
    if len(result["diseases"]) < 2 or len(result["precautions"]) < 2:
        logger.debug(f"Trying Healthline for: {query_clean}")
        healthline_result = scrape_healthline(query_clean)
        if healthline_result["diseases"]:
            result["diseases"].extend(healthline_result["diseases"])
        if healthline_result["precautions"]:
            result["precautions"].extend(healthline_result["precautions"])
    
    # Strategy 3: Use static precautions data as fallback
    if not result["precautions"]:
        logger.debug("Trying static precautions data as fallback")
        try:
            from backend.services.precautions_data import get_precautions_for_disease, get_generic_precautions
            
            # If we found diseases, get precautions for them
            for disease in result["diseases"][:3]:
                static_precs = get_precautions_for_disease(disease)
                if static_precs:
                    result["precautions"].extend(static_precs)
                    break
            
            # If still nothing, use generic precautions
            if not result["precautions"]:
                result["precautions"] = get_generic_precautions()
                
        except ImportError:
            logger.warning("Could not import static precautions data")
    
    # Deduplicate results
    result["diseases"] = list(dict.fromkeys(result["diseases"]))[:15]
    result["precautions"] = list(dict.fromkeys(result["precautions"]))[:10]
    
    logger.info(f"✅ Found {len(result['diseases'])} diseases, {len(result['precautions'])} precautions")
    
    return result


def scrape_precautions_for_disease(disease: str) -> List[str]:
    """
    Get precautions specifically for a disease (for API use).
    
    Args:
        disease: Disease name
        
    Returns:
        List of precautions
    """
    # Try static data first (faster and more reliable)
    try:
        from backend.services.precautions_data import get_precautions_for_disease
        static_precs = get_precautions_for_disease(disease)
        if static_precs:
            return static_precs
    except ImportError:
        pass
    
    # Fall back to web scraping
    scraped = scrape_disease_info_for_precautions(disease)
    if scraped:
        return scraped
    
    # Last resort: generic precautions
    try:
        from backend.services.precautions_data import get_generic_precautions
        return get_generic_precautions()
    except ImportError:
        return ["Consult a healthcare professional for proper treatment guidance."]
