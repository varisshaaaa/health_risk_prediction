# Scrapy spider: run with `scrapy crawl precautions -a disease="influenza"`
import scrapy
from urllib.parse import urljoin
from ..items import PrecautionItem
from ..classifier import extract_precaution_sentences_from_html, MLClassifier, clean_text
import logging

# simple site configs for search URLs
SITES = {
    "who": {
        "name": "WHO",
        "search_url": "https://www.who.int/search?q={q}",
        "base": "https://www.who.int"
    },
    "cdc": {
        "name": "CDC",
        "search_url": "https://www.cdc.gov/search/results.html?query={q}",
        "base": "https://www.cdc.gov"
    },
    "mayoclinic": {
        "name": "Mayo Clinic",
        "search_url": "https://www.mayoclinic.org/search/search-results?q={q}",
        "base": "https://www.mayoclinic.org"
    }
}

class PrecautionsSpider(scrapy.Spider):
    name = "precautions"
    custom_settings = {
        # can be overridden in project settings
    }

    def __init__(self, disease=None, urls=None, model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disease = disease
        # input URLs string must be comma-separated if provided via -a
        if urls:
            if isinstance(urls, str):
                self.start_urls = [u.strip() for u in urls.split(",") if u.strip()]
            else:
                self.start_urls = urls
        else:
            self.start_urls = []
        # classifier
        self.classifier = MLClassifier(model_path=model_path)
        self.seen_page = set()

    def start_requests(self):
        # If explicit URLs given, request them first
        for url in getattr(self, "start_urls", []):
            yield scrapy.Request(url, callback=self.parse_page, meta={"source": "explicit", "disease": self.disease})
        # If disease provided, query configured sites
        if self.disease:
            for cfg in SITES.values():
                search_url = cfg["search_url"].format(q=scrapy.utils.request.quote(self.disease))
                yield scrapy.Request(search_url, callback=self.parse_search, meta={"site": cfg, "disease": self.disease})

    def parse_search(self, response):
        site = response.meta.get("site")
        base = site.get("base")
        qlow = (response.meta.get("disease") or "").lower()
        # find candidate anchors
        candidates = []
        for a in response.css("a[href]"):
            href = a.attrib.get("href")
            text = a.get().lower()
            if not href:
                continue
            # prefer internal links
            score = 0
            if qlow and qlow in text:
                score += 2
            if qlow and qlow in href.lower():
                score += 2
            if href.startswith("/") or (base and href.startswith(base)):
                score += 1
            if score > 0:
                full = urljoin(base, href) if base else response.urljoin(href)
                candidates.append((score, full))
        # fallback: any internal anchor
        if not candidates:
            for a in response.css("a[href]"):
                href = a.attrib.get("href")
                if href and (href.startswith("/") or (base and href.startswith(base))):
                    full = urljoin(base, href) if base else response.urljoin(href)
                    candidates.append((1, full))
        # sort and request top N unique pages
        candidates.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        for score, full in candidates[:5]:
            if full in seen:
                continue
            seen.add(full)
            yield scrapy.Request(full, callback=self.parse_page, meta={"source": site["name"], "disease": self.disease})

    def parse_page(self, response):
        url = response.url
        if url in self.seen_page:
            return
        self.seen_page.add(url)
        html = response.text
        sentences = extract_precaution_sentences_from_html(html)
        # If nothing found, attempt to heuristically find content containers and re-run sentence extraction
        if not sentences:
            # attempt to find main content divs
            main_selectors = ["article", "main", "div.content", "div#content", "section", "div.primary"]
            collected = []
            for sel in main_selectors:
                el = response.css(sel).get()
                if el:
                    collected.append(el)
            if collected:
                text = " ".join([clean_text(scrapy.selector.Selector(text=e).xpath("string(.)").get() or "") for e in collected])
                sentences = extract_precaution_sentences_from_html(text)
        # yield items
        for s in sentences:
            sev = self.classifier.predict(s)
            item = PrecautionItem()
            item["text"] = s
            item["severity"] = sev
            item["severity_label"] = self.classifier.severity_label(sev)
            src_label = response.meta.get("source") or "site"
            item["source"] = f"{src_label} @ {url}"
            item["url"] = url
            item["disease"] = response.meta.get("disease")
            yield item