# Scrapy settings for precautions_scraper project (minimal, tuned for politeness & concurrency)

BOT_NAME = "precautions_scraper"

SPIDER_MODULES = ["precautions_scraper.spiders"]
NEWSPIDER_MODULE = "precautions_scraper.spiders"

# Politeness / throttling
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 8
DOWNLOAD_DELAY = 1.0
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0

# Retries
RETRY_ENABLED = True
RETRY_TIMES = 3

# User-Agent
USER_AGENT = "precautions-scraper/1.0 (+https://yourdomain.example; contact: you@example.com)"

# Pipelines (enable dedup)
ITEM_PIPELINES = {
    "precautions_scraper.pipelines.DedupPipeline": 300,
}

# Feed export: you can override via CLI (-o file)
FEED_EXPORT_ENCODING = "utf-8"