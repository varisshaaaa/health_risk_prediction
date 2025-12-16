from scrapy.exceptions import DropItem
import logging

class DedupPipeline:
    """
    Drop duplicate precaution sentences (by exact text). Keeps the first occurrence.
    """
    def __init__(self):
        self.seen = set()

    def process_item(self, item, spider):
        txt = item.get("text", "").strip()
        if not txt:
            raise DropItem("Empty text")
        if txt in self.seen:
            raise DropItem("Duplicate precaution")
        self.seen.add(txt)
        return item