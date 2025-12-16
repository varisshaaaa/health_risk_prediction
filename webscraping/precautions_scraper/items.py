from scrapy import Item, Field

class PrecautionItem(Item):
    text = Field()
    severity = Field()        # 1..4 (1 basic .. 4 urgent)
    severity_label = Field()  # "BASIC"/"MODERATE"/"IMPORTANT"/"URGENT"
    source = Field()          # e.g. "WHO @ https://..."
    url = Field()
    disease = Field()