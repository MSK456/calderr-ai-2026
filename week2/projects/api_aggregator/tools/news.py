import httpx
import xml.etree.ElementTree as ET

RSS_FEEDS = {
    "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "business":   "https://feeds.bbci.co.uk/news/business/rss.xml",
    "science":    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "world":      "https://feeds.bbci.co.uk/news/world/rss.xml",
}

def fetch_news(category: str = "technology", count: int = 5) -> dict:
    category = category.lower()
    feed_url = RSS_FEEDS.get(category, RSS_FEEDS["technology"])
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        items = channel.findall("item")[:count]

        articles = []
        for item in items:
            title_el = item.find("title")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")

            if title_el is not None:
                articles.append({
                    "title": title_el.text.strip() if title_el.text else "No title",
                    "description": (desc_el.text.strip()[:150] + "...") if desc_el is not None and desc_el.text else "",
                    "published": pub_el.text.strip() if pub_el is not None and pub_el.text else "Unknown",
                })

        return {"category": category, "source": "BBC News", "articles": articles}
    except Exception as e:
        return {"error": str(e)}