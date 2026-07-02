"""
Day 4: External APIs as Tools — httpx, real APIs, error handling
APIs used (all FREE, no key needed):
  - Open-Meteo: weather
  - open.er-api.com: currency
  - BBC RSS: news headlines
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional
from dotenv import load_dotenv
import httpx
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────
# CITY COORDINATES (for weather)
# ─────────────────────────────────────────────────────────

CITY_COORDS = {
    "islamabad": (33.7215, 73.0433),
    "karachi":   (24.8607, 67.0011),
    "lahore":    (31.5204, 74.3587),
    "london":    (51.5074, -0.1278),
    "dubai":     (25.2048, 55.2708),
    "new york":  (40.7128, -74.0060),
    "tokyo":     (35.6762, 139.6503),
    "paris":     (48.8566, 2.3522),
}


# ─────────────────────────────────────────────────────────
# REAL API TOOLS
# ─────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get real current weather for any major city.
    Uses Open-Meteo API (free, no key required).
    Use this when user asks about weather, temperature, or climate in a city.

    Args:
        city: city name like 'Islamabad', 'Karachi', 'London', 'Dubai'
    """
    city_lower = city.lower().strip()
    coords = CITY_COORDS.get(city_lower)

    if not coords:
        available = ", ".join(c.title() for c in CITY_COORDS.keys())
        return f"City '{city}' not in database. Available cities: {available}"

    lat, lon = coords
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
                    "timezone": "auto"
                }
            )
            response.raise_for_status()
            data = response.json()

        current = data["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        precip = current["precipitation"]

        # Weather code mapping
        code = current["weather_code"]
        conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Light snow", 80: "Rain showers", 95: "Thunderstorm"
        }
        condition = conditions.get(code, f"Weather code {code}")

        return (f"Weather in {city.title()}: {condition}, "
                f"{temp}°C, Humidity {humidity}%, "
                f"Wind {wind} km/h, Precipitation {precip}mm")

    except httpx.TimeoutException:
        return f"Weather API timeout for {city}. Try again."
    except httpx.HTTPStatusError as e:
        return f"Weather API error: {e.response.status_code}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def get_exchange_rate(query: str) -> str:
    """Get real-time currency exchange rates.
    Uses open.er-api.com (free, no key required).
    Use this when user asks about currency conversion or exchange rates.

    Args:
        query: currency query like 'USD to PKR', 'EUR to GBP', '100 USD to PKR'
    """
    query_upper = query.upper()

    # Extract amount if present
    amount_match = re.search(r'(\d+(?:\.\d+)?)', query)
    amount = float(amount_match.group(1)) if amount_match else 1.0

    # Extract currency codes
    currencies = re.findall(r'\b([A-Z]{3})\b', query_upper)
    if len(currencies) < 2:
        return "Please specify currencies like: '100 USD to PKR' or 'EUR to GBP'"

    from_currency, to_currency = currencies[0], currencies[1]

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"https://open.er-api.com/v6/latest/{from_currency}")
            response.raise_for_status()
            data = response.json()

        if data.get("result") != "success":
            return f"Could not fetch rates for {from_currency}"

        rates = data["rates"]
        if to_currency not in rates:
            return f"Currency {to_currency} not found. Try: USD, EUR, GBP, PKR, AED, SAR"

        rate = rates[to_currency]
        converted = amount * rate

        return (f"{amount} {from_currency} = {converted:.2f} {to_currency} "
                f"(Rate: 1 {from_currency} = {rate:.4f} {to_currency})")

    except httpx.TimeoutException:
        return "Currency API timeout. Try again."
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"


@tool
def get_tech_news(topic: str = "technology") -> str:
    """Get latest technology news headlines from BBC RSS feed.
    Use this when user asks about recent news, headlines, or current events.

    Args:
        topic: news category - 'technology', 'business', or 'science'
    """
    feeds = {
        "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "business":   "https://feeds.bbci.co.uk/news/business/rss.xml",
        "science":    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    }

    topic_lower = topic.lower()
    feed_url = feeds.get(topic_lower, feeds["technology"])

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()

        # Parse RSS XML
        root = ET.fromstring(response.text)
        channel = root.find("channel")
        items = channel.findall("item")[:5]

        headlines = []
        for item in items:
            title = item.find("title")
            desc = item.find("description")
            if title is not None:
                headline = title.text.strip()
                description = desc.text.strip()[:80] + "..." if desc is not None else ""
                headlines.append(f"• {headline}: {description}")

        if not headlines:
            return f"No news found for topic: {topic}"

        return f"Latest {topic.title()} News (BBC):\n" + "\n".join(headlines)

    except httpx.TimeoutException:
        return "News feed timeout. Try again."
    except Exception as e:
        return f"Error fetching news: {str(e)}"


# ─────────────────────────────────────────────────────────
# BUILD EXTERNAL API AGENT
# ─────────────────────────────────────────────────────────

TOOLS = [get_weather, get_exchange_rate, get_tech_news]

SYSTEM_PROMPT = "...same system message text..."
agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

def ask_agent(question: str):
    console.print(f"\n[bold blue]❓ {question}[/bold blue]")
    try:
        result = agent_executor.invoke({"messages": [("user", question)]})
        console.print(Panel( result["messages"][-1].content, border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    console.print("[bold cyan]🌐 Day 4: External API Tools (Live Data!)[/bold cyan]\n")

    questions = [
        "What's the weather like in Islamabad right now?",
        "How much is 500 USD in Pakistani Rupees?",
        "Give me the latest technology news headlines.",
        "Compare the weather in London vs Karachi.",
        "Convert 1000 EUR to AED",
    ]

    for q in questions:
        ask_agent(q)

    console.print("\n[green]✅ Day 4 Complete! External APIs as tools mastered.[/green]")