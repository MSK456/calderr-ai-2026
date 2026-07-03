"""
Day 5: Production-grade agent — logging, retries, structured output, all tools combined
"""

import os
import re
import logging
import time
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler("week2/day5_integration/agent.log")
    ]
)
logger = logging.getLogger("production_agent")
console = Console()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True
)
def fetch_with_retry(url: str, params: dict = None) -> dict:
    with httpx.Client(timeout=8.0) as client:
        response = client.get(url, params=params or {}, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.json() if "json" in response.headers.get("content-type", "") else response.text


CITY_COORDS = {
    "islamabad": (33.7215, 73.0433), "karachi": (24.8607, 67.0011),
    "lahore": (31.5204, 74.3587), "london": (51.5074, -0.1278),
    "dubai": (25.2048, 55.2708), "new york": (40.7128, -74.0060),
}


@tool
def weather_tool(city: str) -> str:
    """Get current weather for a city. Returns temperature, humidity, conditions.
    Args:
        city: city name like 'Islamabad', 'London', 'Dubai'
    """
    logger.info(f"weather_tool called: city={city}")
    city_lower = city.lower().strip()
    coords = CITY_COORDS.get(city_lower)
    if not coords:
        return f"City '{city}' not found. Available: {', '.join(c.title() for c in CITY_COORDS)}"
    lat, lon = coords
    try:
        data = fetch_with_retry(
            "https://api.open-meteo.com/v1/forecast",
            {"latitude": lat, "longitude": lon,
             "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
             "timezone": "auto"}
        )
        c = data["current"]
        codes = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                 61: "Light rain", 63: "Rain", 95: "Thunderstorm"}
        cond = codes.get(c["weather_code"], f"Code {c['weather_code']}")
        result = f"{city.title()}: {cond}, {c['temperature_2m']}°C, Humidity {c['relative_humidity_2m']}%, Wind {c['wind_speed_10m']} km/h"
        logger.info(f"weather_tool success: {result}")
        return result
    except Exception as e:
        logger.error(f"weather_tool failed: {e}")
        return f"Weather unavailable for {city}: {str(e)}"


@tool
def currency_tool(query: str) -> str:
    """Convert currency amounts. Use for any currency conversion questions.
    Args:
        query: like '100 USD to PKR', 'EUR to GBP rate'
    """
    logger.info(f"currency_tool called: query={query}")
    query_upper = query.upper()
    amount_match = re.search(r'(\d+(?:\.\d+)?)', query)
    amount = float(amount_match.group(1)) if amount_match else 1.0
    currencies = re.findall(r'\b([A-Z]{3})\b', query_upper)
    if len(currencies) < 2:
        return "Specify currencies: '100 USD to PKR'"
    from_cur, to_cur = currencies[0], currencies[1]
    try:
        data = fetch_with_retry(f"https://open.er-api.com/v6/latest/{from_cur}")
        rate = data["rates"].get(to_cur)
        if not rate:
            return f"{to_cur} not found"
        result = f"{amount} {from_cur} = {amount * rate:.2f} {to_cur}"
        logger.info(f"currency_tool success: {result}")
        return result
    except Exception as e:
        logger.error(f"currency_tool failed: {e}")
        return f"Currency API unavailable: {str(e)}"


@tool
def news_tool(category: str = "technology") -> str:
    """Get latest news headlines. Categories: technology, business, science.
    Args:
        category: 'technology', 'business', or 'science'
    """
    logger.info(f"news_tool called: category={category}")
    feeds = {
        "technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    }
    url = feeds.get(category.lower(), feeds["technology"])
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = root.find("channel").findall("item")[:4]
        headlines = [f"• {item.find('title').text}" for item in items if item.find("title") is not None]
        result = "\n".join(headlines)
        logger.info(f"news_tool success: {len(headlines)} headlines")
        return f"Latest {category} news:\n{result}"
    except Exception as e:
        logger.error(f"news_tool failed: {e}")
        return f"News unavailable: {str(e)}"


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions safely.
    Args:
        expression: math expression like '25 * 4 + 10'
    """
    logger.info(f"calculator_tool called: {expression}")
    clean = re.sub(r'[^0-9+\-*/().\s]', '', expression)
    try:
        result = eval(clean)
        return f"{expression} = {round(result, 6)}"
    except Exception as e:
        return f"Math error: {e}"


TOOLS = [weather_tool, currency_tool, news_tool, calculator_tool]

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a production AI assistant with live data tools.
Rules:
- Always use tools for live data (weather, currency, news)
- Never guess or fabricate real-time information
- Be concise and accurate in final answers
- If a tool fails, clearly state what went wrong"""

agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_production_query(question: str) -> str:
    logger.info(f"Query received: {question}")
    start = time.time()
    try:
        result = agent_executor.invoke({"messages": [("user", question)]})
        elapsed = time.time() - start
        logger.info(f"Query completed in {elapsed:.2f}s")
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return f"I encountered an error: {str(e)}"


if __name__ == "__main__":
    console.print("[bold cyan]⚙️  Day 5: Production Agent (with logging + retries)[/bold cyan]")
    console.print("[dim]Logs saved to: week2/day5_integration/agent.log[/dim]\n")

    queries = [
        "What's the weather in Dubai and how much is 500 AED in PKR?",
        "Give me 3 tech news headlines",
        "If I have 1500 USD and spend 30%, how much do I have left? Also convert remaining to EUR.",
    ]

    for q in queries:
        console.print(f"\n[bold blue]❓ {q}[/bold blue]")
        answer = run_production_query(q)
        console.print(Panel(answer, border_style="green"))

    console.print("\n[green]✅ Day 5 Complete! Production agent with retries + logging done.[/green]")