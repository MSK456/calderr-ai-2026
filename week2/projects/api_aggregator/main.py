"""
Project 2-I-C: API Aggregator Agent
Live data from 3 free APIs → Morning Briefing
- Open-Meteo: weather
- open.er-api.com: currency
- BBC RSS: news
"""

import os
import re
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.weather import fetch_weather
from tools.currency import fetch_rates, convert_currency
from tools.news import fetch_news

load_dotenv()
console = Console()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)


# ─────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────

@tool
def tool_get_weather(city: str) -> str:
    """Get current weather and 3-day forecast for a city.
    Available: Islamabad, Karachi, Lahore, London, Dubai, New York, Paris, Tokyo, Riyadh, Istanbul.
    Args:
        city: city name
    """
    data = fetch_weather(city)
    if "error" in data:
        return f"Weather error: {data['error']}"

    forecast = " | ".join(
        f"{d['day']}: {d['max']}°/{d['min']}°C"
        for d in data["forecast_3days"]
    )
    return (
        f"Weather in {data['city']}: {data['condition']}, "
        f"{data['temperature_c']}°C, Humidity {data['humidity_pct']}%, "
        f"Wind {data['wind_kmh']} km/h | "
        f"3-Day: {forecast}"
    )


@tool
def tool_get_exchange_rates(base_currency: str) -> str:
    """Get live exchange rates for a base currency.
    Args:
        base_currency: like 'USD', 'EUR', 'PKR', 'AED', 'GBP'
    """
    data = fetch_rates(base_currency)
    if "error" in data:
        return f"Currency error: {data['error']}"

    rates_str = " | ".join(
        f"1 {data['base']} = {v} {k}"
        for k, v in list(data["rates"].items())[:6]
    )
    return f"Rates ({data['base']}): {rates_str}"


@tool
def tool_convert_currency(amount: str, from_currency: str, to_currency: str) -> str:
    """Convert a specific amount between currencies.
    Args:
        amount: numeric amount as string like '1000' or '500.50'
        from_currency: source currency like 'USD', 'EUR', 'GBP'
        to_currency: target currency like 'PKR', 'AED', 'SAR'
    """
    try:
        amount_float = float(str(amount).replace(",", ""))
    except ValueError:
        return f"Invalid amount: '{amount}'. Please provide a numeric value."

    data = convert_currency(amount_float, from_currency, to_currency)
    if "error" in data:
        return f"Conversion error: {data['error']}"
    return f"{data['from']} = {data['to']} (Rate: {data['rate']})"

@tool
def tool_get_news(category: str) -> str:
    """Get latest BBC news headlines.
    Args:
        category: 'technology', 'business', 'science', or 'world'
    """
    data = fetch_news(category, count=5)
    if "error" in data:
        return f"News error: {data['error']}"

    headlines = "\n".join(
        f"{i+1}. {a['title']}"
        for i, a in enumerate(data["articles"])
    )
    return f"Latest {data['category'].title()} News ({data['source']}):\n{headlines}"


@tool
def tool_generate_briefing(weather: str, currency: str, news: str) -> str:
    """Generate a polished morning briefing from collected data.
    Call this LAST after getting weather, currency and news data.
    Args:
        weather: weather data string
        currency: currency data string
        news: news headlines string
    """
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import SystemMessage, HumanMessage

    summarizer = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )
    chain = summarizer | StrOutputParser()
    return chain.invoke([
        SystemMessage(content="""You write concise morning briefings.
Format with these 3 sections:
🌤️ WEATHER SNAPSHOT — 2 sentences
💱 MARKETS & CURRENCY — 2 sentences
📰 TOP STORIES — 3 bullet points
Be informative and professional."""),
        HumanMessage(content=f"WEATHER: {weather}\n\nCURRENCY: {currency}\n\nNEWS: {news}")
    ])


# ─────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────

TOOLS = [tool_get_weather, tool_get_exchange_rates,
         tool_convert_currency, tool_get_news, tool_generate_briefing]

SYSTEM_PROMPT = """You are a Morning Briefing Agent with live data tools.

For a full morning briefing:
1. Call tool_get_weather for the city
2. Call tool_get_exchange_rates for USD
3. Call tool_get_news for 'business'
4. Call tool_generate_briefing with all collected data

For specific questions (just weather / just currency / just news):
→ Call only the relevant tool(s)

NEVER invent or guess live data. Always use tools."""

agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run(query: str):
    console.print(f"\n[bold blue]❓ {query}[/bold blue]")
    with console.status("[yellow]⚡ Fetching live data...[/yellow]"):
        result = agent_executor.invoke({"messages": [("user", query)]})
    console.print(Panel(
        result["messages"][-1].content,
        title="[bold green]🌅 Response[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))


def interactive_mode():
    console.print(Panel.fit(
        "[bold cyan]🌅 API Aggregator Agent[/bold cyan]\n"
        "[dim]Type your query or 'exit' to quit[/dim]",
        border_style="cyan"
    ))
    while True:
        query = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if query:
            run(query)


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🌅 API Aggregator Agent — Project 2-I-C[/bold cyan]\n"
        "[dim]Live: Open-Meteo + open.er-api.com + BBC RSS[/dim]\n"
        "[dim]No API keys needed![/dim]",
        border_style="cyan"
    ))

    # Demo queries
    demo_queries = [
        "Give me a complete morning briefing for Islamabad.",
        "What's the weather in Dubai?",
        "Convert 1000 USD to Pakistani Rupees.",
        "Latest technology news please.",
    ]

    for q in demo_queries:
        run(q)

    console.print("\n[bold yellow]🎯 Entering interactive mode...[/bold yellow]")
    interactive_mode()
    console.print("\n[green]✅ Project 2-I-C Complete![/green]")