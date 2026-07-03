"""
Lab 2.3: Error Recovery Agent — retries, fallbacks, exponential backoff
"""

import os
import re
import time
import random
import logging
import httpx
from typing import Callable
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("error_recovery")


def exponential_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    delay = min(base * (2 ** attempt) + random.uniform(0, 1), cap)
    return delay


def retry_with_backoff(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> tuple:
    kwargs = kwargs or {}
    attempt_log = []

    for attempt in range(max_attempts):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            attempt_log.append({
                "attempt": attempt + 1,
                "status": "success",
                "elapsed_ms": round(elapsed * 1000),
                "error": None
            })
            logger.info(f"✅ Attempt {attempt+1} succeeded in {elapsed*1000:.0f}ms")
            return result, attempt_log

        except Exception as e:
            elapsed = time.time() - start
            error_type = type(e).__name__
            attempt_log.append({
                "attempt": attempt + 1,
                "status": "failed",
                "elapsed_ms": round(elapsed * 1000),
                "error": f"{error_type}: {str(e)}"
            })
            logger.warning(f"❌ Attempt {attempt+1} failed: {error_type}: {e}")

            if attempt < max_attempts - 1:
                delay = exponential_backoff(attempt, base_delay)
                logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
                time.sleep(delay)

    raise Exception(f"All {max_attempts} attempts failed.")


def _unreliable_weather_api(city: str, fail_rate: float = 0.6) -> str:
    if random.random() < fail_rate:
        raise httpx.TimeoutException(f"Primary weather API timed out for {city}")
    return f"[Primary API] {city}: 30°C, Sunny, 65% humidity"


def _backup_weather_api(city: str) -> str:
    weather_data = {
        "islamabad": "Islamabad: ~28-32°C, typically warm. (Backup source)",
        "karachi":   "Karachi: ~33-38°C, hot and humid. (Backup source)",
        "lahore":    "Lahore: ~35-40°C, very hot summers. (Backup source)",
        "london":    "London: ~15-20°C, often cloudy. (Backup source)",
    }
    return weather_data.get(city.lower(), f"{city}: Weather data unavailable in backup source.")


@tool
def weather_with_fallback(city: str) -> str:
    """Get weather with automatic fallback to backup source if primary fails.
    Includes retry logic with exponential backoff.
    Args:
        city: city name like 'Islamabad', 'London', 'Karachi'
    """
    console.print(f"[dim]🌤️ Fetching weather for {city}...[/dim]")
    try:
        result, log = retry_with_backoff(
            func=_unreliable_weather_api,
            args=(city,),
            kwargs={"fail_rate": 0.7},
            max_attempts=3,
            base_delay=0.5
        )
        console.print(f"[green]Primary API succeeded after {len(log)} attempt(s)[/green]")
        return result
    except Exception:
        console.print(f"[yellow]⚠️ Primary API failed. Switching to backup...[/yellow]")
        logger.warning(f"Primary API exhausted for {city}. Using backup.")
        return f"[FALLBACK] {_backup_weather_api(city)}"


def _unreliable_stock_api(symbol: str, fail_rate: float = 0.5) -> dict:
    if random.random() < fail_rate:
        raise ConnectionError(f"Stock API connection refused for {symbol}")
    prices = {"AAPL": 185.50, "GOOGL": 175.20, "META": 520.30, "MSFT": 420.10}
    price = prices.get(symbol.upper(), random.uniform(50, 500))
    return {"symbol": symbol.upper(), "price": round(price, 2), "change": round(random.uniform(-5, 5), 2)}


@tool
def stock_with_retry(symbol: str) -> str:
    """Get stock price with retry logic. Handles temporary connection failures.
    Args:
        symbol: stock ticker like 'AAPL', 'GOOGL', 'META', 'MSFT'
    """
    console.print(f"[dim]📈 Fetching stock: {symbol}...[/dim]")
    try:
        result, log = retry_with_backoff(
            func=_unreliable_stock_api,
            args=(symbol,),
            max_attempts=4,
            base_delay=0.3
        )
        failed = len([l for l in log if l["status"] == "failed"])
        return f"{result['symbol']}: ${result['price']} ({'+' if result['change']>=0 else ''}{result['change']}%) — fetched after {failed+1} attempt(s)"
    except Exception as e:
        return f"Stock data unavailable for {symbol} after all retries. Error: {str(e)[:50]}"


@tool
def calculate_safe(expression: str) -> str:
    """Safely calculate math expressions with error recovery.
    Args:
        expression: any math expression like '25 * 4 + 10'
    """
    clean = re.sub(r'[^0-9+\-*/().\s%]', '', expression)
    if not clean.strip():
        numbers = re.findall(r'\d+(?:\.\d+)?', expression)
        if numbers:
            return f"Could not parse expression. Fallback sum: {sum(float(n) for n in numbers)}"
        return "Could not evaluate expression. Use format like '25 * 4'"
    try:
        result = eval(clean)
        return f"{expression} = {round(result, 6)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {str(e)}"


TOOLS = [weather_with_fallback, stock_with_retry, calculate_safe]

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a resilient AI assistant.
Your tools have built-in retry and fallback logic.
When a tool returns [FALLBACK] data, mention it in your answer.
Always provide the best available answer, even with imperfect data."""

agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🛡️ Lab 2.3: Error Recovery Agent[/bold cyan]\n"
        "[dim]Retries + Exponential Backoff + Fallbacks | Primary → Backup APIs[/dim]",
        border_style="cyan"
    ))

    console.print("\n[yellow]Note: Primary APIs have 60-70% fail rate to demonstrate retries[/yellow]\n")

    queries = [
        "What's the weather in Islamabad?",
        "What's the current price of Apple stock (AAPL)?",
        "What is 15% of 45000 + 8000?",
        "Check weather in London and stock price of META",
    ]

    for q in queries:
        console.print(f"\n[bold blue]❓ {q}[/bold blue]")
        result = agent_executor.invoke({"messages": [("user", q)]})
        console.print(Panel(result["messages"][-1].content, border_style="green"))

    # Backoff table
    console.print("\n[bold yellow]📊 Exponential Backoff Pattern Demo:[/bold yellow]")
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Attempt", style="cyan", justify="center")
    table.add_column("Wait Before", style="yellow", justify="center")
    table.add_column("Formula", style="dim")

    for i in range(5):
        delay = min(1.0 * (2 ** i), 30)
        table.add_row(str(i+1), f"{delay:.0f}s" if i > 0 else "immediate", f"1 × 2^{i} = {delay:.0f}s")
    console.print(table)

    console.print("\n[green]✅ Lab 2.3 Complete! Error recovery with retries and fallbacks done.[/green]")