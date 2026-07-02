"""
Day 3: Tool Calling — @tool decorator, bind_tools(), 5-tool agent
"""

import os
import re
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────────────────────
# 5 TOOLS — each clearly defined with docstrings
# ─────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """Search a knowledge base of AI and tech facts.
    Use this when user asks about AI concepts, companies, or technologies.

    Args:
        query: keyword or topic to search (e.g., 'langchain', 'groq', 'transformer')
    """
    kb = {
        "langchain": "LangChain is a framework for building LLM-powered apps using chains, agents, and tools.",
        "groq": "Groq is a company with custom LPU hardware enabling ultra-fast LLM inference. Free tier: 14,400 req/day.",
        "transformer": "Transformer architecture (2017) uses self-attention to process sequences in parallel. Powers all modern LLMs.",
        "rag": "RAG (Retrieval Augmented Generation) combines vector search + LLM to answer from specific documents.",
        "pydantic": "Pydantic v2 is a Python library for data validation using type hints. Used for structured LLM outputs.",
        "chromadb": "ChromaDB is an open-source vector database for storing and searching text embeddings.",
        "react": "ReAct is an agent pattern combining Reasoning + Acting. Agent thinks → acts → observes → repeats.",
        "fastapi": "FastAPI is a modern async Python web framework for building REST APIs with automatic documentation.",
        "python": "Python is a high-level language used extensively in AI/ML. Version 3.11+ recommended for AI projects.",
    }
    q = query.lower()
    for key, val in kb.items():
        if key in q:
            return f"Found: {val}"
    return f"No results for '{query}'. Available topics: {', '.join(kb.keys())}"


@tool
def calculate(expression: str) -> str:
    """Calculate any mathematical expression safely.
    Use this for ANY math operation: arithmetic, percentages, basic formulas.

    Args:
        expression: math expression like '25 * 4', '100 / 5 + 10', '2 ** 8'
    """
    clean = re.sub(r'[^0-9+\-*/().\s%]', '', expression)
    if not clean.strip():
        return f"Invalid expression: '{expression}'"
    try:
        result = eval(clean)
        return f"{expression} = {round(result, 6)}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def format_date(date_input: str) -> str:
    """Get current date/time or format a given date string.
    Use when user asks about today's date, current time, or date formatting.

    Args:
        date_input: 'now' for current datetime, or a date string like '2026-07-01'
    """
    if date_input.lower() in ["now", "today", "current"]:
        now = datetime.now()
        return f"Current datetime: {now.strftime('%A, %B %d %Y')} at {now.strftime('%I:%M %p')}"
    try:
        dt = datetime.fromisoformat(date_input)
        return f"Formatted: {dt.strftime('%A, %B %d %Y')}"
    except:
        return f"Could not parse date: '{date_input}'. Use 'now' or ISO format like '2026-07-01'"


@tool
def convert_units(conversion: str) -> str:
    """Convert between common units: temperature, distance, weight, data size.
    Use this for unit conversion questions.

    Args:
        conversion: conversion request like '100 celsius to fahrenheit',
                   '5 km to miles', '70 kg to pounds', '1 GB to MB'
    """
    conv = conversion.lower()

    # Temperature
    if "celsius" in conv and "fahrenheit" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        result = (num * 9 / 5) + 32
        return f"{num}°C = {result:.1f}°F"
    if "fahrenheit" in conv and "celsius" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        result = (num - 32) * 5 / 9
        return f"{num}°F = {result:.1f}°C"

    # Distance
    if "km" in conv and "mile" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        if "km to mile" in conv:
            return f"{num} km = {num * 0.621371:.2f} miles"
        return f"{num} miles = {num * 1.60934:.2f} km"

    # Weight
    if "kg" in conv and "pound" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        return f"{num} kg = {num * 2.20462:.2f} pounds"

    # Data
    if "gb" in conv and "mb" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        return f"{num} GB = {num * 1024:.0f} MB"
    if "mb" in conv and "gb" in conv:
        num = float(re.search(r'[\d.]+', conv).group())
        return f"{num} MB = {num / 1024:.3f} GB"

    return f"Could not understand conversion: '{conversion}'. Example: '100 celsius to fahrenheit'"


@tool
def summarize_text(text: str) -> str:
    """Summarize any long text into 2-3 concise sentences.
    Use when user provides a long text and wants a summary.

    Args:
        text: the text content to summarize (any length)
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import os

    summarizer = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the given text in exactly 2-3 sentences. Be concise and capture the main points."),
        ("user", "{text}")
    ])
    chain = prompt | summarizer | StrOutputParser()
    return chain.invoke({"text": text})


# ─────────────────────────────────────────────────────────
# BUILD THE AGENT
# ─────────────────────────────────────────────────────────

TOOLS = [search_knowledge_base, calculate, format_date, convert_units, summarize_text]

SYSTEM_PROMPT = """You are a smart AI assistant with access to 5 tools.
Always use a tool when the user's question requires factual lookup, calculation, date info, unit conversion, or summarization.
Do NOT make up facts — use the search_knowledge_base tool instead.
After getting a tool result, give a clear, friendly final answer."""

agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

def run_agent(question: str):
    console.print(f"\n[bold blue]❓ {question}[/bold blue]")
    result = agent_executor.invoke({"messages": [("user", question)]})
    final = result["messages"][-1].content
    console.print(Panel(final, border_style="green", title="[green]Agent Answer[/green]"))

if __name__ == "__main__":
    console.print("[bold cyan]🛠️  Day 3: 5-Tool Agent Demo[/bold cyan]\n")

    tests = [
        "What is RAG in AI?",
        "Calculate 15% of 8500 plus 250",
        "What day is today?",
        "Convert 37 celsius to fahrenheit",
        "Summarize this: LangChain Expression Language (LCEL) is a declarative way to compose LangChain components. LCEL was designed from day 1 to support putting prototypes in production with no code changes. Using LCEL means all components implement the Runnable interface.",
    ]

    for q in tests:
        run_agent(q)

    console.print("\n[green]✅ Day 3 Complete! Tool calling with 5-tool agent mastered.[/green]")