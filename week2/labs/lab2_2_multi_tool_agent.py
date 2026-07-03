"""
Lab 2.2: Multi-Tool Research Agent — 6 tools, smart routing
"""

import os
import re
import json
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.panel import Panel
from rich import box

load_dotenv()
console = Console()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

AI_FACTS = {
    "gpt": "GPT-4 is OpenAI's multimodal model with 1.8T parameters, released March 2023.",
    "llama": "Meta's LLaMA models are open-source LLMs. LLaMA 3.3 70B is available free on Groq.",
    "langchain": "LangChain is a framework for building LLM-powered applications. Version 0.3+ uses LCEL.",
    "groq": "Groq uses LPU (Language Processing Unit) hardware for ultra-fast inference, often 10x faster than GPU.",
    "anthropic": "Anthropic created Claude AI. They focus on AI safety and Constitutional AI (CAI) training.",
    "openai": "OpenAI created GPT series, DALL-E, and Whisper. Founded in 2015, now valued at ~$150B.",
    "transformer": "Transformer architecture introduced in 2017 paper 'Attention is All You Need' by Google.",
    "rag": "RAG combines vector search with LLM generation for accurate, grounded responses.",
    "chromadb": "ChromaDB is a free open-source vector database for embeddings storage and similarity search.",
    "pydantic": "Pydantic v2 is rewritten in Rust for 5-50x faster validation. Used for structured LLM outputs.",
}


@tool
def search_ai_facts(topic: str) -> str:
    """Search a knowledge base of AI/ML facts and concepts.
    Use for questions about AI companies, models, frameworks, or technologies.
    Args:
        topic: keyword to search like 'gpt', 'langchain', 'groq', 'rag'
    """
    topic_lower = topic.lower()
    for key, val in AI_FACTS.items():
        if key in topic_lower:
            return f"[FOUND] {val}"
    return f"No facts for '{topic}'. Topics available: {', '.join(AI_FACTS.keys())}"


@tool
def web_search_mock(query: str) -> str:
    """Mock web search returning relevant results for AI/tech topics.
    Use when user needs current web information or news not in the knowledge base.
    Args:
        query: search query like 'latest AI news 2026' or 'Python best practices'
    """
    mock_results = {
        "ai 2026": "Top results: 1) OpenAI releases GPT-5 with multimodal agents (TechCrunch) 2) Google DeepMind launches Gemini 3.0 (Wired) 3) AI agents now handle 40% of software tasks (MIT Review)",
        "python": "Top results: 1) Python 3.14 released with JIT compiler 2) FastAPI becomes most popular web framework 3) Pydantic v3 announced with full Rust core",
        "langchain": "Top results: 1) LangChain 0.4 adds native streaming 2) LangGraph becomes standard for agent workflows 3) LangChain raises $100M Series B",
    }
    query_lower = query.lower()
    for key, val in mock_results.items():
        if any(k in query_lower for k in key.split()):
            return val
    return f"Search results for '{query}': No specific results found in mock database. In production, this would call a real search API."


@tool
def calculate_stats(expression: str) -> str:
    """Perform mathematical calculations including percentages and basic stats.
    Args:
        expression: math expression like '25 * 4', '(100 + 200) / 3', '15% of 8500'
    """
    expr = expression.lower()
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', expr)
    if pct_match:
        pct, total = float(pct_match.group(1)), float(pct_match.group(2))
        return f"{pct}% of {total} = {pct * total / 100:.2f}"
    clean = re.sub(r'[^0-9+\-*/().\s]', '', expression)
    try:
        result = eval(clean)
        return f"{expression} = {round(result, 6)}"
    except:
        return f"Cannot evaluate: '{expression}'"


@tool
def get_current_date() -> str:
    """Get the current date, time, and day of week.
    Use whenever user asks about today's date, current time, or what day it is.
    """
    now = datetime.now()
    return (f"Current date: {now.strftime('%A, %B %d, %Y')}\n"
            f"Time: {now.strftime('%I:%M %p')}\n"
            f"Week number: {now.isocalendar()[1]}")


@tool
def classify_sentiment(text: str) -> str:
    """Classify the sentiment of any text as positive, negative, or neutral.
    Use when asked to analyze tone, opinion, or emotional content of text.
    Args:
        text: any text to analyze for sentiment
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    analyzer = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze sentiment. Reply in format: SENTIMENT: [positive/negative/neutral] | CONFIDENCE: [0-100%] | REASON: [one sentence]"),
        ("user", "{text}")
    ])
    chain = prompt | analyzer | StrOutputParser()
    return chain.invoke({"text": text})


@tool
def summarize_content(content: str) -> str:
    """Summarize long text into 3 bullet points.
    Use when user provides text and wants it condensed or simplified.
    Args:
        content: text to summarize
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    summarizer = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize in exactly 3 bullet points. Each bullet: max 20 words. Start each with •"),
        ("user", "{content}")
    ])
    chain = prompt | summarizer | StrOutputParser()
    return chain.invoke({"content": content})


TOOLS = [search_ai_facts, web_search_mock, calculate_stats,
         get_current_date, classify_sentiment, summarize_content]

SYSTEM_PROMPT = """You are a research assistant with 6 specialized tools:
1. search_ai_facts → AI/ML knowledge base
2. web_search_mock → web search for current info
3. calculate_stats → math and percentages
4. get_current_date → today's date/time
5. classify_sentiment → analyze text tone
6. summarize_content → condense long text

ROUTING RULES:
- Tech/AI concepts → search_ai_facts first
- Current events/news → web_search_mock
- Numbers/calculations → calculate_stats
- Date questions → get_current_date
- Sentiment analysis → classify_sentiment
- Summarization → summarize_content
Always use the most appropriate tool. Combine tools when needed."""

agent_executor = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def research(query: str):
    console.print(f"\n[bold blue]❓ {query}[/bold blue]")
    with console.status("[yellow]Researching...[/yellow]"):
        result = agent_executor.invoke({"messages": [("user", query)]})
    console.print(Panel(
        result["messages"][-1].content,
        border_style="green",
        title="[green]Research Result[/green]"
    ))


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🔭 Lab 2.2: Multi-Tool Research Agent[/bold cyan]\n"
        "[dim]6 tools | Smart routing | Research assistant[/dim]",
        border_style="cyan"
    ))

    test_queries = [
        "What is Groq and how does it differ from regular GPU inference?",
        "Search the web for latest AI news in 2026",
        "What is 30% of 85,000 plus 12,500?",
        "What day is today?",
        "Is this review positive or negative: 'The product works fine but delivery took forever and customer support never responded.'",
        "Summarize this: LangChain Expression Language (LCEL) is a declarative way to compose LangChain components. LCEL was designed from day 1 to support putting prototypes in production, no code changes required. Using LCEL means all components implement the Runnable interface which supports sync, async, batch and streaming calls.",
    ]

    for q in test_queries:
        research(q)

    console.print("\n[green]✅ Lab 2.2 Complete! 6-tool agent routing successfully tested.[/green]")