# 🌅 API Aggregator Agent — Project 2-I-C

**CalderR Agentic AI Engineering Internship 2026 | Week 2**

---

## 🎯 What it does

Pulls real-time data from 3 completely free APIs and synthesizes a morning briefing:

- 🌤️ **Weather** — Open-Meteo API (current conditions + 3-day forecast, 10 cities)
- 💱 **Currency** — open.er-api.com (live exchange rates, 11 currencies)
- 📰 **News** — BBC RSS Feed (latest headlines, 4 categories)

**No API Keys Needed — All 3 sources are completely free!**

---

## 🛠️ Setup

```bash
pip install langchain langchain-groq langgraph python-dotenv httpx rich
```

Add `GROQ_API_KEY=gsk_...` to your `.env` file.

---

## 🚀 Run

```bash
cd week2/projects/api_aggregator
python main.py
```

---

## 🏙️ Supported Cities

Islamabad · Karachi · Lahore · London · Dubai · New York · Paris · Tokyo · Riyadh · Istanbul

## 💱 Supported Currencies

USD · EUR · GBP · PKR · AED · SAR · JPY · CAD · AUD · INR · TRY

## 📰 News Categories

technology · business · science · world

---

## 🏗️ Architecture

    User Query
        ↓
    Agent (llama-4-scout-17b via Groq)
        ↓
        ├── tool_get_weather          →  Open-Meteo API (free, no key)
        ├── tool_get_exchange_rates   →  open.er-api.com (free, no key)
        ├── tool_convert_currency     →  open.er-api.com (free, no key)
        ├── tool_get_news             →  BBC RSS Feed (free, no key)
        └── tool_generate_briefing    →  Groq LLM synthesis
                    ↓
        Formatted Response to User

---

## 💬 Example Queries

- "Give me a complete morning briefing for Islamabad."
- "What is the weather in Dubai?"
- "Convert 1000 USD to Pakistani Rupees."
- "Latest technology news please."
- "Compare weather in London vs Karachi."
- "What is 500 EUR in AED?"

---

## 📁 Project Structure

    api_aggregator/
    ├── main.py           ← Agent + tools + interactive CLI
    ├── tools/
    │   ├── __init__.py
    │   ├── weather.py    ← Open-Meteo API wrapper
    │   ├── currency.py   ← open.er-api.com wrapper
    │   └── news.py       ← BBC RSS parser
    └── README.md

---

## 🔑 Key Concepts Used

- **LangGraph `create_react_agent`** — modern agent loop (replaces AgentExecutor)
- **`@tool` decorator** — wraps Python functions as LLM-callable tools
- **httpx** — async-ready HTTP client for API calls
- **RSS XML parsing** — `xml.etree.ElementTree` for BBC news feed
- **Tool chaining** — agent calls multiple tools then synthesizes final answer

---

*Built with 🔥 during the CalderR Agentic AI Engineering Internship 2026*