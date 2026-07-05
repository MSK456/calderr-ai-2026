# 🌅 API Aggregator Agent — Project 2-I-C

**CalderR Agentic AI Engineering Internship 2026 | Week 2**

## What it does
Pulls real-time data from 3 completely free APIs and synthesizes a morning briefing:
- 🌤️ **Weather**: Open-Meteo (current + 3-day forecast, 10 cities)
- 💱 **Currency**: open.er-api.com (live exchange rates, 11 currencies)
- 📰 **News**: BBC RSS (latest headlines, 4 categories)

## No API Keys Needed — All Free!

## Setup
```bash
pip install langchain langchain-groq langgraph python-dotenv httpx rich
```
Add `GROQ_API_KEY=gsk_...` to `.env`

## Run
```bash
cd week2/projects/api_aggregator
python main.py
```

## Architecture