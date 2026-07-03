# 🤖 CalderR Agentic AI Engineering Internship 2026

**Intern:** Muhammad Shaheed (MSK456)
**University:** FAST University, Islamabad
**Program:** Agentic AI Engineering Internship 2026
**GitHub:** [MSK456](https://github.com/MSK456)

---

## 📌 About This Repository

This repository contains all my work from the **CalderR Agentic AI Engineering Internship 2026** — a 10-week intensive program covering LLM engineering, agentic systems, prompt engineering, tool calling, RAG pipelines, and production AI deployment.

---

## ✅ Week 1 — AI Fundamentals & Agentic AI Foundations
**22 June – 26 June 2026**

### Daily Work
| Day | Topic | File |
|-----|-------|------|
| Mon | LLM Foundations — model comparison, temperature, context window | `week1/day1_llm_foundations/explore_models.py` |
| Tue | Agentic AI — Manual ReAct loop from scratch (no framework) | `week1/day2_agentic_concepts/react_loop.py` |
| Wed | LangChain Core — LCEL chains, RunnableParallel, ChromaDB RAG | `week1/day3_langchain_core/chains.py` |
| Thu | Prompt Engineering — CoT, few-shot, negative prompting | `week1/day4_prompt_engineering/prompt_experiments.py` |
| Fri | Integration — full multi-tool chatbot with memory | `week1/day5_integration/` |

### Labs
| Lab | Description | File |
|-----|-------------|------|
| Lab 1.1 | Groq CLI Chatbot — conversation history, Rich UI, token tracking | `week1/labs/lab1_1_groq_chatbot.py` |
| Lab 1.2 | Manual ReAct Loop — 3 tools, no framework, pure Python | `week1/labs/lab1_2_react_loop.py` |
| Lab 1.3 | Prompt A/B Test — 5 system prompts, news summarization comparison | `week1/labs/lab1_3_prompt_ab_test.py` |

### Projects
| Project | Description | Stack |
|---------|-------------|-------|
| **1-I-A: Intelligent CLI Assistant** | Multi-topic chatbot (cooking/history/programming/ai), Rich terminal UI, 10+ turn memory, topic switching | Python, LangChain, Groq, Rich |
| **1-P-B: Multi-Model Comparison Engine** | Benchmarks 4 Groq models on 8 tasks — latency, accuracy, token usage — generates HTML + JSON reports | asyncio, YAML, Groq, Plotly |

---

## ✅ Week 2 — Prompt Engineering & Tool Calling
**29 June – 3 July 2026**

### Daily Work
| Day | Topic | File |
|-----|-------|------|
| Mon | Advanced Prompting — CoT, ToT, Self-Consistency, Meta-Prompting | `week2/day1_advanced_prompting/advanced_prompting.py` |
| Tue | Structured Outputs — Pydantic v2, with_structured_output(), 5 schemas | `week2/day2_structured_outputs/structured_outputs.py` |
| Wed | Tool Calling Basics — 5-tool agent with LangGraph create_react_agent | `week2/day3_tool_calling/tool_calling_basics.py` |
| Thu | External APIs as Tools — Open-Meteo, open.er-api, BBC RSS (all free) | `week2/day4_external_apis/external_api_tools.py` |
| Fri | Integration — production agent with logging, retries, error handling | `week2/day5_integration/production_agent.py` |

### Labs
| Lab | Description | File |
|-----|-------------|------|
| Lab 2.1 | Structured Job Posting Extractor — Pydantic v2 + json_mode | `week2/labs/lab2_1_structured_extractor.py` |
| Lab 2.2 | Multi-Tool Research Agent — 6 tools, smart routing | `week2/labs/lab2_2_multi_tool_agent.py` |
| Lab 2.3 | Error Recovery Agent — exponential backoff, retries, fallbacks | `week2/labs/lab2_3_error_recovery.py` |

### Projects
| Project | Description | Stack |
|---------|-------------|-------|
| **2-I-C: API Aggregator Agent** | Pulls live data from 3 free APIs (weather, currency, news) and synthesizes a morning briefing | Python, LangGraph, httpx, BBC RSS, Open-Meteo |
| **2-P-A: Intelligent Document Processing Pipeline** | Upload PDF/DOCX/TXT → AI extracts entities, summary, action items → Pydantic validated → stored in SQLite → REST API | FastAPI, PyMuPDF, python-docx, Pydantic v2, SQLite |

---

## 🛠️ Tech Stack