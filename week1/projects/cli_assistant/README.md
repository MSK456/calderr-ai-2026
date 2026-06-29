# 🧠 Intelligent CLI Assistant — Project 1-I-A

**CalderR Agentic AI Engineering Internship 2026 | Week 1**

## Features
- 5 topic domains: cooking, history, programming, ai, general
- Full conversation history (10+ turns)
- Rich terminal UI with markdown rendering
- Token usage tracking per response
- /clear, /exit, /history, /stats, /topic commands
- Graceful error handling

## Setup
```bash
pip install groq langchain-groq python-dotenv rich
```
Add `GROQ_API_KEY=gsk_...` to `.env` file.

## Run
```bash
python main.py
```

## Example Conversations
See `examples/` folder for 3 documented conversations.

## Architecture
```
User Input → Command Parser → Topic Manager → Groq API → Rich Display
                                    ↓
                           Conversation History Buffer
```