# 📄 Document Processing Pipeline — Project 2-P-A

**CalderR Agentic AI Engineering Internship 2026 | Week 2**

## Features
- Upload PDF, DOCX, or TXT documents via web UI or REST API
- AI extracts: title, summary, entities, key terms, action items, sentiment
- Pydantic v2 validated structured output (json_mode)
- SQLite storage for all processed documents
- Clean dark-themed web UI + full REST API
- Auto-generated FastAPI docs at /docs

## Setup
```bash
pip install fastapi uvicorn pymupdf python-docx langchain langchain-groq pydantic python-dotenv rich
```
Add `GROQ_API_KEY=gsk_...` to `.env`

## Run
```bash
cd week2/projects/doc_processor
python main.py
```
Open browser: http://localhost:8000

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Web UI |
| POST | /process | Upload + process document |
| GET | /documents | List all documents |
| GET | /documents/{id} | Get specific document |
| DELETE | /documents/{id} | Delete document |
| GET | /docs | Auto-generated API docs |

## Architecture
```
File Upload (FastAPI)
    ↓
Document Parser
  PDF  → PyMuPDF (fitz)
  DOCX → python-docx
  TXT  → built-in
    ↓
LLM Extraction (llama-3.3-70b + json_mode)
    ↓
Pydantic v2 Validation (DocumentAnalysis)
    ↓
SQLite Storage
    ↓
REST API Response + Web UI
```