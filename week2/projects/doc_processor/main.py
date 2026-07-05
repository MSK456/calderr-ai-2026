"""
Project 2-P-A: Intelligent Document Processing Pipeline
FastAPI + Document Parser + LLM Extraction + Pydantic v2 + SQLite
Supports: PDF, DOCX, TXT
"""

import os
import sys
import time
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parsers import parse_document
from extractor import extract_from_text
from database import init_db, save_result, get_all_documents, get_document

load_dotenv()
init_db()

UPLOAD_DIR = Path("week2/projects/doc_processor/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="📄 Document Processing Pipeline",
    description="Upload PDF, DOCX, or TXT → AI extracts structured information",
    version="1.0.0"
)


# ─────────────────────────────────────────────────────────
# WEB UI
# ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>📄 Document Processor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; padding: 30px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 8px; }
        p { color: #8b949e; margin-bottom: 24px; }
        .upload-box { background: #161b22; border: 2px dashed #30363d; border-radius: 12px; padding: 40px; text-align: center; margin-bottom: 24px; transition: border-color 0.2s; }
        .upload-box:hover { border-color: #58a6ff; }
        input[type=file] { display: block; margin: 16px auto; color: #e6edf3; cursor: pointer; }
        button { background: #238636; color: white; border: none; padding: 12px 32px; border-radius: 8px; cursor: pointer; font-size: 15px; font-weight: 600; transition: background 0.2s; }
        button:hover { background: #2ea043; }
        button:disabled { background: #30363d; cursor: not-allowed; }
        .result { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
        .result h3 { color: #58a6ff; margin-bottom: 16px; }
        .field { display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid #21262d; }
        .field-name { color: #8b949e; min-width: 140px; font-size: 13px; }
        .field-value { color: #e6edf3; font-size: 13px; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin: 2px; }
        .badge-blue { background: #1f6feb33; color: #79c0ff; }
        .badge-green { background: #23863633; color: #3fb950; }
        .badge-orange { background: #b0880033; color: #f0c000; }
        .doc-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 10px; }
        .doc-card h4 { color: #e6edf3; margin-bottom: 6px; }
        .doc-card p { color: #8b949e; font-size: 12px; margin: 0; }
        .loading { display: none; color: #58a6ff; margin-top: 12px; }
        h2 { color: #e6edf3; margin: 24px 0 12px; }
    </style>
</head>
<body>
<div class="container">
    <h1>📄 AI Document Processor</h1>
    <p>Upload PDF, DOCX, or TXT → Instant structured extraction via AI</p>

    <div class="upload-box">
        <h3 style="color:#e6edf3;margin-bottom:12px">📁 Upload Document</h3>
        <input type="file" id="fileInput" accept=".pdf,.docx,.txt">
        <button id="uploadBtn" onclick="uploadFile()">🚀 Process Document</button>
        <div class="loading" id="loading">⏳ Processing... this may take 10-20 seconds</div>
    </div>

    <div id="result"></div>

    <h2>📋 Processed Documents</h2>
    <div id="docs"></div>
</div>

<script>
async function uploadFile() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) { alert('Please select a file first!'); return; }

    const btn = document.getElementById('uploadBtn');
    const loading = document.getElementById('loading');
    btn.disabled = true;
    loading.style.display = 'block';
    document.getElementById('result').innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/process', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
            document.getElementById('result').innerHTML =
                `<div class="result"><h3 style="color:#f85149">❌ Error</h3><p>${data.detail}</p></div>`;
            return;
        }

        const a = data.analysis;
        const entities = (a.entities || []).map(e =>
            `<span class="badge badge-blue">${e.text} (${e.entity_type})</span>`).join('');
        const terms = (a.key_terms || []).map(t =>
            `<span class="badge badge-green">${t}</span>`).join('');
        const actions = (a.action_items || []).map(i =>
            `<div style="padding:4px 0">• ${i.task}${i.owner ? ' → <b>'+i.owner+'</b>' : ''}${i.deadline ? ' ('+i.deadline+')' : ''} <span class="badge badge-orange">${i.priority}</span></div>`).join('');

        document.getElementById('result').innerHTML = `
            <div class="result">
                <h3>✅ ${data.filename}</h3>
                <div class="field"><span class="field-name">Type</span><span class="field-value">${a.document_type}</span></div>
                <div class="field"><span class="field-name">Sentiment</span><span class="field-value">${a.sentiment}</span></div>
                <div class="field"><span class="field-name">Words</span><span class="field-value">${a.word_count}</span></div>
                <div class="field"><span class="field-name">Confidence</span><span class="field-value">${(a.confidence_score*100).toFixed(0)}%</span></div>
                <div class="field"><span class="field-name">Processing</span><span class="field-value">${data.processing_time_ms.toFixed(0)}ms</span></div>
                <div class="field"><span class="field-name">Summary</span><span class="field-value">${a.summary}</span></div>
                <div class="field"><span class="field-name">Key Terms</span><span class="field-value">${terms || 'None found'}</span></div>
                ${entities ? `<div class="field"><span class="field-name">Entities</span><span class="field-value">${entities}</span></div>` : ''}
                ${actions ? `<div class="field"><span class="field-name">Action Items</span><span class="field-value">${actions}</span></div>` : ''}
            </div>`;
        loadDocs();
    } catch(e) {
        document.getElementById('result').innerHTML =
            `<div class="result"><h3 style="color:#f85149">❌ Error</h3><p>${e.message}</p></div>`;
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

async function loadDocs() {
    const resp = await fetch('/documents');
    const docs = await resp.json();
    const div = document.getElementById('docs');
    if (!docs.length) {
        div.innerHTML = '<p style="color:#8b949e">No documents yet. Upload one above!</p>';
        return;
    }
    div.innerHTML = docs.map(d => `
        <div class="doc-card">
            <h4>${d.filename}</h4>
            <p>${d.document_type} · ${d.sentiment} · ${d.word_count} words · ${d.processed_at.split('T')[0]}</p>
            <p style="margin-top:6px;color:#6e7681">${d.summary ? d.summary.substring(0,120)+'...' : ''}</p>
        </div>`).join('');
}

loadDocs();
</script>
</body>
</html>""")


# ─────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    allowed = [".pdf", ".docx", ".txt"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file '{ext}'. Use: {allowed}")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(save_path)
    start_time = time.time()

    text, file_type = parse_document(str(save_path))

    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Could not extract text from document.")

    try:
        analysis = extract_from_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    processing_time = (time.time() - start_time) * 1000

    result = {
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": file_type,
        "file_size_bytes": file_size,
        "processed_at": datetime.now().isoformat(),
        "raw_text_preview": text[:300] + "..." if len(text) > 300 else text,
        "analysis": analysis.model_dump(),
        "processing_time_ms": round(processing_time, 2)
    }

    save_result(result)
    return result


@app.get("/documents")
async def list_documents():
    return get_all_documents()


@app.get("/documents/{doc_id}")
async def get_single_document(doc_id: str):
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    import sqlite3
    from database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"deleted": doc_id}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─────────────────────────────────────────────────────────
# SAMPLE DOCS FOR TESTING
# ─────────────────────────────────────────────────────────

def create_sample_docs():
    samples = {
        "meeting_notes.txt": """Project Sync — July 1, 2026
Attendees: Ali Khan (PM), Sara Ahmed (Dev), Kamran Malik (Design)

Key Decisions:
- Launch MVP by July 15, 2026
- Budget approved: PKR 500,000 for Q3

Action Items:
- Ali: Complete API documentation by July 8
- Sara: Fix authentication bug — CRITICAL — due July 5
- Kamran: Submit UI mockups by July 7

Next meeting: July 8, 2026 at 10 AM.""",

        "job_posting.txt": """Senior AI Engineer — CalderR Technologies, Islamabad

We are looking for a talented AI engineer to join our growing team.
Requirements:
- 4+ years Python experience
- Strong knowledge of LangChain, LLMs, RAG systems
- Experience with FastAPI, PostgreSQL, Docker
- Nice to have: LangGraph, Pydantic v2, React

Salary: PKR 350,000 - 550,000/month
Hybrid work model (3 days office)
Full-time position. Apply by July 20, 2026.""",

        "q2_report.txt": """Q2 2026 Performance Report — TechVentures Pakistan

Executive Summary:
Q2 exceeded targets across all metrics. Revenue grew 34% YoY to PKR 12.5M.
Customer satisfaction reached 94%, up from 87% in Q1.

Key Achievements:
- Launched AI-powered product suite in April 2026
- Expanded to 3 new cities: Islamabad, Lahore, Karachi
- Team grew from 45 to 67 employees

Challenges:
Server downtime in May caused 2% revenue loss.
Recommend infrastructure investment of PKR 2M in Q3."""
    }

    for filename, content in samples.items():
        path = UPLOAD_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    return list(samples.keys())


if __name__ == "__main__":
    import uvicorn
    from rich.console import Console
    from rich.panel import Panel

    c = Console()
    c.print(Panel.fit(
        "[bold cyan]📄 Document Processing Pipeline — Project 2-P-A[/bold cyan]\n"
        "[dim]FastAPI + LLM Extraction + Pydantic v2 + SQLite[/dim]",
        border_style="cyan"
    ))

    samples = create_sample_docs()
    c.print(f"[green]✅ Created {len(samples)} sample docs in uploads/[/green]")
    c.print("[yellow]📡 Starting server...[/yellow]")
    c.print("[dim]Open: http://localhost:8000[/dim]")
    c.print("[dim]API docs: http://localhost:8000/docs[/dim]")
    c.print("[dim]Ctrl+C to stop[/dim]\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")