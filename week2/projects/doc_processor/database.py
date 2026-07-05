"""SQLite database for storing processing results"""

import sqlite3
import json
import os

DB_PATH = "week2/projects/doc_processor/documents.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT,
            file_size_bytes INTEGER,
            processed_at TEXT,
            word_count INTEGER,
            document_type TEXT,
            summary TEXT,
            key_terms TEXT,
            entities TEXT,
            action_items TEXT,
            sentiment TEXT,
            confidence_score REAL,
            processing_time_ms REAL,
            raw_text_preview TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_result(result: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    analysis = result.get("analysis", {})
    cursor.execute("""
        INSERT OR REPLACE INTO documents VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        result["document_id"],
        result["filename"],
        result["file_type"],
        result["file_size_bytes"],
        result["processed_at"],
        analysis.get("word_count", 0),
        analysis.get("document_type", "other"),
        analysis.get("summary", ""),
        json.dumps(analysis.get("key_terms", [])),
        json.dumps(analysis.get("entities", [])),
        json.dumps(analysis.get("action_items", [])),
        analysis.get("sentiment", "neutral"),
        analysis.get("confidence_score", 0.0),
        result["processing_time_ms"],
        result["raw_text_preview"]
    ))
    conn.commit()
    conn.close()


def get_all_documents() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY processed_at DESC")
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        d["key_terms"] = json.loads(d["key_terms"] or "[]")
        d["entities"] = json.loads(d["entities"] or "[]")
        d["action_items"] = json.loads(d["action_items"] or "[]")
        results.append(d)
    return results


def get_document(doc_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["key_terms"] = json.loads(d["key_terms"] or "[]")
    d["entities"] = json.loads(d["entities"] or "[]")
    d["action_items"] = json.loads(d["action_items"] or "[]")
    return d