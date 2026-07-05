"""LLM extraction using structured output with json_mode"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from models import DocumentAnalysis

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

structured_llm = llm.with_structured_output(DocumentAnalysis, method="json_mode")

EXTRACTION_PROMPT = """You are a document analysis expert.
Analyze the document and return a JSON object with these exact fields:
{
  "title": string or null,
  "document_type": "report"|"email"|"contract"|"meeting_notes"|"resume"|"invoice"|"technical_doc"|"other",
  "language": string,
  "word_count": integer,
  "summary": string (2-3 sentences),
  "key_terms": [list of strings, max 8],
  "entities": [{"text": string, "entity_type": "person"|"organization"|"location"|"date"|"money"|"other"}],
  "dates_mentioned": [list of date strings],
  "action_items": [{"task": string, "owner": string or null, "deadline": string or null, "priority": "low"|"medium"|"high"}],
  "sentiment": "positive"|"negative"|"neutral"|"mixed",
  "confidence_score": float 0.0-1.0
}
Return ONLY the JSON object."""


def extract_from_text(text: str, max_chars: int = 4000) -> DocumentAnalysis:
    if len(text) > max_chars:
        half = max_chars // 2
        truncated = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
    else:
        truncated = text

    word_count = len(text.split())

    result = structured_llm.invoke([
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"Analyze this document:\n\n{truncated}")
    ])

    result.word_count = word_count
    return result