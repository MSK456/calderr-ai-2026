"""Pydantic v2 models for document extraction"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    text: Optional[str] = Field(default="Unknown")
    entity_type: Literal["person", "organization", "location", "date", "money", "other"] = Field(
        default="other"
    )


class ActionItem(BaseModel):
    task: Optional[str] = Field(default="Unspecified task")
    owner: Optional[str] = Field(default=None)
    deadline: Optional[str] = Field(default=None)
    priority: Literal["low", "medium", "high"] = Field(default="medium")


class DocumentAnalysis(BaseModel):
    title: Optional[str] = Field(default="Untitled")
    document_type: Literal[
        "report", "email", "contract", "meeting_notes",
        "resume", "invoice", "technical_doc", "other"
    ] = Field(default="other")
    language: Optional[str] = Field(default="English")
    word_count: int = Field(default=0)
    summary: Optional[str] = Field(default="No summary available")
    key_terms: List[str] = Field(default=[])
    entities: List[ExtractedEntity] = Field(default=[])
    dates_mentioned: List[str] = Field(default=[])
    action_items: List[ActionItem] = Field(default=[])
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(default="neutral")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = {"str_strip_whitespace": True}