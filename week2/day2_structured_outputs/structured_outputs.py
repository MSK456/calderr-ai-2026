"""
Day 2: Structured Outputs — Pydantic v2 + with_structured_output()
"""

import os
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()
console = Console()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ─────────────────────────────────────────────────────────
# PYDANTIC V2 MODELS — 5 different schemas
# ─────────────────────────────────────────────────────────

class SentimentAnalysis(BaseModel):
    """Model 1: Sentiment analysis result"""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the text"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    key_phrases: List[str] = Field(
        description="Key phrases that influenced the sentiment",
        max_length=5
    )
    reasoning: str = Field(description="One sentence explaining the sentiment")

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


class PersonProfile(BaseModel):
    """Model 2: Extract person info from text"""
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, ge=0, le=150, description="Age in years")
    occupation: str = Field(description="Job or role")
    skills: List[str] = Field(description="List of skills or expertise areas")
    location: Optional[str] = Field(default=None, description="City or country")
    is_student: bool = Field(description="True if person is a student")

    model_config = {"str_strip_whitespace": True}


class MeetingNotes(BaseModel):
    """Model 3: Extract structured data from meeting notes"""
    meeting_title: str = Field(description="Title or topic of the meeting")
    attendees: List[str] = Field(description="List of people who attended")
    decisions_made: List[str] = Field(description="Key decisions taken in the meeting")
    action_items: List[str] = Field(description="Tasks assigned with owner if mentioned")
    next_meeting_date: Optional[str] = Field(
        default=None,
        description="Next meeting date if mentioned, as written in notes"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall priority level of meeting outcomes"
    )


class CodeReviewResult(BaseModel):
    """Model 4: Structured code review output"""
    has_bugs: bool = Field(description="True if bugs were found")
    bugs: List[str] = Field(default=[], description="List of bugs found")
    security_issues: List[str] = Field(default=[], description="Security vulnerabilities")
    style_violations: List[str] = Field(default=[], description="PEP8 or style issues")
    performance_tips: List[str] = Field(default=[], description="Performance improvements")
    overall_score: int = Field(ge=1, le=10, description="Code quality score 1-10")
    summary: str = Field(description="One sentence overall assessment")

    @model_validator(mode="after")
    def bugs_must_exist_if_has_bugs(self) -> "CodeReviewResult":
        if self.has_bugs and not self.bugs:
            raise ValueError("If has_bugs is True, bugs list must not be empty")
        return self


class NewsArticle(BaseModel):
    """Model 5: Extract structured data from news articles"""
    headline: str = Field(description="Article headline")
    topic_category: Literal["tech", "politics", "business", "science", "sports", "other"] = Field(
        description="Main topic category"
    )
    entities_mentioned: List[str] = Field(description="People, companies, places mentioned")
    key_facts: List[str] = Field(description="Top 3 key facts from the article", max_length=3)
    sentiment_toward_subject: Literal["positive", "negative", "neutral"] = Field(
        description="Article's tone toward the main subject"
    )
    is_breaking_news: bool = Field(description="True if this appears to be breaking/urgent news")


# ─────────────────────────────────────────────────────────
# EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────────────────

def extract_sentiment(text: str) -> SentimentAnalysis:
    structured_llm = llm.with_structured_output(SentimentAnalysis)
    return structured_llm.invoke(f"Analyze the sentiment of this text:\n\n{text}")


def extract_person(text: str) -> PersonProfile:
    structured_llm = llm.with_structured_output(PersonProfile)
    return structured_llm.invoke(f"Extract person information from this text:\n\n{text}")


def extract_meeting_notes(notes: str) -> MeetingNotes:
    structured_llm = llm.with_structured_output(MeetingNotes)
    return structured_llm.invoke(f"Extract structured information from these meeting notes:\n\n{notes}")


def review_code(code: str) -> CodeReviewResult:
    structured_llm = llm.with_structured_output(CodeReviewResult)
    prompt = f"""Review this Python code for bugs, security issues, style violations, and performance.
Be thorough but concise in each finding.

Code:
````python
{code}
```"""
    return structured_llm.invoke(prompt)


def extract_news(article: str) -> NewsArticle:
    structured_llm = llm.with_structured_output(NewsArticle)
    return structured_llm.invoke(f"Extract structured information from this news article:\n\n{article}")


# ─────────────────────────────────────────────────────────
# DEMO + DISPLAY
# ─────────────────────────────────────────────────────────

def demo_all():
    console.print("\n[bold cyan]🏗️  Day 2: Structured Output Extraction[/bold cyan]\n")

    # Test 1: Sentiment
    console.print("[bold yellow]Test 1: Sentiment Analysis[/bold yellow]")
    text = "The new iPhone is absolutely stunning! The camera is incredible but the price is way too high for most people."
    sentiment = extract_sentiment(text)
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Sentiment", sentiment.sentiment)
    table.add_row("Confidence", str(sentiment.confidence))
    table.add_row("Key Phrases", ", ".join(sentiment.key_phrases))
    table.add_row("Reasoning", sentiment.reasoning)
    console.print(table)

    # Test 2: Person
    console.print("\n[bold yellow]Test 2: Person Profile Extraction[/bold yellow]")
    bio = "Muhammad Shaheed, 20, is a CS student at FAST University Islamabad. He's skilled in Python, ROS2, YOLOv8, and LangChain. He loves building robots and AI systems."
    person = extract_person(bio)
    console.print(f"Name: {person.name} | Age: {person.age} | Student: {person.is_student}")
    console.print(f"Skills: {', '.join(person.skills)}")

    # Test 3: Meeting Notes
    console.print("\n[bold yellow]Test 3: Meeting Notes Extraction[/bold yellow]")
    notes = """
    Product sync - July 1 2026. Attendees: Ali, Sara, Kamran, Fatima.
    We decided to launch the MVP by July 15. Ali will finish the backend API by July 8.
    Sara handles UI design by July 10. Kamran needs to write test cases.
    The launch is CRITICAL — client demo is on July 16. Next sync: July 8.
    """
    meeting = extract_meeting_notes(notes)
    console.print(f"Title: {meeting.meeting_title} | Priority: [red]{meeting.priority}[/red]")
    console.print(f"Decisions: {meeting.decisions_made}")
    console.print(f"Action Items: {meeting.action_items}")

    # Test 4: Code Review
    console.print("\n[bold yellow]Test 4: Code Review[/bold yellow]")
    bad_code = """
def get_user(user_id):
    import sqlite3
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE id = " + user_id
    result = conn.execute(query)
    password = result.fetchone()[3]
    print("Password is: " + password)
    return result.fetchall()
"""
    review = review_code(bad_code)
    console.print(f"Score: [red]{review.overall_score}/10[/red]")
    console.print(f"Bugs: {review.bugs}")
    console.print(f"Security: {review.security_issues}")
    console.print(f"Summary: {review.summary}")

    # Test 5: News Article
    console.print("\n[bold yellow]Test 5: News Article Extraction[/bold yellow]")
    article = """
    BREAKING: OpenAI announces GPT-5 releasing next month with 10x performance improvement.
    CEO Sam Altman confirmed in a tweet that the model will be available to ChatGPT Plus users first.
    Google's DeepMind team responded saying their Gemini Ultra 2.0 surpasses GPT-5 benchmarks.
    Microsoft, which invested $13B in OpenAI, saw its stock jump 4% on the news.
    """
    news = extract_news(article)
    console.print(f"Headline: {news.headline}")
    console.print(f"Category: {news.topic_category} | Breaking: {news.is_breaking_news}")
    console.print(f"Entities: {', '.join(news.entities_mentioned)}")

    console.print("\n[green]✅ Day 2 Complete! Pydantic structured outputs mastered.[/green]")


if __name__ == "__main__":
    demo_all()
