"""
Lab 2.1: Structured Output Extractor — Job Posting Parser
"""

import os
import json
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()
console = Console()


# ── Pydantic Models ───────────────────────────────────────

class SalaryRange(BaseModel):
    min_amount: Optional[int] = Field(default=None)
    max_amount: Optional[int] = Field(default=None)
    currency: Optional[str] = Field(default="USD")
    period: Literal["monthly", "yearly", "hourly"] = Field(default="yearly")

    @field_validator("currency", mode="before")
    @classmethod
    def uppercase_currency(cls, v) -> str:
        if v is None:
            return "USD"
        return str(v).upper()


class JobPosting(BaseModel):
    title: Optional[str] = Field(default="Unknown", description="Exact job title")
    company: Optional[str] = Field(default="Unknown", description="Company name")
    salary: Optional[SalaryRange] = Field(default=None)
    required_skills: List[str] = Field(default=[], description="Required technical skills")
    nice_to_have_skills: List[str] = Field(default=[])
    location: Optional[str] = Field(default="Remote", description="City, country, or Remote")
    is_remote: bool = Field(default=False, description="True if remote work offered")
    is_hybrid: bool = Field(default=False, description="True if hybrid work offered")
    experience_years: Optional[int] = Field(default=None)
    job_type: Literal["full-time", "part-time", "contract", "internship", "freelance"] = Field(default="full-time")
    seniority: Literal["junior", "mid", "senior", "lead", "manager", "not_specified"] = Field(default="not_specified")

    model_config = {"str_strip_whitespace": True}


# ── LLM + Extractor ──────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

structured_llm = llm.with_structured_output(JobPosting, method="json_mode")

SYSTEM_PROMPT = """You are an expert job posting parser.
Extract ALL information accurately from job postings.
Return a valid JSON object matching this exact schema:
{
  "title": string,
  "company": string,
  "location": string,
  "is_remote": boolean,
  "is_hybrid": boolean,
  "experience_years": integer or null,
  "job_type": "full-time" | "part-time" | "contract" | "internship" | "freelance",
  "seniority": "junior" | "mid" | "senior" | "lead" | "manager" | "not_specified",
  "required_skills": [list of strings],
  "nice_to_have_skills": [list of strings],
  "salary": {
    "min_amount": integer or null,
    "max_amount": integer or null,
    "currency": string,
    "period": "monthly" | "yearly" | "hourly"
  } or null
}
Return ONLY the JSON object, no explanation, no markdown."""


def extract_job_posting(raw_text: str) -> JobPosting:
    return structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract from this job posting:\n\n{raw_text}")
    ])


# ── Test Cases ────────────────────────────────────────────

TEST_POSTINGS = [
    """Senior Python Developer — TechCorp Islamabad
    We need 5+ years Python experience. Must know: Django, PostgreSQL, Docker, AWS.
    Nice to have: React, Redis, Kubernetes.
    Salary: PKR 400,000 - 600,000/month. Hybrid work (3 days office).
    Full-time position.""",

    """ML Engineer Intern — AI Startup (Remote)
    Fresh graduates welcome! Required: Python basics, NumPy, familiarity with ML concepts.
    Bonus: PyTorch, scikit-learn experience.
    Unpaid internship, 3 months. 100% remote.""",

    """Lead DevOps Engineer — FinTech Ltd London
    10+ years experience. Must have: Kubernetes, Terraform, AWS/GCP, CI/CD, Linux.
    Salary: £80,000 - £110,000 per year.
    On-site London. Senior position.""",

    """Junior Frontend Developer wanted!
    React, TypeScript, CSS required. Git knowledge needed.
    2 years experience preferred but not required.
    Rs 150,000/month. Karachi office. Full-time.""",

    """Data Scientist — Contract Role (6 months)
    Skills: Python, Pandas, ML, SQL, Power BI.
    Nice to have: Spark, Snowflake.
    $50-70/hour. Fully remote. Mid-level role.""",
]


def display_result(posting: JobPosting, index: int):
    table = Table(title=f"Job #{index+1}: {posting.title}", box=box.ROUNDED)
    table.add_column("Field", style="cyan", width=22)
    table.add_column("Value", style="green")

    salary_str = "Not mentioned"
    if posting.salary:
        s = posting.salary
        salary_str = (f"{s.min_amount}-{s.max_amount} {s.currency}/{s.period}"
                      if s.min_amount else f"{s.currency} (amount not specified)")

    table.add_row("Company", posting.company)
    table.add_row("Location", posting.location)
    table.add_row("Type", posting.job_type)
    table.add_row("Seniority", posting.seniority)
    table.add_row("Experience", f"{posting.experience_years}+ yrs" if posting.experience_years else "Not specified")
    table.add_row("Remote", "✅ Yes" if posting.is_remote else "❌ No")
    table.add_row("Hybrid", "✅ Yes" if posting.is_hybrid else "❌ No")
    table.add_row("Salary", salary_str)
    table.add_row("Required Skills", ", ".join(posting.required_skills))
    table.add_row("Nice-to-Have", ", ".join(posting.nice_to_have_skills) or "None")
    console.print(table)


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]🔬 Lab 2.1: Structured Job Posting Extractor[/bold cyan]\n"
        "[dim]Pydantic v2 + json_mode | 5 test cases[/dim]",
        border_style="cyan"
    ))

    results = []
    for i, posting in enumerate(TEST_POSTINGS):
        console.print(f"\n[dim]Extracting posting {i+1}/5...[/dim]")
        with console.status("[yellow]Processing...[/yellow]"):
            extracted = extract_job_posting(posting)
        display_result(extracted, i)
        results.append(extracted.model_dump())

    with open("week2/labs/lab2_1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print("\n[green]✅ Results saved to lab2_1_results.json[/green]")
    console.print("[green]✅ Lab 2.1 Complete![/green]")