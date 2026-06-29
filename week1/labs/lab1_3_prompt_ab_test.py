"""
Lab 1.3: Prompt Engineering A/B Test
Task: News summarization with 5 different system prompts
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

NEWS_ARTICLE = """
KARACHI, Pakistan - Scientists at the University of Karachi announced a breakthrough 
in solar energy storage technology. The new system, developed over 5 years, can store 
solar energy for up to 30 days using a novel chemical process involving salt compounds.
The technology could reduce electricity costs by 40% in developing countries. 
The team, led by Dr. Ayesha Rahman, plans to begin pilot testing in rural Sindh by 2027.
International investors have pledged $50 million for commercialization.
"""

PROMPTS = {
    "Prompt A — Basic": "Summarize the following news article.",

    "Prompt B — Length Constrained": "Summarize this news article in exactly 2 sentences. No more, no less.",

    "Prompt C — Structured Output": """Summarize this news article using this exact format:
HEADLINE: (one punchy line)
KEY FACTS: (3 bullet points)
IMPACT: (one sentence on why it matters)""",

    "Prompt D — Audience Specific": """You write for busy executives who read on mobile.
Summarize in under 50 words. Lead with the most important business implication.
Use active voice. No jargon.""",

    "Prompt E — Chain-of-Thought": """First, identify: Who, What, When, Where, Why.
Then identify the most newsworthy element.
Then write a crisp 3-sentence summary prioritizing newsworthiness.
Show your thinking process."""
}

def evaluate_summary(summary: str, prompt_name: str) -> dict:
    """Score each summary on key dimensions"""
    word_count = len(summary.split())
    has_numbers = any(char.isdigit() for char in summary)
    has_structure = any(marker in summary for marker in ["•", "-", ":", "\n"])

    return {
        "word_count": word_count,
        "has_key_facts": has_numbers,
        "is_structured": has_structure,
        "conciseness": "High" if word_count < 60 else "Medium" if word_count < 100 else "Low"
    }

def run_ab_test():
    console.print(Panel.fit(
        "[bold cyan]🧪 Lab 1.3: Prompt Engineering A/B Test[/bold cyan]\n"
        "[dim]Task: News Summarization | 5 Prompts | Same Article[/dim]",
        border_style="cyan"
    ))

    results = {}

    for prompt_name, system_prompt in PROMPTS.items():
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Article:\n{article}")
        ])
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"article": NEWS_ARTICLE})
        metrics = evaluate_summary(summary, prompt_name)
        results[prompt_name] = {"summary": summary, "metrics": metrics}

        console.print(f"\n[bold yellow]{prompt_name}[/bold yellow]")
        console.print(f"[dim]{system_prompt[:80]}...[/dim]" if len(system_prompt) > 80 else f"[dim]{system_prompt}[/dim]")
        console.print(Panel(summary, border_style="blue"))
        console.print(f"[dim]Words: {metrics['word_count']} | "
                     f"Has facts: {metrics['has_key_facts']} | "
                     f"Structured: {metrics['is_structured']} | "
                     f"Conciseness: {metrics['conciseness']}[/dim]")

    # Comparison table
    console.print("\n" + "="*60)
    console.print("[bold]📊 COMPARISON TABLE[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Prompt", style="yellow", width=20)
    table.add_column("Words", justify="center")
    table.add_column("Has Facts", justify="center")
    table.add_column("Structured", justify="center")
    table.add_column("Conciseness", justify="center")

    for name, data in results.items():
        m = data["metrics"]
        table.add_row(
            name.split("—")[0].strip(),
            str(m["word_count"]),
            "✅" if m["has_key_facts"] else "❌",
            "✅" if m["is_structured"] else "❌",
            m["conciseness"]
        )

    console.print(table)

    # Save findings
    with open("week1/labs/lab1_3_findings.json", "w") as f:
        json.dump({k: {**v, "summary": v["summary"]} for k, v in results.items()}, f, indent=2)

    console.print("\n[green]✅ Findings saved to lab1_3_findings.json[/green]")
    console.print("\n[bold]Key Takeaway:[/bold] Structured prompts (C, D, E) give more consistent, usable output.")
    console.print("For production: always specify format, length, and audience in your system prompt!")


if __name__ == "__main__":
    run_ab_test()