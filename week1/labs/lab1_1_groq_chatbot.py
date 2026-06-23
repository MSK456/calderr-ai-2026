"""
Lab 1.1: Groq CLI Chatbot
Features: conversation history, /clear, /exit, token usage display
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
console = Console()

SYSTEM_PROMPT = """You are an expert AI Engineering assistant for the CalderR internship.
You specialize in: Python, LangChain, Groq API, LLMs, and Agentic AI.
Be concise but thorough. Use examples when helpful."""

def display_banner():
    console.print(Panel.fit(
        "[bold cyan]🤖 CalderR AI Assistant[/bold cyan]\n"
        "[dim]Powered by Groq llama-3.1-8b-instant[/dim]\n\n"
        "[yellow]Commands:[/yellow] /clear → reset chat | /exit → quit | /history → show chat",
        border_style="cyan"
    ))

def display_token_usage(usage):
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[dim]Prompt tokens:[/dim]", f"[cyan]{usage.prompt_tokens}[/cyan]")
    table.add_row("[dim]Response tokens:[/dim]", f"[green]{usage.completion_tokens}[/green]")
    table.add_row("[dim]Total tokens:[/dim]", f"[yellow]{usage.total_tokens}[/yellow]")
    console.print(table)

def display_history(history):
    if len(history) <= 1:
        console.print("[dim]No conversation history yet.[/dim]")
        return
    for msg in history[1:]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            console.print(f"[bold blue]You:[/bold blue] {content}")
        else:
            console.print(f"[bold green]AI:[/bold green] {content[:100]}...")

def chat():
    display_banner()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_tokens_used = 0

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()

            if not user_input:
                continue

            # Commands
            if user_input.lower() == "/exit":
                console.print(f"\n[yellow]👋 Session ended. Total tokens used: {total_tokens_used}[/yellow]")
                break

            if user_input.lower() == "/clear":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                total_tokens_used = 0
                console.clear()
                display_banner()
                console.print("[green]✅ Conversation cleared![/green]")
                continue

            if user_input.lower() == "/history":
                display_history(history)
                continue

            # Add user message
            history.append({"role": "user", "content": user_input})

            # Get AI response
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=history,
                    temperature=0.7,
                    max_tokens=500
                )

            ai_message = response.choices[0].message.content
            history.append({"role": "assistant", "content": ai_message})

            total_tokens_used += response.usage.total_tokens

            # Display response
            console.print(Panel(
                ai_message,
                title="[bold green]🤖 Assistant[/bold green]",
                border_style="green"
            ))

            # Token usage
            display_token_usage(response.usage)
            console.print(f"[dim]Session total: {total_tokens_used} tokens | History: {len(history)-1} messages[/dim]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit properly.[/yellow]")
            continue


if __name__ == "__main__":
    chat()