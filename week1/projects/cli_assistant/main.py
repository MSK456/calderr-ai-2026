"""
Project 1-I-A: Intelligent CLI Assistant
Features: topic switching, 10+ turn memory, Rich UI, error handling, typed code
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich import box

load_dotenv()

console = Console()

TOPICS = {
    "cooking": {
        "emoji": "👨‍🍳",
        "system": "You are a world-class chef and culinary expert. Help with recipes, techniques, and food science. Be enthusiastic about food. Use cooking terminology appropriately."
    },
    "history": {
        "emoji": "📜",
        "system": "You are a passionate historian with expertise across all eras. Connect historical events to modern context. Use vivid storytelling. Cite specific dates and figures."
    },
    "programming": {
        "emoji": "💻",
        "system": "You are a senior software engineer with 15 years experience. Give practical, production-ready advice. Always include code examples. Mention edge cases and best practices."
    },
    "ai": {
        "emoji": "🤖",
        "system": "You are an AI research scientist. Explain complex ML/AI concepts clearly. Reference real papers and techniques. Be precise about what models can and cannot do."
    },
    "general": {
        "emoji": "🌟",
        "system": "You are a brilliant generalist assistant. Be helpful, accurate, and concise."
    }
}


class CLIAssistant:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.current_topic = "general"
        self.history: list[dict] = []
        self.total_tokens = 0
        self.turn_count = 0
        self._reset_history()

    def _reset_history(self):
        topic = TOPICS[self.current_topic]
        self.history = [{"role": "system", "content": topic["system"]}]

    def _get_response(self, user_input: str) -> Optional[dict]:
        self.history.append({"role": "user", "content": user_input})
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=self.history,
                temperature=0.7,
                max_tokens=600
            )
            ai_msg = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": ai_msg})
            self.total_tokens += response.usage.total_tokens
            self.turn_count += 1
            return {"message": ai_msg, "usage": response.usage}
        except Exception as e:
            self.history.pop()  # Remove failed user message
            return {"error": str(e)}

    def switch_topic(self, topic: str) -> str:
        if topic not in TOPICS:
            return f"Unknown topic. Available: {', '.join(TOPICS.keys())}"
        self.current_topic = topic
        self._reset_history()
        self.turn_count = 0
        emoji = TOPICS[topic]["emoji"]
        return f"{emoji} Switched to [bold]{topic}[/bold] mode! Previous history cleared."

    def display_banner(self):
        console.clear()
        topic = TOPICS[self.current_topic]
        console.print(Panel.fit(
            f"[bold cyan]🧠 CalderR Intelligent CLI Assistant[/bold cyan]\n"
            f"[dim]Current Topic: {topic['emoji']} {self.current_topic.title()}[/dim]\n\n"
            f"[yellow]/topic <name>[/yellow]  Switch topic (cooking/history/programming/ai/general)\n"
            f"[yellow]/clear[/yellow]         Clear conversation\n"
            f"[yellow]/history[/yellow]       Show conversation summary\n"
            f"[yellow]/stats[/yellow]         Show session statistics\n"
            f"[yellow]/exit[/yellow]          Quit",
            border_style="cyan",
            box=box.ROUNDED
        ))

    def show_stats(self):
        table = Table(title="📊 Session Statistics", box=box.SIMPLE_HEAVY)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        table.add_row("Current Topic", f"{TOPICS[self.current_topic]['emoji']} {self.current_topic}")
        table.add_row("Total Turns", str(self.turn_count))
        table.add_row("Total Tokens", str(self.total_tokens))
        table.add_row("Messages in History", str(len(self.history) - 1))
        console.print(table)

    def show_history(self):
        if len(self.history) <= 1:
            console.print("[dim]No conversation yet.[/dim]")
            return
        for i, msg in enumerate(self.history[1:], 1):
            role = msg["role"]
            content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
            prefix = "[bold blue]You:[/bold blue]" if role == "user" else "[bold green]AI:[/bold green]"
            console.print(f"{i}. {prefix} {content}")

    def run(self):
        self.display_banner()
        console.print(f"\n[dim]Type your message below. Start asking about {self.current_topic}![/dim]\n")

        while True:
            try:
                user_input = Prompt.ask(
                    f"[bold blue]{TOPICS[self.current_topic]['emoji']} You[/bold blue]"
                ).strip()

                if not user_input:
                    continue

                # Commands
                if user_input.lower() == "/exit":
                    self.show_stats()
                    console.print("\n[yellow]👋 Goodbye! Keep building![/yellow]")
                    sys.exit(0)

                elif user_input.lower() == "/clear":
                    self._reset_history()
                    self.turn_count = 0
                    self.display_banner()
                    console.print("[green]✅ Cleared![/green]")
                    continue

                elif user_input.lower().startswith("/topic"):
                    parts = user_input.split()
                    if len(parts) < 2:
                        console.print(f"[yellow]Usage: /topic <name> | Options: {', '.join(TOPICS.keys())}[/yellow]")
                    else:
                        result = self.switch_topic(parts[1].lower())
                        console.print(f"[green]{result}[/green]")
                    continue

                elif user_input.lower() == "/history":
                    self.show_history()
                    continue

                elif user_input.lower() == "/stats":
                    self.show_stats()
                    continue

                # Get AI response
                with console.status("[dim]⚡ Groq is thinking...[/dim]", spinner="point"):
                    result = self._get_response(user_input)

                if "error" in result:
                    console.print(Panel(
                        f"[red]❌ Error: {result['error']}[/red]\n[dim]Please try again.[/dim]",
                        border_style="red"
                    ))
                    continue

                # Display response as markdown
                console.print(Panel(
                    Markdown(result["message"]),
                    title=f"[bold green]{TOPICS[self.current_topic]['emoji']} Assistant[/bold green]",
                    border_style="green",
                    box=box.ROUNDED
                ))

                # Token info
                usage = result["usage"]
                console.print(
                    f"[dim]🪙 Tokens: {usage.total_tokens} | "
                    f"Turn {self.turn_count} | "
                    f"Session total: {self.total_tokens}[/dim]"
                )

            except KeyboardInterrupt:
                console.print("\n[dim]Tip: Use /exit to quit properly[/dim]")


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]❌ GROQ_API_KEY not found in .env file![/red]")
        sys.exit(1)

    assistant = CLIAssistant()
    assistant.run()