"""
Project 1-P-B: Multi-Model Comparison Engine
Benchmarks Groq models: speed, quality, consistency
Uses: asyncio, YAML, statistical analysis, HTML report
"""

import os
import asyncio
import time
import yaml
import json
import statistics
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from groq import AsyncGroq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

load_dotenv()

console = Console()

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]


@dataclass
class TaskResult:
    model: str
    task_id: str
    category: str
    difficulty: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    keyword_hits: int
    total_keywords: int
    error: Optional[str] = None

    @property
    def accuracy_score(self) -> float:
        if self.total_keywords == 0:
            return 1.0
        return self.keyword_hits / self.total_keywords

    @property
    def response_length(self) -> int:
        return len(self.response.split())


@dataclass
class ModelStats:
    model: str
    results: list[TaskResult] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        latencies = [r.latency_ms for r in self.results if not r.error]
        return statistics.mean(latencies) if latencies else 0

    @property
    def avg_accuracy(self) -> float:
        scores = [r.accuracy_score for r in self.results if not r.error]
        return statistics.mean(scores) if scores else 0

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results if not r.error)

    @property
    def error_rate(self) -> float:
        errors = sum(1 for r in self.results if r.error)
        return errors / len(self.results) if self.results else 0

    @property
    def avg_response_length(self) -> float:
        lengths = [r.response_length for r in self.results if not r.error]
        return statistics.mean(lengths) if lengths else 0


async def run_single_task(
    client: AsyncGroq,
    model: str,
    task: dict,
    semaphore: asyncio.Semaphore
) -> TaskResult:
    async with semaphore:
        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Be precise and concise."},
                    {"role": "user", "content": task["prompt"]}
                ],
                temperature=0.3,
                max_tokens=400
            )
            latency = (time.time() - start) * 1000
            content = response.choices[0].message.content

            # Score keyword hits
            keywords = task.get("expected_keywords", [])
            hits = sum(1 for kw in keywords if kw.lower() in content.lower())

            return TaskResult(
                model=model,
                task_id=task["id"],
                category=task["category"],
                difficulty=task["difficulty"],
                response=content,
                latency_ms=latency,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                keyword_hits=hits,
                total_keywords=len(keywords)
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return TaskResult(
                model=model,
                task_id=task["id"],
                category=task["category"],
                difficulty=task["difficulty"],
                response="",
                latency_ms=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                keyword_hits=0,
                total_keywords=len(task.get("expected_keywords", [])),
                error=str(e)
            )


async def run_benchmark(tasks: list[dict]) -> dict[str, ModelStats]:
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
    model_stats = {model: ModelStats(model=model) for model in MODELS}

    all_jobs = [
        (model, task)
        for model in MODELS
        for task in tasks
    ]

    console.print(f"\n[cyan]🚀 Running {len(all_jobs)} benchmark jobs...[/cyan]")
    console.print(f"[dim]{len(MODELS)} models × {len(tasks)} tasks[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task_progress = progress.add_task("Benchmarking...", total=len(all_jobs))

        coroutines = []
        for model, task in all_jobs:
            coro = run_single_task(client, model, task, semaphore)
            coroutines.append((model, coro))

        async def run_and_update(model, coro):
            result = await coro
            progress.advance(task_progress)
            return model, result

        results = await asyncio.gather(*[
            run_and_update(model, coro)
            for model, coro in coroutines
        ])

    for model, result in results:
        model_stats[model].results.append(result)

    await client.close()
    return model_stats


def display_results(model_stats: dict[str, ModelStats]):
    console.print("\n" + "="*70)
    console.print("[bold cyan]📊 BENCHMARK RESULTS[/bold cyan]")
    console.print("="*70)

    # Main comparison table
    table = Table(
        title="Model Performance Summary",
        box=box.ROUNDED,
        header_style="bold cyan"
    )
    table.add_column("Model", style="yellow", width=28)
    table.add_column("Avg Latency", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Avg Words", justify="center")
    table.add_column("Total Tokens", justify="center")
    table.add_column("Errors", justify="center")

    ranked = sorted(model_stats.values(), key=lambda x: x.avg_latency)

    for stats in ranked:
        latency_color = "green" if stats.avg_latency < 2000 else "yellow" if stats.avg_latency < 5000 else "red"
        acc_color = "green" if stats.avg_accuracy > 0.8 else "yellow" if stats.avg_accuracy > 0.5 else "red"
        error_color = "red" if stats.error_rate > 0 else "green"

        table.add_row(
            stats.model,
            f"[{latency_color}]{stats.avg_latency:.0f}ms[/{latency_color}]",
            f"[{acc_color}]{stats.avg_accuracy:.0%}[/{acc_color}]",
            f"{stats.avg_response_length:.0f}",
            f"{stats.total_tokens:,}",
            f"[{error_color}]{stats.error_rate:.0%}[/{error_color}]"
        )

    console.print(table)

    # Category breakdown
    console.print("\n[bold]📋 Category Breakdown[/bold]")
    categories = list(set(r.category for stats in model_stats.values() for r in stats.results))

    cat_table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    cat_table.add_column("Category", style="cyan")
    for model in MODELS:
        cat_table.add_column(model.split("-")[0], justify="center")

    for cat in sorted(categories):
        row = [cat]
        for model in MODELS:
            cat_results = [r for r in model_stats[model].results if r.category == cat and not r.error]
            if cat_results:
                avg_lat = statistics.mean(r.latency_ms for r in cat_results)
                row.append(f"{avg_lat:.0f}ms")
            else:
                row.append("[red]ERR[/red]")
        cat_table.add_row(*row)

    console.print(cat_table)

    # Winner announcement
    fastest = min(model_stats.values(), key=lambda x: x.avg_latency)
    most_accurate = max(model_stats.values(), key=lambda x: x.avg_accuracy)

    console.print(Panel(
        f"⚡ [bold green]Fastest:[/bold green] {fastest.model} ({fastest.avg_latency:.0f}ms avg)\n"
        f"🎯 [bold blue]Most Accurate:[/bold blue] {most_accurate.model} ({most_accurate.avg_accuracy:.0%})",
        title="[bold]🏆 Winners[/bold]",
        border_style="yellow"
    ))


def generate_html_report(model_stats: dict[str, ModelStats], tasks: list[dict]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = ""
    for model, stats in model_stats.items():
        short_name = model.split("-")[0]
        for result in stats.results:
            status = "❌" if result.error else "✅"
            response_preview = (result.response[:100] + "...") if len(result.response) > 100 else result.response
            rows += f"""
            <tr>
                <td>{short_name}</td>
                <td>{result.task_id}</td>
                <td><span class="badge {result.category}">{result.category}</span></td>
                <td>{result.latency_ms:.0f}ms</td>
                <td>{result.accuracy_score:.0%}</td>
                <td>{result.total_tokens}</td>
                <td>{status}</td>
                <td class="response">{response_preview}</td>
            </tr>"""

    summary_cards = ""
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1].avg_latency):
        summary_cards += f"""
        <div class="card">
            <h3>{model}</h3>
            <div class="metric"><span>Avg Latency</span><strong>{stats.avg_latency:.0f}ms</strong></div>
            <div class="metric"><span>Accuracy</span><strong>{stats.avg_accuracy:.0%}</strong></div>
            <div class="metric"><span>Total Tokens</span><strong>{stats.total_tokens:,}</strong></div>
            <div class="metric"><span>Errors</span><strong>{stats.error_rate:.0%}</strong></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalderR Model Benchmark Report</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; text-align: center; }}
        h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .meta {{ text-align: center; color: #8b949e; margin-bottom: 30px; }}
        .cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 30px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; flex: 1; min-width: 200px; }}
        .card h3 {{ color: #f0f6fc; margin: 0 0 12px; font-size: 14px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
        th {{ background: #21262d; padding: 10px 12px; text-align: left; color: #79c0ff; font-size: 13px; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 12px; }}
        tr:hover {{ background: #1c2128; }}
        .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
        .reasoning {{ background: #1f6feb33; color: #79c0ff; }}
        .coding {{ background: #238636; color: #3fb950; }}
        .creative {{ background: #6e40c933; color: #d2a8ff; }}
        .factual {{ background: #b08800; color: #f0c000; }}
        .summarization {{ background: #cf222e33; color: #f85149; }}
        .response {{ max-width: 300px; font-size: 11px; color: #8b949e; }}
    </style>
</head>
<body>
    <h1>🤖 CalderR Model Benchmark Report</h1>
    <p class="meta">Generated: {timestamp} | Models: {len(MODELS)} | Tasks: {len(tasks)}</p>

    <h2>📊 Model Summary</h2>
    <div class="cards">{summary_cards}</div>

    <h2>📋 Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Task</th>
                <th>Category</th>
                <th>Latency</th>
                <th>Accuracy</th>
                <th>Tokens</th>
                <th>Status</th>
                <th>Response Preview</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
</body>
</html>"""

    report_path = "week1/projects/model_benchmark/benchmark_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


async def main():
    console.print(Panel.fit(
        "[bold cyan]🏎️ CalderR Multi-Model Comparison Engine[/bold cyan]\n"
        "[dim]Benchmarking Groq models: latency · accuracy · efficiency[/dim]",
        border_style="cyan"
    ))

    # Load tasks
    yaml_path = "week1/projects/model_benchmark/tasks.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    console.print(f"[green]✅ Loaded {len(tasks)} tasks from {config['benchmark_name']}[/green]")

    # Run benchmark
    model_stats = await run_benchmark(tasks)

    # Display results
    display_results(model_stats)

    # Save JSON results
    json_results = {}
    for model, stats in model_stats.items():
        json_results[model] = {
            "summary": {
                "avg_latency_ms": round(stats.avg_latency, 2),
                "avg_accuracy": round(stats.avg_accuracy, 4),
                "total_tokens": stats.total_tokens,
                "error_rate": round(stats.error_rate, 4)
            },
            "tasks": [asdict(r) for r in stats.results]
        }

    with open("week1/projects/model_benchmark/results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Generate HTML report
    report_path = generate_html_report(model_stats, tasks)

    console.print(f"\n[green]✅ JSON results saved: week1/projects/model_benchmark/results.json[/green]")
    console.print(f"[green]✅ HTML report saved: {report_path}[/green]")
    console.print("[bold yellow]\n📌 Open benchmark_report.html in your browser![/bold yellow]")


if __name__ == "__main__":
    asyncio.run(main())