"""
Project 3-I-C: Hybrid Search Engine
Industry-grade search combining BM25 + Semantic + Cross-encoder Re-ranking
"""

import os
import sys
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.columns import Columns
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.corpus import get_all_articles
from indexer import HybridIndexer, INDEX_DIR
from searcher import HybridSearcher, SearchResponse
from reranker import CrossEncoderReranker
from evaluator import run_full_evaluation

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()


def display_results(response: SearchResponse, show_preview: bool = True):
    """Rich display of search results."""
    strategy_color = {
        "bm25": "yellow",
        "semantic": "blue",
        "hybrid": "green",
        "hybrid+rerank": "magenta",
    }.get(response.strategy, "white")

    title = (
        f"[{strategy_color}]{response.strategy.upper()}[/{strategy_color}] "
        f"— '{response.query}' "
        f"[dim]({response.elapsed_ms:.0f}ms, {response.total_found} results)[/dim]"
    )

    table = Table(title=title, box=box.ROUNDED, show_lines=show_preview)
    table.add_column("#", width=3, justify="center", style="dim")
    table.add_column("Title", style="cyan", width=45)
    table.add_column("Category", width=10, justify="center")
    table.add_column("Date", width=10, justify="center")
    table.add_column("Score", width=7, justify="center")

    if show_preview:
        table.add_column("Preview", style="dim", width=40)

    for r in response.results:
        cat_colors = {
            "ai": "blue", "tech": "green", "science": "yellow",
            "education": "cyan", "health": "red"
        }
        cat_color = cat_colors.get(r.category, "white")
        score_color = "green" if r.score > 0.5 else "yellow" if r.score > 0.2 else "dim"

        row = [
            str(r.rank),
            r.title[:43] + ".." if len(r.title) > 43 else r.title,
            f"[{cat_color}]{r.category}[/{cat_color}]",
            r.date,
            f"[{score_color}]{r.score:.3f}[/{score_color}]",
        ]
        if show_preview:
            row.append(r.content_preview[:38] + "...")

        table.add_row(*row)

    console.print(table)


def display_comparison(
    query: str,
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    top_k: int = 5
):
    """Show side-by-side comparison of all search strategies."""
    console.print(f"\n[bold]Comparing strategies for: '[yellow]{query}[/yellow]'[/bold]")

    for strategy in ["bm25", "semantic", "hybrid"]:
        response = searcher.search(query, strategy=strategy, top_k=top_k)
        display_results(response, show_preview=False)

    # With reranking
    hybrid_response = searcher.search(query, strategy="hybrid", top_k=top_k * 2)
    reranked_response = reranker.rerank(hybrid_response, top_k=top_k)
    display_results(reranked_response, show_preview=False)


def display_evaluation_results(eval_results):
    """Display evaluation comparison table."""
    table = Table(
        title="📊 Search Strategy Evaluation (30 Benchmark Queries)",
        box=box.ROUNDED
    )
    table.add_column("Strategy", style="cyan")
    table.add_column("P@1", justify="center", style="yellow")
    table.add_column("P@3", justify="center", style="green")
    table.add_column("R@5", justify="center", style="blue")
    table.add_column("Avg Latency", justify="center", style="dim")
    table.add_column("Winner?", justify="center")

    best_p3 = max(r.precision_at_3 for r in eval_results)

    for r in eval_results:
        is_best = "🏆" if r.precision_at_3 == best_p3 else ""
        p3_color = "green" if r.precision_at_3 == best_p3 else "yellow"

        table.add_row(
            r.strategy,
            f"{r.precision_at_1:.1%}",
            f"[{p3_color}]{r.precision_at_3:.1%}[/{p3_color}]",
            f"{r.recall_at_5:.1%}",
            f"{r.avg_latency_ms:.0f}ms",
            is_best
        )

    console.print(table)


def interactive_search_cli(searcher: HybridSearcher, reranker: CrossEncoderReranker):
    """Interactive search CLI."""
    console.print(Panel.fit(
        "[bold cyan]🔍 Hybrid Search Engine — Interactive Mode[/bold cyan]\n"
        "[dim]Commands:[/dim]\n"
        "  [yellow]/bm25[/yellow]     → BM25 keyword search\n"
        "  [yellow]/semantic[/yellow] → Semantic vector search\n"
        "  [yellow]/hybrid[/yellow]   → Hybrid search (default)\n"
        "  [yellow]/rerank[/yellow]   → Hybrid + cross-encoder re-ranking\n"
        "  [yellow]/compare[/yellow]  → Compare all strategies\n"
        "  [yellow]/filter[/yellow] <cat> → Filter by category (ai/tech/science)\n"
        "  [yellow]/exit[/yellow]     → Quit",
        border_style="cyan"
    ))

    strategy = "hybrid"
    category_filter = None

    while True:
        prompt_text = f"[bold blue][{strategy}][/bold blue] Search"
        query = Prompt.ask(f"\n{prompt_text}").strip()

        if not query:
            continue

        if query.lower() == "/exit":
            break
        elif query.lower() == "/bm25":
            strategy = "bm25"
            console.print("[yellow]Switched to BM25[/yellow]")
            continue
        elif query.lower() == "/semantic":
            strategy = "semantic"
            console.print("[blue]Switched to Semantic[/blue]")
            continue
        elif query.lower() == "/hybrid":
            strategy = "hybrid"
            category_filter = None
            console.print("[green]Switched to Hybrid[/green]")
            continue
        elif query.lower() == "/rerank":
            strategy = "rerank"
            console.print("[magenta]Switched to Hybrid + Re-ranking[/magenta]")
            continue
        elif query.lower().startswith("/filter"):
            parts = query.split()
            category_filter = parts[1] if len(parts) > 1 else None
            console.print(f"[cyan]Category filter: {category_filter or 'none'}[/cyan]")
            continue
        elif query.lower() == "/compare":
            compare_q = Prompt.ask("Enter query to compare").strip()
            if compare_q:
                display_comparison(compare_q, searcher, reranker)
            continue

        # Execute search
        if strategy == "rerank":
            response = searcher.search(query, strategy="hybrid", top_k=10, category_filter=category_filter)
            response = reranker.rerank(response, top_k=5)
        else:
            response = searcher.search(query, strategy=strategy, top_k=5, category_filter=category_filter)

        display_results(response)


def main():
    console.print(Panel.fit(
        "[bold cyan]🔍 Hybrid Search Engine — Project 3-I-C[/bold cyan]\n"
        "[dim]BM25 + Semantic + Cross-Encoder Re-ranking[/dim]\n"
        "[dim]30 News Articles | 15 Benchmark Queries[/dim]",
        border_style="cyan"
    ))

    # 1. Load corpus
    articles = get_all_articles()
    console.print(f"[green]✅ Loaded {len(articles)} articles[/green]")

    # 2. Build indexes
    console.print("[yellow]Building indexes...[/yellow]")
    indexer = HybridIndexer()
    indexer.build_index(articles)
    console.print(f"[green]✅ Index built: {indexer.get_stats()}[/green]")

    # 3. Initialize search components
    searcher = HybridSearcher(
        bm25_retriever=indexer.bm25_retriever,
        vectorstore=indexer.vectorstore,
        bm25_weight=0.4,
        semantic_weight=0.6,
    )

    console.print("[yellow]Loading cross-encoder re-ranker...[/yellow]")
    reranker = CrossEncoderReranker()
    console.print("[green]✅ Re-ranker loaded[/green]")

    # 4. Demo searches
    console.print("\n[bold cyan]━━ Demo Searches ━━[/bold cyan]")

    demo_queries = [
        "OpenAI language model release",
        "Pakistan AI technology innovation",
        "vector database search performance",
    ]

    for query in demo_queries:
        response = searcher.search(query, strategy="hybrid", top_k=3)
        reranked = reranker.rerank(response, top_k=3)
        display_results(reranked, show_preview=True)

    # 5. Strategy comparison
    console.print("\n[bold cyan]━━ Strategy Comparison ━━[/bold cyan]")
    display_comparison("RAG retrieval evaluation framework", searcher, reranker)

    # 6. Evaluation
    console.print("\n[bold cyan]━━ Running Benchmark Evaluation ━━[/bold cyan]")
    console.print("[dim]Evaluating all strategies on 15 queries...[/dim]")
    eval_results = run_full_evaluation(searcher, reranker)
    display_evaluation_results(eval_results)

    # 7. Interactive mode
    console.print("\n[bold cyan]━━ Interactive Search Mode ━━[/bold cyan]")
    interactive_search_cli(searcher, reranker)

    console.print("\n[green]✅ Project 3-I-C Complete![/green]")


if __name__ == "__main__":
    main()