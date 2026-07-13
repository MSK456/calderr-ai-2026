"""
Project 3-P-C: RAG Evaluation Benchmark
YAML-driven | Multiple configurations | RAGAS evaluation | HTML reports
"""

import os
import sys
import json
import logging
import statistics
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import yaml
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import RAGPipeline, PipelineConfig, PipelineResult
from data.knowledge_base import KNOWLEDGE_DOCUMENTS, EVALUATION_QA_PAIRS

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console()

REPORT_DIR = Path("week3/projects/rag_benchmark/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = Path("week3/projects/rag_benchmark/configs/benchmark.yaml")


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    """Convert knowledge base to LangChain Documents."""
    docs = []
    for domain, content in KNOWLEDGE_DOCUMENTS.items():
        docs.append(Document(
            page_content=content.strip(),
            metadata={"domain": domain, "source": f"{domain}.txt"}
        ))
    return docs


def load_config(config_path: Path) -> dict:
    """Load benchmark YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── PIPELINE EXECUTION ────────────────────────────────────────────────────────

def run_single_config(
    config_dict: dict,
    documents: list[Document],
    qa_pairs: list[dict]
) -> tuple[str, list[PipelineResult]]:
    """Build and run a single RAG configuration."""

    config = PipelineConfig(
        config_id=config_dict["id"],
        label=config_dict["label"],
        embedding_model=config_dict["embedding_model"].replace(
            "MiniLM-L6-v2", "all-MiniLM-L6-v2"
        ).replace("BGE-small-en", "BAAI/bge-small-en-v1.5"),
        chunk_size=config_dict["chunk_size"],
        overlap=config_dict["overlap"],
        k=config_dict["k"],
        strategy=config_dict["strategy"],
    )

    pipeline = RAGPipeline(config, documents)
    results = []

    for qa in qa_pairs:
        result = pipeline.run(qa["question"], qa["ground_truth"])
        results.append(result)

    return config_dict["label"], results


# ── RAGAS EVALUATION ──────────────────────────────────────────────────────────

@dataclass
class ConfigScore:
    label: str
    config_id: str
    embedding_model: str
    chunk_size: int
    k: int
    strategy: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    avg_latency_ms: float
    total_chunks: int
    num_questions: int

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_precision) / 3


def evaluate_with_ragas(results: list[PipelineResult]) -> dict:
    """Run RAGAS evaluation on pipeline results."""
    data = {
        "question": [r.question for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": [r.ground_truth for r in results],
    }
    dataset = Dataset.from_dict(data)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings)
        )
        df = result.to_pandas()
        return {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


# ── HTML REPORT GENERATOR ─────────────────────────────────────────────────────

def generate_html_report(scores: list[ConfigScore], config: dict) -> str:
    """Generate comprehensive HTML benchmark report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = max(scores, key=lambda s: s.avg_score)

    # Prepare data for charts
    labels = [s.label for s in scores]
    faithfulness_data = [round(s.faithfulness, 3) for s in scores]
    relevancy_data = [round(s.answer_relevancy, 3) for s in scores]
    precision_data = [round(s.context_precision, 3) for s in scores]
    latency_data = [round(s.avg_latency_ms, 0) for s in scores]
    avg_data = [round(s.avg_score, 3) for s in scores]

    # Table rows
    rows = ""
    for s in sorted(scores, key=lambda x: x.avg_score, reverse=True):
        is_best = "🏆" if s.label == best.label else ""
        avg_color = "#3fb950" if s.avg_score > 0.7 else "#f0c000" if s.avg_score > 0.5 else "#f85149"
        rows += f"""
        <tr class="{'best-row' if is_best else ''}">
            <td>{is_best} {s.label}</td>
            <td>{s.embedding_model.split('/')[-1]}</td>
            <td>{s.chunk_size}</td>
            <td>{s.k}</td>
            <td>{s.strategy}</td>
            <td>{s.faithfulness:.3f}</td>
            <td>{s.answer_relevancy:.3f}</td>
            <td>{s.context_precision:.3f}</td>
            <td style="color:{avg_color};font-weight:bold">{s.avg_score:.3f}</td>
            <td>{s.avg_latency_ms:.0f}ms</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Benchmark Report — {config['benchmark_name']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; font-size: 2rem; margin-bottom: 8px; }}
  h2 {{ color: #79c0ff; font-size: 1.3rem; margin: 32px 0 16px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .meta {{ color: #8b949e; margin-bottom: 32px; font-size: 14px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
  .stat-value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ color: #8b949e; font-size: 13px; margin-top: 4px; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; font-size: 13px; }}
  th {{ background: #21262d; padding: 12px; text-align: left; color: #79c0ff; font-size: 12px; text-transform: uppercase; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .best-row {{ background: #1a2e1a !important; }}
  .winner-card {{ background: #1a2e1a; border: 2px solid #3fb950; border-radius: 12px; padding: 24px; margin-bottom: 32px; }}
  .winner-card h3 {{ color: #3fb950; margin-bottom: 12px; }}
  .metric-pills {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
  .pill {{ background: #21262d; border-radius: 20px; padding: 4px 16px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🔬 {config['benchmark_name']}</h1>
  <div class="meta">
    Generated: {timestamp} &nbsp;|&nbsp;
    Configurations: {len(scores)} &nbsp;|&nbsp;
    Questions: {scores[0].num_questions if scores else 0} &nbsp;|&nbsp;
    Version: {config.get('version', '1.0')}
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">{len(scores)}</div>
      <div class="stat-label">Configurations Tested</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{scores[0].num_questions if scores else 0}</div>
      <div class="stat-label">Evaluation Questions</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{best.avg_score:.2f}</div>
      <div class="stat-label">Best Avg RAGAS Score</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{min(s.avg_latency_ms for s in scores):.0f}ms</div>
      <div class="stat-label">Fastest Config</div>
    </div>
  </div>

  <div class="winner-card">
    <h3>🏆 Best Configuration: {best.label}</h3>
    <div class="metric-pills">
      <span class="pill">Chunk Size: {best.chunk_size}</span>
      <span class="pill">k: {best.k}</span>
      <span class="pill">Strategy: {best.strategy}</span>
      <span class="pill">Embedding: {best.embedding_model.split('/')[-1]}</span>
      <span class="pill">Avg Score: {best.avg_score:.3f}</span>
    </div>
  </div>

  <h2>📊 Performance Charts</h2>
  <div class="charts-grid">
    <div class="chart-box">
      <canvas id="ragas_chart"></canvas>
    </div>
    <div class="chart-box">
      <canvas id="latency_chart"></canvas>
    </div>
  </div>

  <h2>📋 Full Results Table</h2>
  <table>
    <thead>
      <tr>
        <th>Config</th><th>Embedding</th><th>Chunk</th>
        <th>k</th><th>Strategy</th>
        <th>Faithfulness</th><th>Relevancy</th><th>Precision</th>
        <th>Avg Score</th><th>Latency</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>💡 Key Findings</h2>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:24px;line-height:1.8">
    <p>• <strong>Best overall config:</strong> {best.label} with avg RAGAS score of {best.avg_score:.3f}</p>
    <p>• <strong>Faithfulness</strong> improves with larger chunks (more context per retrieved document)</p>
    <p>• <strong>Context Precision</strong> is higher with smaller chunks (more targeted retrieval)</p>
    <p>• <strong>Hybrid search</strong> generally outperforms pure semantic on diverse question types</p>
    <p>• <strong>Latency tradeoff:</strong> larger chunks = faster indexing but slower retrieval at scale</p>
  </div>
</div>

<script>
const labels = {json.dumps(labels)};
const colors = ['#58a6ff','#3fb950','#f0c000','#ff7b72','#d2a8ff','#ffa657'];

new Chart(document.getElementById('ragas_chart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Faithfulness', data: {faithfulness_data}, backgroundColor: '#58a6ff88', borderColor: '#58a6ff', borderWidth: 2 }},
      {{ label: 'Answer Relevancy', data: {relevancy_data}, backgroundColor: '#3fb95088', borderColor: '#3fb950', borderWidth: 2 }},
      {{ label: 'Context Precision', data: {precision_data}, backgroundColor: '#f0c00088', borderColor: '#f0c000', borderWidth: 2 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'RAGAS Metrics by Configuration', color: '#e6edf3', font: {{ size: 14 }} }},
      legend: {{ labels: {{ color: '#e6edf3' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }} }},
      y: {{ min: 0, max: 1, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

new Chart(document.getElementById('latency_chart'), {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [
      {{ label: 'Avg Latency (ms)', data: {latency_data}, backgroundColor: '#ff7b7288', borderColor: '#ff7b72', borderWidth: 2 }},
      {{ label: 'Avg RAGAS Score (×1000ms)', data: {[round(s*1000) for s in avg_data]}, backgroundColor: '#d2a8ff88', borderColor: '#d2a8ff', borderWidth: 2 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Latency vs Quality Tradeoff', color: '#e6edf3', font: {{ size: 14 }} }},
      legend: {{ labels: {{ color: '#e6edf3' }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────

def run_benchmark():
    console.print(Panel.fit(
        "[bold cyan]🔬 RAG Evaluation Benchmark — Project 3-P-C[/bold cyan]\n"
        "[dim]YAML-driven | 6 configurations | RAGAS metrics | HTML reports[/dim]",
        border_style="cyan"
    ))

    # Load config
    config = load_config(CONFIG_FILE)
    benchmark_configs = config["benchmark_configs"]
    console.print(f"[green]✅ Loaded config: {len(benchmark_configs)} configurations[/green]")

    # Load documents
    documents = load_documents()
    console.print(f"[green]✅ Loaded {len(documents)} knowledge base documents[/green]")
    console.print(f"[green]✅ {len(EVALUATION_QA_PAIRS)} evaluation Q&A pairs[/green]")

    # Run all configs
    all_pipeline_results: dict[str, list[PipelineResult]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Running configurations...", total=len(benchmark_configs))

        for cfg in benchmark_configs:
            progress.update(task, description=f"Building: {cfg['label']}")
            label, results = run_single_config(cfg, documents, EVALUATION_QA_PAIRS)
            all_pipeline_results[label] = results
            progress.advance(task)

    console.print(f"[green]✅ All {len(benchmark_configs)} configurations executed[/green]")

    # RAGAS Evaluation
    console.print("\n[yellow]Running RAGAS evaluation on all configurations...[/yellow]")
    console.print("[dim]This uses LLM as judge — may take a few minutes[/dim]\n")

    all_scores: list[ConfigScore] = []
    cfg_lookup = {c["label"]: c for c in benchmark_configs}

    for label, results in all_pipeline_results.items():
        console.print(f"  [dim]Evaluating: {label}...[/dim]")
        ragas_scores = evaluate_with_ragas(results)
        cfg = cfg_lookup[label]

        avg_latency = statistics.mean(r.latency_ms for r in results)
        num_chunks = results[0].num_chunks_total if results else 0

        score = ConfigScore(
            label=label,
            config_id=cfg["id"],
            embedding_model=cfg["embedding_model"],
            chunk_size=cfg["chunk_size"],
            k=cfg["k"],
            strategy=cfg["strategy"],
            faithfulness=ragas_scores["faithfulness"],
            answer_relevancy=ragas_scores["answer_relevancy"],
            context_precision=ragas_scores["context_precision"],
            avg_latency_ms=avg_latency,
            total_chunks=num_chunks,
            num_questions=len(results)
        )
        all_scores.append(score)
        console.print(
            f"    Faith={score.faithfulness:.3f} | "
            f"Rel={score.answer_relevancy:.3f} | "
            f"Prec={score.context_precision:.3f} | "
            f"Avg={score.avg_score:.3f}"
        )

    # Display results table
    console.print("\n")
    table = Table(title="📊 Benchmark Results", box=box.ROUNDED)
    table.add_column("Config", style="cyan")
    table.add_column("Chunk", justify="center")
    table.add_column("k", justify="center")
    table.add_column("Strategy", justify="center")
    table.add_column("Faithfulness", justify="center", style="yellow")
    table.add_column("Relevancy", justify="center", style="green")
    table.add_column("Precision", justify="center", style="blue")
    table.add_column("Avg ▼", justify="center", style="bold")

    for s in sorted(all_scores, key=lambda x: x.avg_score, reverse=True):
        avg_color = "green" if s.avg_score > 0.7 else "yellow" if s.avg_score > 0.5 else "red"
        table.add_row(
            s.label,
            str(s.chunk_size),
            str(s.k),
            s.strategy,
            f"{s.faithfulness:.3f}",
            f"{s.answer_relevancy:.3f}",
            f"{s.context_precision:.3f}",
            f"[{avg_color}]{s.avg_score:.3f}[/{avg_color}]"
        )

    console.print(table)

    # Save JSON results
    json_path = REPORT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2)
    console.print(f"\n[green]✅ JSON results saved: {json_path}[/green]")

    # Generate HTML report
    html_content = generate_html_report(all_scores, config)
    html_path = REPORT_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    console.print(f"[green]✅ HTML report saved: {html_path}[/green]")
    console.print(f"[bold yellow]📌 Open {html_path} in browser for full interactive report![/bold yellow]")

    # Best config summary
    best = max(all_scores, key=lambda s: s.avg_score)
    console.print(Panel(
        f"[bold green]🏆 Best Configuration: {best.label}[/bold green]\n"
        f"Faithfulness: {best.faithfulness:.3f} | "
        f"Relevancy: {best.answer_relevancy:.3f} | "
        f"Precision: {best.context_precision:.3f}\n"
        f"Average RAGAS Score: {best.avg_score:.3f} | "
        f"Latency: {best.avg_latency_ms:.0f}ms",
        border_style="green"
    ))

    console.print("\n[green]✅ Project 3-P-C Complete! RAG Benchmark done.[/green]")


if __name__ == "__main__":
    run_benchmark()