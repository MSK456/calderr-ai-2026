"""
Evaluator: Compares search strategies on benchmark queries
"""

import json
import logging
from dataclasses import dataclass, asdict
from searcher import HybridSearcher, SearchStrategy
from reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
    {"query": "OpenAI language model release", "relevant_ids": ["art_001", "art_012", "art_022"]},
    {"query": "Pakistan technology innovation", "relevant_ids": ["art_004", "art_014", "art_020"]},
    {"query": "vector database similarity search", "relevant_ids": ["art_011", "art_026", "art_016"]},
    {"query": "quantum computing breakthrough", "relevant_ids": ["art_005"]},
    {"query": "machine learning framework Python", "relevant_ids": ["art_006", "art_016"]},
    {"query": "electric vehicles charging infrastructure", "relevant_ids": ["art_024"]},
    {"query": "AI agent automation enterprise", "relevant_ids": ["art_025", "art_008"]},
    {"query": "renewable energy solar wind power", "relevant_ids": ["art_015"]},
    {"query": "cybersecurity phishing AI threats", "relevant_ids": ["art_029"]},
    {"query": "RAG retrieval augmented generation evaluation", "relevant_ids": ["art_018", "art_016"]},
    {"query": "GPU chip AI training hardware", "relevant_ids": ["art_009"]},
    {"query": "university students AI competition award", "relevant_ids": ["art_014", "art_027"]},
    {"query": "open source LLM model weights", "relevant_ids": ["art_003", "art_022", "art_023"]},
    {"query": "space rocket mission launch", "relevant_ids": ["art_019"]},
    {"query": "nuclear energy data center power", "relevant_ids": ["art_030"]},
]


@dataclass
class EvalResult:
    strategy: str
    precision_at_1: float
    precision_at_3: float
    recall_at_5: float
    avg_latency_ms: float
    total_queries: int


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def evaluate_strategy(
    searcher: HybridSearcher,
    strategy: SearchStrategy,
    reranker: CrossEncoderReranker | None = None,
    top_k: int = 5,
) -> EvalResult:
    """Evaluate a search strategy on benchmark queries."""

    p1_scores, p3_scores, r5_scores, latencies = [], [], [], []

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        relevant = item["relevant_ids"]

        response = searcher.search(query, strategy=strategy, top_k=top_k * 2)

        if reranker:
            response = reranker.rerank(response, top_k=top_k)
        else:
            response.results = response.results[:top_k]

        retrieved_ids = [r.document_id for r in response.results]

        p1_scores.append(precision_at_k(retrieved_ids, relevant, 1))
        p3_scores.append(precision_at_k(retrieved_ids, relevant, 3))
        r5_scores.append(recall_at_k(retrieved_ids, relevant, 5))
        latencies.append(response.elapsed_ms)

    label = strategy
    if reranker:
        label = f"{strategy}+rerank"

    return EvalResult(
        strategy=label,
        precision_at_1=sum(p1_scores) / len(p1_scores),
        precision_at_3=sum(p3_scores) / len(p3_scores),
        recall_at_5=sum(r5_scores) / len(r5_scores),
        avg_latency_ms=sum(latencies) / len(latencies),
        total_queries=len(BENCHMARK_QUERIES)
    )


def run_full_evaluation(
    searcher: HybridSearcher,
    reranker: CrossEncoderReranker,
    output_path: str = "week3/projects/hybrid_search/evaluation_report.json"
) -> list[EvalResult]:
    """Run evaluation across all strategies."""

    strategies = [
        ("bm25",    False),
        ("semantic", False),
        ("hybrid",   False),
        ("hybrid",   True),   # hybrid + reranking
    ]

    results = []
    for strategy, use_reranker in strategies:
        logger.info(f"Evaluating: {strategy} (rerank={use_reranker})")
        result = evaluate_strategy(
            searcher, strategy,
            reranker=reranker if use_reranker else None
        )
        results.append(result)

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Evaluation saved: {output_path}")

    return results  