"""
Searcher: Implements BM25, Semantic, and Hybrid search strategies
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SearchStrategy = Literal["bm25", "semantic", "hybrid"]


@dataclass
class SearchResult:
    rank: int
    document_id: str
    title: str
    category: str
    date: str
    content_preview: str
    score: float
    strategy: str
    reranked: bool = False


@dataclass
class SearchResponse:
    query: str
    strategy: str
    results: list[SearchResult]
    total_found: int
    elapsed_ms: float
    reranked: bool = False


class HybridSearcher:
    """Multi-strategy search engine with BM25, semantic, and hybrid modes."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vectorstore: Chroma,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        self.bm25_retriever = bm25_retriever
        self.vectorstore = vectorstore
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self._build_retrievers()

    def _build_retrievers(self, k: int = 10):
        self.bm25_retriever.k = k
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.semantic_retriever],
            weights=[self.bm25_weight, self.semantic_weight]
        )

    def _docs_to_results(
        self,
        docs: list[Document],
        strategy: str,
        reranked: bool = False
    ) -> list[SearchResult]:
        results = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            content = doc.page_content
            # Extract content preview (skip title line)
            lines = content.split("\n\n")
            preview = lines[1][:200] + "..." if len(lines) > 1 else content[:200]

            results.append(SearchResult(
                rank=i + 1,
                document_id=meta.get("id", f"doc_{i}"),
                title=meta.get("title", "Unknown"),
                category=meta.get("category", "unknown"),
                date=meta.get("date", "unknown"),
                content_preview=preview,
                score=1.0 / (i + 1),  # reciprocal rank score
                strategy=strategy,
                reranked=reranked
            ))
        return results

    def search(
        self,
        query: str,
        strategy: SearchStrategy = "hybrid",
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> SearchResponse:
        """Execute search with specified strategy."""
        start = time.time()

        if strategy == "bm25":
            self.bm25_retriever.k = top_k * 2
            docs = self.bm25_retriever.invoke(query)
        elif strategy == "semantic":
            docs = self.vectorstore.similarity_search(query, k=top_k * 2)
        elif strategy == "hybrid":
            self._build_retrievers(k=top_k * 2)
            docs = self.ensemble_retriever.invoke(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Category filter
        if category_filter:
            docs = [d for d in docs if d.metadata.get("category") == category_filter]

        # Limit to top_k
        docs = docs[:top_k]

        elapsed = (time.time() - start) * 1000
        results = self._docs_to_results(docs, strategy)

        return SearchResponse(
            query=query,
            strategy=strategy,
            results=results,
            total_found=len(results),
            elapsed_ms=round(elapsed, 2)
        )