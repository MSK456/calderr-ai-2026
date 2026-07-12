"""
Reranker: Cross-encoder based re-ranking of search results
"""

import logging
import time
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from searcher import SearchResponse, SearchResult

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-ranks search results using a cross-encoder model."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        response: SearchResponse,
        top_k: int | None = None,
    ) -> SearchResponse:
        """Re-rank a SearchResponse using cross-encoder scores."""
        if not response.results:
            return response

        start = time.time()
        query = response.query

        # Score each (query, content_preview) pair
        pairs = [(query, r.content_preview) for r in response.results]
        scores = self.model.predict(pairs)

        # Sort by cross-encoder score
        scored_results = sorted(
            zip(scores, response.results),
            key=lambda x: x[0],
            reverse=True
        )

        # Rebuild results with new ranks and scores
        reranked = []
        limit = top_k or len(scored_results)
        for new_rank, (score, result) in enumerate(scored_results[:limit]):
            reranked.append(SearchResult(
                rank=new_rank + 1,
                document_id=result.document_id,
                title=result.title,
                category=result.category,
                date=result.date,
                content_preview=result.content_preview,
                score=float(score),
                strategy=result.strategy,
                reranked=True
            ))

        elapsed = (time.time() - start) * 1000
        logger.debug(f"Re-ranking took {elapsed:.1f}ms for {len(pairs)} results")

        return SearchResponse(
            query=response.query,
            strategy=f"{response.strategy}+rerank",
            results=reranked,
            total_found=len(reranked),
            elapsed_ms=response.elapsed_ms + elapsed,
            reranked=True
        )