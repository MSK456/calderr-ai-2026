"""
Indexer: Builds BM25 and Vector indexes from article corpus
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

INDEX_DIR = Path("week3/projects/hybrid_search/indexes")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class HybridIndexer:
    """Builds and persists BM25 + Vector indexes for hybrid search."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents: list[Document] = []
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )

    def _articles_to_documents(self, articles: list[dict]) -> list[Document]:
        """Convert raw articles to LangChain Documents."""
        docs = []
        for article in articles:
            doc = Document(
                page_content=f"{article['title']}\n\n{article['content']}",
                metadata={
                    "id": article["id"],
                    "title": article["title"],
                    "category": article["category"],
                    "date": article["date"],
                    "word_count": len(article["content"].split()),
                }
            )
            docs.append(doc)
        return docs

    def build_index(self, articles: list[dict], force_rebuild: bool = False) -> None:
        """Build BM25 and vector indexes from articles."""
        bm25_path = self.index_dir / "bm25_index.pkl"
        chroma_path = self.index_dir / "chroma_db"
        meta_path = self.index_dir / "index_meta.json"

        # Check if already built
        if not force_rebuild and bm25_path.exists() and chroma_path.exists():
            logger.info("Loading existing indexes from disk...")
            self._load_indexes()
            return

        logger.info(f"Building indexes for {len(articles)} articles...")

        self.documents = self._articles_to_documents(articles)

        # Build BM25
        logger.info("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=10
        )
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        logger.info(f"BM25 index saved: {bm25_path}")

        # Build Vector index
        logger.info("Building vector index...")
        self.vectorstore = Chroma.from_documents(
            self.documents,
            self.embeddings,
            collection_name="hybrid_search_articles",
            persist_directory=str(chroma_path)
        )
        logger.info(f"Vector index saved: {chroma_path}")

        # Save metadata
        meta = {
            "num_articles": len(articles),
            "embedding_model": EMBEDDING_MODEL,
            "categories": list(set(a["category"] for a in articles)),
            "date_range": f"{min(a['date'] for a in articles)} to {max(a['date'] for a in articles)}"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Index build complete!")

    def _load_indexes(self) -> None:
        """Load existing indexes from disk."""
        bm25_path = self.index_dir / "bm25_index.pkl"
        chroma_path = self.index_dir / "chroma_db"

        with open(bm25_path, "rb") as f:
            self.bm25_retriever = pickle.load(f)

        self.vectorstore = Chroma(
            collection_name="hybrid_search_articles",
            embedding_function=self.embeddings,
            persist_directory=str(chroma_path)
        )

        # Reconstruct documents list
        all_docs = self.vectorstore.get()
        self.documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ] if all_docs["documents"] else []

        logger.info(f"Loaded {len(self.documents)} documents from existing index")

    def get_stats(self) -> dict:
        """Return index statistics."""
        if self.vectorstore is None:
            return {"status": "not built"}
        count = self.vectorstore._collection.count()
        return {
            "total_documents": count,
            "embedding_model": EMBEDDING_MODEL,
            "index_dir": str(self.index_dir),
        }