"""
Day 4: Advanced Retrieval
- BM25 keyword search
- Hybrid search (BM25 + semantic)
- Cross-encoder re-ranking
- Multi-query retrieval
"""

import os
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()
console = Console()

# ── SAMPLE CORPUS ─────────────────────────────────────────────────────────────

CORPUS_TEXTS = [
    "Python is the most popular programming language for machine learning and data science.",
    "LangChain is an open-source framework for building LLM-powered applications.",
    "ChromaDB is a vector database specifically designed for AI and machine learning.",
    "FastAPI is a modern, fast web framework for building APIs with Python.",
    "Docker containers package applications with all dependencies for consistent deployment.",
    "Kubernetes is an open-source system for automating deployment of containerized apps.",
    "FAISS is Facebook AI Similarity Search library for efficient vector similarity.",
    "RAG combines information retrieval with language model generation.",
    "Transformers use self-attention mechanisms to process sequential data in parallel.",
    "Pydantic is a data validation library using Python type hints.",
    "GPT-4 is a large language model developed by OpenAI with multimodal capabilities.",
    "Sentence transformers produce embeddings optimized for semantic similarity tasks.",
    "Vector databases enable efficient similarity search over millions of embeddings.",
    "Cosine similarity measures the angle between two vectors in high-dimensional space.",
    "Cross-encoders re-rank retrieved documents by jointly encoding query and document.",
    "BM25 is a classic keyword-based ranking algorithm used in information retrieval.",
    "Hybrid search combines BM25 keyword matching with semantic vector similarity.",
    "RAGAS evaluates RAG systems using faithfulness, relevancy, and precision metrics.",
    "LangGraph provides a graph-based framework for building complex agent workflows.",
    "Groq offers ultra-fast LLM inference using custom LPU hardware architecture.",
]

QUERIES = [
    "Which databases are used for storing vectors?",
    "How does BM25 work for information retrieval?",
    "What is the difference between semantic and keyword search?",
    "How to deploy Python applications?",
]


# ── DEMO 1: BM25 Keyword Search ───────────────────────────────────────────────

def demo_bm25():
    console.print("\n[bold cyan]━━ Demo 1: BM25 Keyword Search ━━[/bold cyan]")

    docs = [Document(page_content=t) for t in CORPUS_TEXTS]
    bm25_retriever = BM25Retriever.from_documents(docs, k=3)

    query = "vector database similarity search"
    results = bm25_retriever.invoke(query)

    console.print(f"Query: '[yellow]{query}[/yellow]'")
    console.print("[dim]BM25 matches on keyword overlap:[/dim]")
    for i, doc in enumerate(results):
        console.print(f"  {i+1}. {doc.page_content[:70]}...")

    console.print("\n[dim]Key: BM25 ranks by keyword frequency. 'vector' + 'similarity' + 'search' = high score[/dim]")


# ── DEMO 2: Semantic Search ───────────────────────────────────────────────────

def demo_semantic():
    console.print("\n[bold cyan]━━ Demo 2: Semantic Search ━━[/bold cyan]")

    docs = [Document(page_content=t) for t in CORPUS_TEXTS]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="day4_semantic")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = "fast database for finding similar items"
    results = retriever.invoke(query)

    console.print(f"Query: '[yellow]{query}[/yellow]'")
    console.print("[dim]Semantic search matches on MEANING (no keyword overlap needed):[/dim]")
    for i, doc in enumerate(results):
        console.print(f"  {i+1}. {doc.page_content[:70]}...")

    return vectorstore


# ── DEMO 3: Hybrid Search ─────────────────────────────────────────────────────

def demo_hybrid(vectorstore):
    console.print("\n[bold cyan]━━ Demo 3: Hybrid Search (BM25 + Semantic) ━━[/bold cyan]")

    docs = [Document(page_content=t) for t in CORPUS_TEXTS]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # BM25 retriever
    bm25 = BM25Retriever.from_documents(docs, k=5)
    bm25.k = 5

    # Semantic retriever
    semantic = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Ensemble: 40% BM25 + 60% semantic
    ensemble = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.4, 0.6]  # weight toward semantic
    )

    query = "fast database for finding similar items"
    results = ensemble.invoke(query)

    console.print(f"Query: '[yellow]{query}[/yellow]'")
    console.print("[dim]Hybrid combines keyword + meaning matching:[/dim]")
    for i, doc in enumerate(results[:4]):
        console.print(f"  {i+1}. {doc.page_content[:70]}...")

    return ensemble


# ── DEMO 4: Cross-Encoder Re-ranking ─────────────────────────────────────────

def demo_reranking(ensemble):
    console.print("\n[bold cyan]━━ Demo 4: Cross-Encoder Re-ranking ━━[/bold cyan]")

    console.print("[dim]Loading cross-encoder model...[/dim]")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "vector similarity search database"
    initial_results = ensemble.invoke(query)

    console.print(f"\nQuery: '[yellow]{query}[/yellow]'")
    console.print("\n[yellow]Before re-ranking:[/yellow]")
    for i, doc in enumerate(initial_results[:5]):
        console.print(f"  {i+1}. {doc.page_content[:65]}...")

    # Cross-encoder scores each (query, doc) pair jointly
    pairs = [(query, doc.page_content) for doc in initial_results]
    scores = reranker.predict(pairs)

    # Re-sort by cross-encoder score
    reranked = sorted(
        zip(scores, initial_results),
        key=lambda x: x[0],
        reverse=True
    )

    console.print("\n[green]After re-ranking:[/green]")
    for i, (score, doc) in enumerate(reranked[:5]):
        console.print(f"  {i+1}. [score={score:.3f}] {doc.page_content[:60]}...")

    console.print("\n[dim]Cross-encoder reads (query + doc) together → more accurate relevance[/dim]")


# ── DEMO 5: Multi-Query Retrieval ─────────────────────────────────────────────

def demo_multi_query():
    console.print("\n[bold cyan]━━ Demo 5: Multi-Query Retrieval ━━[/bold cyan]")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )

    original_query = "how to store and search vectors efficiently"

    # Generate query variations
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 3 different search queries that capture different aspects of the original question. Return only the queries, one per line."),
        ("user", f"Original: {original_query}")
    ])
    chain = prompt | llm | StrOutputParser()
    variations_text = chain.invoke({})
    variations = [q.strip() for q in variations_text.strip().split("\n") if q.strip()][:3]

    console.print(f"Original: '[yellow]{original_query}[/yellow]'")
    console.print("\nGenerated variations:")
    for i, v in enumerate(variations):
        console.print(f"  {i+1}. {v}")

    console.print("\n[dim]Each variation captures a different angle → more comprehensive retrieval[/dim]")


if __name__ == "__main__":
    console.print("[bold cyan]🔍 Day 4: Advanced Retrieval[/bold cyan]\n")

    demo_bm25()
    vectorstore = demo_semantic()
    ensemble = demo_hybrid(vectorstore)
    demo_reranking(ensemble)
    demo_multi_query()

    console.print("\n[green]✅ Day 4 Complete! Advanced retrieval mastered.[/green]")