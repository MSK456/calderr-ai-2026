"""
Day 2: Vector Databases — ChromaDB + FAISS
- ChromaDB: create, add, query, filter by metadata
- FAISS: IndexFlatL2 vs IndexIVFFlat
- Persistence strategies
- Tradeoffs comparison
"""

import os
import time
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Sample document corpus
DOCUMENTS = [
    {"id": "1",  "text": "Python is a high-level programming language known for simplicity.", "source": "tech", "year": 2024},
    {"id": "2",  "text": "FastAPI is a modern web framework for building APIs with Python.", "source": "tech", "year": 2024},
    {"id": "3",  "text": "LangChain helps developers build LLM-powered applications.", "source": "ai", "year": 2024},
    {"id": "4",  "text": "ChromaDB is an open-source vector database for AI applications.", "source": "ai", "year": 2024},
    {"id": "5",  "text": "FAISS is Facebook's library for efficient similarity search.", "source": "ai", "year": 2023},
    {"id": "6",  "text": "RAG combines retrieval with language model generation.", "source": "ai", "year": 2023},
    {"id": "7",  "text": "Transformers revolutionized natural language processing in 2017.", "source": "ai", "year": 2017},
    {"id": "8",  "text": "Pakistan was founded on August 14, 1947 by Muhammad Ali Jinnah.", "source": "history", "year": 2024},
    {"id": "9",  "text": "Islamabad became the capital of Pakistan in 1966.", "source": "history", "year": 2024},
    {"id": "10", "text": "The Indus Valley Civilization is one of the world's oldest.", "source": "history", "year": 2024},
    {"id": "11", "text": "Docker containers enable consistent software deployment.", "source": "tech", "year": 2024},
    {"id": "12", "text": "Kubernetes orchestrates containerized applications at scale.", "source": "tech", "year": 2024},
    {"id": "13", "text": "Neural networks are inspired by the human brain structure.", "source": "ai", "year": 2024},
    {"id": "14", "text": "GPT-4 is a multimodal language model created by OpenAI.", "source": "ai", "year": 2023},
    {"id": "15", "text": "Pydantic v2 uses Rust for 5-50x faster data validation.", "source": "tech", "year": 2024},
]


# ── CHROMADB DEMO ─────────────────────────────────────────────────────────────

def demo_chromadb():
    console.print("\n[bold cyan]━━ ChromaDB Demo ━━[/bold cyan]")

    # 1. Create client (in-memory)
    client = chromadb.Client()

    # 2. Create collection
    collection = client.create_collection(
        name="week3_docs",
        metadata={"hnsw:space": "cosine"}  # use cosine distance
    )

    # 3. Add documents with metadata
    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    metadatas = [{"source": d["source"], "year": d["year"]} for d in DOCUMENTS]
    embeddings = MODEL.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    console.print(f"[green]✅ Added {len(DOCUMENTS)} documents to ChromaDB[/green]")

    # 4. Basic semantic query
    console.print("\n[yellow]Query 1: Basic semantic search[/yellow]")
    query = "vector database for AI applications"
    query_emb = MODEL.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=3
    )
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        console.print(f"  {i+1}. [{1-dist:.4f}] {doc[:60]}...")

    # 5. Metadata filtering
    console.print("\n[yellow]Query 2: Filter by source='ai' only[/yellow]")
    results_filtered = collection.query(
        query_embeddings=query_emb,
        n_results=3,
        where={"source": "ai"}  # METADATA FILTER
    )
    for i, (doc, meta) in enumerate(zip(results_filtered["documents"][0], results_filtered["metadatas"][0])):
        console.print(f"  {i+1}. [{meta['source']}] {doc[:60]}...")

    # 6. Year filter
    console.print("\n[yellow]Query 3: Filter by year >= 2024[/yellow]")
    results_year = collection.query(
        query_embeddings=query_emb,
        n_results=3,
        where={"year": {"$gte": 2024}}
    )
    for i, doc in enumerate(results_year["documents"][0]):
        console.print(f"  {i+1}. {doc[:60]}...")

    # 7. Persistent ChromaDB
    console.print("\n[yellow]Persistent ChromaDB (saves to disk)[/yellow]")
    persist_client = chromadb.PersistentClient(path="week3/day2_vector_dbs/chroma_db")
    persist_col = persist_client.get_or_create_collection("persistent_docs")
    persist_col.add(documents=texts[:5], embeddings=embeddings[:5], ids=ids[:5])
    console.print(f"[green]✅ Persisted to disk at week3/day2_vector_dbs/chroma_db[/green]")

    return collection


# ── FAISS DEMO ────────────────────────────────────────────────────────────────

def demo_faiss():
    console.print("\n[bold cyan]━━ FAISS Demo ━━[/bold cyan]")

    texts = [d["text"] for d in DOCUMENTS]
    embeddings = MODEL.encode(texts).astype(np.float32)
    dim = embeddings.shape[1]  # 384 for MiniLM

    console.print(f"Embedding dimension: {dim}")
    console.print(f"Number of vectors: {len(embeddings)}")

    # IndexFlatL2 = exact search, brute force
    console.print("\n[yellow]IndexFlatL2 (exact search):[/yellow]")
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(embeddings)
    console.print(f"  Vectors in index: {index_flat.ntotal}")

    query = MODEL.encode(["neural network deep learning"]).astype(np.float32)
    start = time.time()
    distances, indices = index_flat.search(query, k=3)
    elapsed = (time.time() - start) * 1000

    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        console.print(f"  {i+1}. [L2={dist:.4f}] {texts[idx][:55]}...")
    console.print(f"  Search time: {elapsed:.2f}ms")

    # IndexIVFFlat = approximate search, faster for large datasets
    console.print("\n[yellow]IndexIVFFlat (approximate, faster at scale):[/yellow]")
    nlist = 4  # number of clusters (usually sqrt(n))
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index_ivf.train(embeddings)  # must train first
    index_ivf.add(embeddings)
    index_ivf.nprobe = 2  # check 2 nearest clusters

    start = time.time()
    distances_ivf, indices_ivf = index_ivf.search(query, k=3)
    elapsed_ivf = (time.time() - start) * 1000

    for i, (dist, idx) in enumerate(zip(distances_ivf[0], indices_ivf[0])):
        console.print(f"  {i+1}. [L2={dist:.4f}] {texts[idx][:55]}...")
    console.print(f"  Search time: {elapsed_ivf:.2f}ms")

    # Save FAISS index to disk
    faiss.write_index(index_flat, "week3/day2_vector_dbs/faiss_index.bin")
    console.print("\n[green]✅ FAISS index saved to disk[/green]")

    # Load it back
    loaded_index = faiss.read_index("week3/day2_vector_dbs/faiss_index.bin")
    console.print(f"[green]✅ FAISS index loaded: {loaded_index.ntotal} vectors[/green]")


# ── COMPARISON TABLE ──────────────────────────────────────────────────────────

def comparison_table():
    console.print("\n[bold cyan]━━ Vector DB Comparison ━━[/bold cyan]")

    table = Table(box=box.ROUNDED, title="ChromaDB vs FAISS")
    table.add_column("Feature", style="cyan")
    table.add_column("ChromaDB", style="green", justify="center")
    table.add_column("FAISS", style="yellow", justify="center")

    rows = [
        ("Type", "Full vector DB", "Library"),
        ("Persistence", "✅ Built-in", "✅ Manual save/load"),
        ("Metadata filtering", "✅ Yes", "❌ No"),
        ("Search type", "Approximate (HNSW)", "Exact or Approximate"),
        ("Python API", "Simple", "Low-level"),
        ("Scale", "Millions", "Billions"),
        ("Best for", "Dev + production", "Ultra-fast at scale"),
        ("Namespaces/Collections", "✅ Yes", "❌ Separate indexes"),
    ]

    for row in rows:
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    console.print("[bold cyan]🗄️  Day 2: Vector Databases[/bold cyan]\n")
    os.makedirs("week3/day2_vector_dbs", exist_ok=True)

    demo_chromadb()
    demo_faiss()
    comparison_table()

    console.print("\n[green]✅ Day 2 Complete! Vector databases mastered.[/green]")