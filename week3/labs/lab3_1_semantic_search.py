"""
Lab 3.1: Semantic Search CLI
- 100 Wikipedia-style sentences
- Two embedding models compared
- Interactive CLI
"""

import os
import sys
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

console = Console()

# ── 100 FACTUAL SENTENCES ─────────────────────────────────────────────────────

SENTENCES = [
    # AI/ML (25)
    "Machine learning enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with multiple hidden layers.",
    "Transformers use self-attention to process text in parallel.",
    "BERT is a bidirectional transformer model pre-trained on masked language modeling.",
    "GPT models generate text by predicting the next token in a sequence.",
    "Reinforcement learning trains agents through reward and penalty signals.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Recurrent neural networks process sequential data like time series.",
    "Transfer learning reuses knowledge from one task to improve another.",
    "Gradient descent optimizes neural network weights by minimizing loss.",
    "Overfitting occurs when a model learns noise instead of patterns.",
    "Cross-validation evaluates model performance across multiple data splits.",
    "Embeddings represent text as dense numerical vectors capturing meaning.",
    "Cosine similarity measures the angle between two embedding vectors.",
    "Vector databases store embeddings for fast similarity search.",
    "RAG retrieves relevant documents before generating an answer.",
    "ChromaDB is an open-source vector database for AI applications.",
    "FAISS enables efficient similarity search over millions of vectors.",
    "LangChain provides components for building LLM-powered applications.",
    "Pydantic v2 validates data using Python type hints with Rust speed.",
    "FastAPI builds REST APIs with automatic documentation generation.",
    "Docker packages applications with dependencies for consistent deployment.",
    "Kubernetes orchestrates containerized applications across clusters.",
    "Python is the dominant language for machine learning and data science.",
    "The attention mechanism allows models to focus on relevant context.",

    # Space/Astronomy (20)
    "The solar system has eight planets orbiting the sun.",
    "Black holes are regions where gravity prevents even light from escaping.",
    "The James Webb Space Telescope observes galaxies from the early universe.",
    "Mars has two moons: Phobos and Deimos.",
    "Jupiter is the largest planet with a mass greater than all other planets combined.",
    "The Milky Way is a barred spiral galaxy containing over 200 billion stars.",
    "Neutron stars are incredibly dense remnants of massive stellar explosions.",
    "The Hubble Space Telescope orbits Earth at about 340 miles altitude.",
    "Pluto was reclassified as a dwarf planet in 2006 by the IAU.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "A light year is the distance light travels in one year.",
    "The Big Bang occurred approximately 13.8 billion years ago.",
    "Saturn's rings are made of ice and rock particles orbiting the planet.",
    "The Andromeda Galaxy is the nearest spiral galaxy to the Milky Way.",
    "Solar flares are explosions on the sun's surface releasing radiation.",
    "Comets are icy bodies that develop tails when near the sun.",
    "Asteroids orbit the sun primarily in the asteroid belt between Mars and Jupiter.",
    "The International Space Station orbits Earth at approximately 400 km altitude.",
    "Exoplanets are planets orbiting stars other than our sun.",
    "Dark matter makes up about 27% of the universe's mass-energy content.",

    # Pakistan/South Asia (20)
    "Pakistan was founded on August 14, 1947 by Muhammad Ali Jinnah.",
    "Islamabad is the capital city of Pakistan built in the 1960s.",
    "Karachi is Pakistan's largest city and main financial center.",
    "Lahore is known as Pakistan's cultural capital and heart of Punjab.",
    "K2 at 8,611 meters is the world's second highest mountain in Pakistan.",
    "The Indus River is one of the longest rivers flowing through Pakistan.",
    "Urdu is Pakistan's national language used in government and education.",
    "Punjab is Pakistan's most populous province with over 110 million people.",
    "The Mohenjo-daro ruins are part of the ancient Indus Valley Civilization.",
    "Pakistan shares borders with India, Afghanistan, Iran, and China.",
    "FAST University (NUCES) is a leading computer science institution in Pakistan.",
    "The Pakistan Super League (PSL) is a major cricket franchise tournament.",
    "Peshawar is the capital of Khyber Pakhtunkhwa province.",
    "Quetta is the capital of Balochistan, Pakistan's largest province.",
    "The Karakoram Highway is one of the highest paved roads in the world.",
    "Taxila is an ancient city and UNESCO World Heritage Site near Islamabad.",
    "Pakistan's currency is the Pakistani Rupee (PKR).",
    "The Thar Desert is located in southeastern Pakistan and India.",
    "Sialkot is famous for manufacturing sports goods and surgical instruments.",
    "The Badshahi Mosque in Lahore was built in 1673 during the Mughal era.",

    # Technology/Programming (20)
    "Git is a distributed version control system for tracking code changes.",
    "REST APIs use HTTP methods like GET, POST, PUT, DELETE for communication.",
    "SQL databases store data in structured tables with relationships.",
    "NoSQL databases store unstructured or semi-structured data flexibly.",
    "JSON is a lightweight data format for transmitting data between systems.",
    "APIs allow different software applications to communicate with each other.",
    "Cloud computing delivers computing services over the internet on demand.",
    "AWS, Azure, and Google Cloud are the three major cloud providers.",
    "Microservices architecture splits applications into small, independent services.",
    "CI/CD pipelines automate the testing and deployment of software.",
    "React is a JavaScript library for building user interfaces.",
    "TypeScript adds static type checking to JavaScript.",
    "GraphQL is a query language for APIs providing flexible data fetching.",
    "WebSockets enable real-time bidirectional communication between clients and servers.",
    "Redis is an in-memory data store used for caching and sessions.",
    "PostgreSQL is an advanced open-source relational database system.",
    "Nginx is a high-performance web server and reverse proxy.",
    "Linux is the dominant operating system for servers and cloud infrastructure.",
    "SSH provides secure encrypted remote access to computer systems.",
    "HTTPS encrypts data between web browsers and servers using TLS.",

    # Science/Biology (15)
    "DNA carries genetic information in sequences of nucleotides.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The human genome contains approximately 3 billion base pairs.",
    "CRISPR-Cas9 is a gene editing tool allowing precise DNA modification.",
    "Viruses replicate by injecting their genetic material into host cells.",
    "Antibiotics kill bacteria but are ineffective against viral infections.",
    "The human brain contains approximately 86 billion neurons.",
    "Evolution occurs through natural selection and genetic mutation.",
    "mRNA vaccines train the immune system to recognize protein antigens.",
    "Climate change is driven primarily by greenhouse gas emissions.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "Quantum computers use qubits that can be in superposition of states.",
    "3D printing creates objects layer by layer from digital designs.",
    "The periodic table organizes elements by atomic number and properties.",
    "Gravity is a fundamental force that attracts objects with mass.",
]


# ── SEMANTIC SEARCH ENGINE ────────────────────────────────────────────────────

class SemanticSearchEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.sentences = SENTENCES
        self.embeddings = None
        self._index()

    def _index(self):
        start = time.time()
        self.embeddings = self.model.encode(
            self.sentences,
            batch_size=32,
            show_progress_bar=False
        )
        elapsed = time.time() - start
        console.print(f"  [{self.model_name}] Indexed {len(self.sentences)} sentences in {elapsed:.2f}s")
        console.print(f"  Embedding shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        start = time.time()
        query_emb = self.model.encode([query])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        elapsed = (time.time() - start) * 1000

        return [
            {
                "rank": i + 1,
                "sentence": self.sentences[idx],
                "score": float(sims[idx]),
                "elapsed_ms": elapsed
            }
            for i, idx in enumerate(top_indices)
        ]


# ── MODEL COMPARISON ──────────────────────────────────────────────────────────

def compare_models():
    console.print("\n[bold cyan]Loading both embedding models...[/bold cyan]")

    engines = {
        "MiniLM-L6-v2": SemanticSearchEngine("all-MiniLM-L6-v2"),
        "BGE-small-en":  SemanticSearchEngine("BAAI/bge-small-en-v1.5"),
    }

    test_queries = [
        "machine learning neural network training",
        "planets and stars in space",
        "Pakistan capital city culture",
        "programming language for AI",
    ]

    for query in test_queries:
        console.print(f"\n[bold]Query: '[yellow]{query}[/yellow]'[/bold]")

        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Rank", width=4, justify="center")
        table.add_column("MiniLM-L6-v2 (score)", style="cyan")
        table.add_column("BGE-small-en (score)", style="green")

        results = {name: eng.search(query, top_k=3) for name, eng in engines.items()}

        for i in range(3):
            r1 = results["MiniLM-L6-v2"][i]
            r2 = results["BGE-small-en"][i]
            table.add_row(
                str(i + 1),
                f"{r1['sentence'][:40]}... [{r1['score']:.3f}]",
                f"{r2['sentence'][:40]}... [{r2['score']:.3f}]",
            )

        console.print(table)


# ── INTERACTIVE CLI ───────────────────────────────────────────────────────────

def interactive_cli():
    console.print(Panel.fit(
        "[bold cyan]🔍 Semantic Search CLI[/bold cyan]\n"
        "[dim]100 sentences | all-MiniLM-L6-v2 | Type 'exit' to quit[/dim]",
        border_style="cyan"
    ))

    engine = SemanticSearchEngine("all-MiniLM-L6-v2")

    while True:
        query = Prompt.ask("\n[bold blue]Search[/bold blue]").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query:
            continue

        results = engine.search(query, top_k=5)

        table = Table(title=f"Results for: '{query}'", box=box.ROUNDED)
        table.add_column("Rank", width=5, justify="center", style="dim")
        table.add_column("Score", width=7, justify="center", style="yellow")
        table.add_column("Sentence", style="green")

        for r in results:
            color = "green" if r["score"] > 0.6 else "yellow" if r["score"] > 0.4 else "red"
            table.add_row(
                str(r["rank"]),
                f"[{color}]{r['score']:.4f}[/{color}]",
                r["sentence"]
            )

        console.print(table)
        console.print(f"[dim]Search time: {results[0]['elapsed_ms']:.1f}ms[/dim]")


if __name__ == "__main__":
    console.print("[bold cyan]🔍 Lab 3.1: Semantic Search CLI[/bold cyan]\n")

    compare_models()
    interactive_cli()

    console.print("\n[green]✅ Lab 3.1 Complete![/green]")