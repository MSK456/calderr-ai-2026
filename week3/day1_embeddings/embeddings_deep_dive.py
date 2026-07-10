"""
Day 1: Embeddings Deep Dive
- What embeddings are
- Cosine similarity
- Model comparison: all-MiniLM-L6-v2 vs bge-small-en-v1.5
- 2D PCA visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── SENTENCES TO EMBED ────────────────────────────────────────────────────────

SENTENCES = [
    # AI/ML cluster
    "Machine learning models learn patterns from training data.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Large language models are trained on massive text datasets.",
    "Transformers use attention mechanisms to process sequences.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and search high-dimensional vectors.",
    "RAG combines retrieval with language model generation.",

    # Space/Science cluster
    "The solar system contains eight planets orbiting the sun.",
    "Black holes are regions where gravity is extremely strong.",
    "NASA launched the James Webb Space Telescope in 2021.",
    "Stars are massive balls of plasma held together by gravity.",
    "The Milky Way galaxy contains billions of stars.",
    "Mars has two small moons called Phobos and Deimos.",

    # Pakistan cluster
    "Islamabad is the capital city of Pakistan.",
    "Pakistan was founded on August 14, 1947.",
    "The Indus River flows through Pakistan.",
    "K2 is the second highest mountain in the world located in Pakistan.",
    "Lahore is known as the cultural capital of Pakistan.",
    "Karachi is the largest city and financial hub of Pakistan.",
]

QUERY_EXAMPLES = [
    "How do neural networks work?",
    "Tell me about space exploration",
    "What is the capital of Pakistan?",
]


# ── EXPERIMENT 1: Embed and compute similarity ────────────────────────────────

def experiment_cosine_similarity():
    console.print("\n[bold cyan]EXPERIMENT 1: Cosine Similarity[/bold cyan]")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed query and sentences
    query = "How do AI language models process text?"
    query_embedding = model.encode([query])
    sentence_embeddings = model.encode(SENTENCES)

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

    # Sort by similarity
    ranked = sorted(
        zip(SENTENCES, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    table = Table(title=f"Query: '{query}'", box=box.ROUNDED)
    table.add_column("Score", style="yellow", width=8, justify="center")
    table.add_column("Sentence", style="green")

    for sentence, score in ranked[:8]:
        table.add_row(f"{score:.4f}", sentence)

    console.print(table)

    console.print(f"\n[bold]Key insight:[/bold] Sentences about AI scored high ({ranked[0][1]:.3f})")
    console.print(f"Sentences about space/Pakistan scored low ({ranked[-1][1]:.3f})")
    console.print("→ Cosine similarity captures MEANING, not just word overlap!")


# ── EXPERIMENT 2: Model Comparison ───────────────────────────────────────────

def experiment_model_comparison():
    console.print("\n[bold cyan]EXPERIMENT 2: Model Comparison[/bold cyan]")

    models = {
        "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),
        "bge-small-en-v1.5": SentenceTransformer("BAAI/bge-small-en-v1.5"),
    }

    query = "machine learning neural network"
    test_sentences = [
        "Deep learning uses multiple neural network layers",      # should be HIGH
        "AI models are trained on large datasets",               # should be MEDIUM-HIGH
        "Islamabad is the capital of Pakistan",                  # should be LOW
        "The sun is a star in our solar system",                 # should be LOW
    ]

    table = Table(title="Model Comparison", box=box.ROUNDED)
    table.add_column("Sentence", style="cyan", width=45)
    for model_name in models:
        table.add_column(model_name[:15], style="yellow", justify="center")

    sentence_embeddings_by_model = {}
    query_embeddings_by_model = {}

    for model_name, model in models.items():
        query_emb = model.encode([query])
        sent_embs = model.encode(test_sentences)
        sims = cosine_similarity(query_emb, sent_embs)[0]
        sentence_embeddings_by_model[model_name] = sims
        query_embeddings_by_model[model_name] = query_emb

    for i, sentence in enumerate(test_sentences):
        scores = [f"{sentence_embeddings_by_model[m][i]:.4f}" for m in models]
        table.add_row(sentence[:45], *scores)

    console.print(table)


# ── EXPERIMENT 3: PCA Visualization ──────────────────────────────────────────

def experiment_pca_visualization():
    console.print("\n[bold cyan]EXPERIMENT 3: PCA Visualization[/bold cyan]")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(SENTENCES)

    console.print(f"Embedding shape: {embeddings.shape}")
    console.print(f"Each sentence = {embeddings.shape[1]}-dimensional vector")

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    console.print(f"After PCA: 2-dimensional")
    console.print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Plot
    colors = ["#58a6ff"] * 8 + ["#3fb950"] * 6 + ["#f0c000"] * 6
    labels = ["AI/ML"] * 8 + ["Space"] * 6 + ["Pakistan"] * 6

    plt.figure(figsize=(12, 8))
    plt.style.use("dark_background")

    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, c=colors[i], s=100, zorder=2)
        plt.annotate(
            SENTENCES[i][:30] + "...",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color=colors[i]
        )

    plt.title("Sentence Embeddings in 2D (PCA)", fontsize=14, color="white")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#58a6ff", label="AI/ML"),
        Patch(color="#3fb950", label="Space"),
        Patch(color="#f0c000", label="Pakistan"),
    ]
    plt.legend(handles=legend)
    plt.tight_layout()

    save_path = "week3/day1_embeddings/embeddings_pca.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    console.print(f"\n[green]✅ PCA plot saved: {save_path}[/green]")
    console.print("[bold]Key insight:[/bold] Similar-topic sentences cluster together in vector space!")


# ── EXPERIMENT 4: Embedding Properties ───────────────────────────────────────

def experiment_embedding_properties():
    console.print("\n[bold cyan]EXPERIMENT 4: Embedding Properties[/bold cyan]")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    pairs = [
        ("King", "Queen", "royalty pair"),
        ("Paris", "France", "capital-country pair"),
        ("Python", "Programming", "related tech"),
        ("Cat", "Airplane", "completely unrelated"),
        ("I love AI", "Artificial Intelligence is amazing", "same meaning, different words"),
    ]

    table = Table(title="Similarity Between Pairs", box=box.ROUNDED)
    table.add_column("Text A", style="cyan")
    table.add_column("Text B", style="green")
    table.add_column("Relation", style="dim")
    table.add_column("Cosine Sim", style="yellow", justify="center")

    for a, b, relation in pairs:
        emb_a = model.encode([a])
        emb_b = model.encode([b])
        sim = cosine_similarity(emb_a, emb_b)[0][0]
        color = "green" if sim > 0.5 else "yellow" if sim > 0.3 else "red"
        table.add_row(a, b, relation, f"[{color}]{sim:.4f}[/{color}]")

    console.print(table)


if __name__ == "__main__":
    console.print("[bold cyan]🔢 Day 1: Embeddings Deep Dive[/bold cyan]\n")

    experiment_cosine_similarity()
    experiment_model_comparison()
    experiment_pca_visualization()
    experiment_embedding_properties()

    console.print("\n[green]✅ Day 1 Complete! Embeddings mastered.[/green]")