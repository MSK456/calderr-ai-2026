"""
Lab 3.2: Complete Naive RAG Pipeline
- PDF-style document ingestion
- ChromaDB with metadata
- Chunk size: 256/512/1024
- Retrieval accuracy measurement
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()
console = Console()

# ── MULTI-DOCUMENT CORPUS (simulating 50 PDF pages) ──────────────────────────

DOCUMENTS_RAW = [
    {
        "title": "Introduction to Python",
        "content": """Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991.
Python's design philosophy emphasizes code readability with its notable use of significant indentation.
Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.
Python is consistently ranked as one of the most popular programming languages. Python was designed to be highly extensible.
This extensibility led to a variety of third-party Python packages that provide specialized functionality.
The Python Package Index (PyPI) is the official third-party software repository for Python.
Python is an interpreted language. Python programs are run by an interpreter which converts Python code into machine instructions.
Python 3.0 was released in 2008 and introduced several incompatible changes to the language.
Python's standard library is very extensive, offering a wide range of facilities including regular expressions, internet protocols, and web services.
Virtual environments allow Python users to install packages in isolated environments.
Popular Python web frameworks include Django, Flask, and FastAPI.
NumPy and Pandas are popular Python libraries for data manipulation and analysis.
TensorFlow, PyTorch, and scikit-learn are Python libraries for machine learning.
Python supports list comprehensions, generators, and decorators as advanced language features.
PEP 8 is Python's style guide providing conventions for writing readable Python code.""",
        "source": "python_docs",
        "page": 1
    },
    {
        "title": "Machine Learning Fundamentals",
        "content": """Machine learning is a method of data analysis that automates analytical model building.
It is based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
Supervised learning trains models on labeled examples where the correct output is provided for each input.
Unsupervised learning finds hidden patterns in data that has no predefined labels.
Reinforcement learning trains agents to make sequential decisions through reward signals.
Classification predicts discrete category labels for input data points.
Regression predicts continuous numerical values for input data points.
Clustering groups similar data points together without predefined labels.
Feature engineering transforms raw data into features suitable for machine learning models.
Cross-validation assesses how well a model generalizes to independent data sets.
Overfitting occurs when a model learns the training data too well, including its noise.
Regularization techniques like L1 and L2 penalize model complexity to prevent overfitting.
The bias-variance tradeoff describes the tension between underfitting and overfitting.
Ensemble methods combine multiple models to improve predictive performance.
Gradient boosting builds models sequentially, with each new model correcting previous errors.
Random forests combine multiple decision trees trained on different data subsets.
Support vector machines find the optimal hyperplane separating different classes.
Decision trees partition data recursively based on feature values.
Neural networks consist of interconnected layers of artificial neurons.
Backpropagation calculates gradients used to update neural network weights.""",
        "source": "ml_textbook",
        "page": 2
    },
    {
        "title": "Vector Databases and Embeddings",
        "content": """Vector databases are specialized systems designed to store and query high-dimensional vectors.
Embeddings are dense vector representations that capture the semantic meaning of data.
Similar items have embeddings that are close to each other in the vector space.
Cosine similarity measures the angle between two vectors, independent of their magnitude.
Dot product similarity is another common metric for comparing vectors.
Euclidean distance measures the straight-line distance between two vectors.
HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.
FAISS (Facebook AI Similarity Search) provides efficient similarity search and clustering of dense vectors.
ChromaDB is an open-source vector database with a simple Python API.
Pinecone is a managed vector database service designed for production workloads.
Weaviate is an open-source vector database with built-in machine learning capabilities.
Qdrant is a vector similarity search engine written in Rust for high performance.
Vector indexes can be stored in memory for fast access or persisted to disk for durability.
Metadata filtering allows vector search to be constrained by document properties.
Namespaces in vector databases allow multiple separate indexes within a single instance.
Batch operations improve performance when adding large numbers of vectors.
The embedding dimension determines the granularity of the vector representation.
Sentence-transformers library provides pre-trained models optimized for semantic similarity.
all-MiniLM-L6-v2 produces 384-dimensional embeddings and is popular for its balance of speed and accuracy.
Large embedding models like text-embedding-ada-002 produce 1536-dimensional vectors.""",
        "source": "vector_db_guide",
        "page": 3
    },
    {
        "title": "RAG Systems Architecture",
        "content": """Retrieval-Augmented Generation (RAG) was introduced in a landmark 2020 paper by Lewis et al.
RAG combines information retrieval with language model generation to produce accurate responses.
The ingestion pipeline converts raw documents into searchable vector representations.
Document loaders support various formats including PDF, DOCX, HTML, and plain text.
Text splitters divide documents into smaller chunks that fit within model context windows.
Chunk size determines the granularity of retrieved information.
Chunk overlap prevents information loss at chunk boundaries.
The query pipeline retrieves relevant chunks and generates answers from retrieved context.
Naive RAG performs basic similarity search to retrieve relevant documents.
Advanced RAG incorporates techniques like re-ranking, query expansion, and hybrid search.
Modular RAG breaks the pipeline into independent components that can be swapped.
Context stuffing provides all retrieved chunks to the language model simultaneously.
Iterative retrieval performs multiple retrieval steps based on intermediate model outputs.
Self-RAG allows the model to decide when and what to retrieve.
HyDE (Hypothetical Document Embeddings) generates a hypothetical answer before retrieval.
Multi-query retrieval generates multiple query variations to improve recall.
Parent document retrieval stores small chunks but retrieves larger parent documents.
Reciprocal Rank Fusion combines rankings from multiple retrieval methods.
Lost in the middle is a phenomenon where models ignore context in the middle of long inputs.
RAGAS provides automated evaluation of RAG systems using reference-free metrics.""",
        "source": "rag_textbook",
        "page": 4
    },
    {
        "title": "LangChain Framework",
        "content": """LangChain is an open-source framework for building applications powered by language models.
LangChain provides a standard interface for interacting with various LLM providers.
LangChain Expression Language (LCEL) uses the pipe operator to compose chains of components.
Runnables are the fundamental abstraction in LCEL, supporting invoke, stream, and batch operations.
RunnablePassthrough passes input through unchanged to the next component in a chain.
RunnableParallel executes multiple runnables simultaneously with the same input.
Prompt templates define the structure of inputs to language models.
Output parsers convert raw LLM output into structured formats.
Memory components store conversation history for multi-turn interactions.
Document loaders connect LangChain to various data sources.
Text splitters divide documents into appropriate-sized chunks for processing.
Vector stores integrate with various vector databases for similarity search.
Retrievers fetch relevant documents based on a query.
Chains compose multiple components into reusable pipelines.
Agents use language models to dynamically decide which tools to call.
Tools are functions that agents can call to interact with external systems.
LangGraph extends LangChain with graph-based workflows for complex agent scenarios.
LangSmith provides observability and debugging for LangChain applications.
LangChain supports callbacks for monitoring and logging throughout the pipeline.
The LangChain Hub provides a repository of shareable prompts and chains.""",
        "source": "langchain_docs",
        "page": 5
    }
]

# Q&A Test Set (20 pairs)
QA_TEST_SET = [
    {"question": "Who created Python?", "expected_source": "python_docs", "answer_contains": "Guido van Rossum"},
    {"question": "When was Python first released?", "expected_source": "python_docs", "answer_contains": "1991"},
    {"question": "What is PyPI?", "expected_source": "python_docs", "answer_contains": "repository"},
    {"question": "What is supervised learning?", "expected_source": "ml_textbook", "answer_contains": "labeled"},
    {"question": "What is overfitting?", "expected_source": "ml_textbook", "answer_contains": "training"},
    {"question": "What does HNSW stand for?", "expected_source": "vector_db_guide", "answer_contains": "Hierarchical"},
    {"question": "What dimensions does all-MiniLM-L6-v2 produce?", "expected_source": "vector_db_guide", "answer_contains": "384"},
    {"question": "Who introduced RAG?", "expected_source": "rag_textbook", "answer_contains": "Lewis"},
    {"question": "What is HyDE in RAG?", "expected_source": "rag_textbook", "answer_contains": "Hypothetical"},
    {"question": "What operator does LCEL use?", "expected_source": "langchain_docs", "answer_contains": "pipe"},
]


class NaiveRAGPipeline:
    def __init__(self, chunk_size: int, chunk_overlap: int = 50, k: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.vectorstore = None
        self.chain = None
        self.num_chunks = 0
        self._build()

    def _build(self):
        # 1. Create documents
        raw_docs = [
            Document(
                page_content=d["content"],
                metadata={"source": d["source"], "page": d["page"], "title": d["title"]}
            )
            for d in DOCUMENTS_RAW
        ]

        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(raw_docs)
        self.num_chunks = len(chunks)

        # 3. Embed + Store
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=f"lab32_chunk{self.chunk_size}"
        )

        # 4. Build chain
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question using ONLY the provided context.
Be concise. If information is not in context, say "Not found in documents."
Always mention which document the information comes from."""),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        def format_docs(docs):
            formatted = []
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                formatted.append(f"[Source: {src}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        self.retriever = retriever

    def query(self, question: str) -> dict:
        start = time.time()
        retrieved = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        elapsed = (time.time() - start) * 1000
        return {
            "answer": answer,
            "sources": [d.metadata.get("source") for d in retrieved],
            "elapsed_ms": round(elapsed)
        }

    def evaluate(self) -> dict:
        """Evaluate on test set."""
        correct_source = 0
        correct_answer = 0
        total = len(QA_TEST_SET)

        for qa in QA_TEST_SET:
            result = self.query(qa["question"])
            if qa["expected_source"] in result["sources"]:
                correct_source += 1
            if qa["answer_contains"].lower() in result["answer"].lower():
                correct_answer += 1

        return {
            "source_accuracy": correct_source / total,
            "answer_accuracy": correct_answer / total,
            "total": total
        }


if __name__ == "__main__":
    console.print("[bold cyan]📚 Lab 3.2: Naive RAG Pipeline[/bold cyan]\n")
    os.makedirs("week3/labs", exist_ok=True)

    chunk_configs = [
        (256, 30, 3),
        (512, 50, 3),
        (1024, 100, 3),
        (512, 50, 5),   # k=5 experiment
    ]

    results = []

    for chunk_size, overlap, k in chunk_configs:
        label = f"chunk={chunk_size}, overlap={overlap}, k={k}"
        console.print(f"\n[yellow]Building: {label}...[/yellow]")

        pipeline = NaiveRAGPipeline(chunk_size=chunk_size, chunk_overlap=overlap, k=k)
        console.print(f"  → {pipeline.num_chunks} chunks created")

        console.print("  → Evaluating...")
        eval_results = pipeline.evaluate()
        eval_results["config"] = label
        eval_results["num_chunks"] = pipeline.num_chunks
        results.append(eval_results)

        console.print(f"  → Source accuracy: {eval_results['source_accuracy']:.0%}")
        console.print(f"  → Answer accuracy: {eval_results['answer_accuracy']:.0%}")

    # Display comparison
    table = Table(title="RAG Configuration Comparison", box=box.ROUNDED)
    table.add_column("Config", style="cyan")
    table.add_column("# Chunks", justify="center")
    table.add_column("Source Acc", justify="center", style="yellow")
    table.add_column("Answer Acc", justify="center", style="green")

    for r in results:
        src = f"{r['source_accuracy']:.0%}"
        ans = f"{r['answer_accuracy']:.0%}"
        src_color = "green" if r["source_accuracy"] > 0.7 else "yellow"
        ans_color = "green" if r["answer_accuracy"] > 0.7 else "yellow"
        table.add_row(
            r["config"],
            str(r["num_chunks"]),
            f"[{src_color}]{src}[/{src_color}]",
            f"[{ans_color}]{ans}[/{ans_color}]"
        )

    console.print(table)

    # Save results
    with open("week3/labs/lab3_2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print("\n[green]✅ Results saved to lab3_2_results.json[/green]")
    console.print("[green]✅ Lab 3.2 Complete![/green]")