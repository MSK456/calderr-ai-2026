"""
Lab 3.3: Hybrid Retrieval + RAGAS Evaluation
- BM25 + Semantic = Hybrid (EnsembleRetriever)
- Cross-encoder re-ranking
- RAGAS: faithfulness, answer_relevancy, context_precision
"""

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()
console = Console()

CORPUS = """
Python was created by Guido van Rossum and released in 1991. It emphasizes code readability.
Python supports multiple paradigms: object-oriented, functional, and procedural programming.
The Python Package Index (PyPI) hosts over 400,000 open-source packages for Python.
Python virtual environments isolate project dependencies using venv or conda.
FastAPI is a modern Python web framework that auto-generates OpenAPI documentation.
Django is a batteries-included web framework following the MVC pattern.
Flask is a lightweight Python microframework for building simple web applications.

LangChain is an open-source framework for building LLM-powered applications.
LangChain Expression Language (LCEL) uses the pipe operator to compose components.
LangChain supports memory, tools, agents, and retrieval components out of the box.
LangGraph extends LangChain for building complex stateful multi-actor workflows.
LangSmith provides tracing and observability for LangChain applications.

RAG (Retrieval-Augmented Generation) combines retrieval with language model generation.
RAG was introduced in 2020 by Lewis et al. in a paper on knowledge-intensive NLP.
RAG reduces hallucinations by grounding responses in retrieved factual context.
Hybrid search combines BM25 keyword matching with semantic vector similarity.
BM25 (Best Matching 25) is a probabilistic ranking function for information retrieval.
Cross-encoders jointly encode query and document pairs for accurate re-ranking.
RAGAS evaluates RAG pipelines using faithfulness, answer relevancy, and context precision.
Faithfulness measures whether the answer is supported by the retrieved context.
Answer relevancy measures whether the answer addresses the question asked.
Context precision measures whether the retrieved chunks are relevant to the query.

ChromaDB is an open-source vector database with metadata filtering support.
FAISS enables billion-scale similarity search using approximate nearest neighbor algorithms.
Sentence-transformers produce dense embeddings optimized for semantic similarity tasks.
The all-MiniLM-L6-v2 model produces 384-dimensional embeddings with good speed-accuracy balance.
Cosine similarity measures semantic closeness between two embedding vectors.
"""

QA_PAIRS = [
    {"question": "Who created Python?", "ground_truth": "Python was created by Guido van Rossum."},
    {"question": "What is LangChain used for?", "ground_truth": "LangChain is a framework for building LLM-powered applications."},
    {"question": "What is RAG?", "ground_truth": "RAG combines retrieval with language model generation to produce accurate responses."},
    {"question": "What does faithfulness measure in RAGAS?", "ground_truth": "Faithfulness measures whether the answer is supported by the retrieved context."},
    {"question": "What is BM25?", "ground_truth": "BM25 is a probabilistic ranking function for information retrieval."},
    {"question": "What embedding model produces 384-dimensional vectors?", "ground_truth": "all-MiniLM-L6-v2 produces 384-dimensional embeddings."},
    {"question": "What is LCEL in LangChain?", "ground_truth": "LCEL is LangChain Expression Language that uses the pipe operator to compose components."},
]


def build_documents():
    doc = Document(page_content=CORPUS, metadata={"source": "lab33_corpus"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents([doc])


def build_retriever(strategy: str, docs: list):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if strategy == "bm25":
        return BM25Retriever.from_documents(docs, k=4)

    elif strategy == "semantic":
        vs = Chroma.from_documents(docs, embeddings, collection_name="lab33_semantic")
        return vs.as_retriever(search_kwargs={"k": 4})

    elif strategy == "hybrid":
        bm25 = BM25Retriever.from_documents(docs, k=4)
        vs = Chroma.from_documents(docs, embeddings, collection_name="lab33_hybrid")
        semantic = vs.as_retriever(search_kwargs={"k": 4})
        return EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.4, 0.6])


def build_chain(retriever, use_reranker: bool = False):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if use_reranker else None

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the context provided. Be concise and accurate."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def get_context(query: str) -> str:
        docs = retriever.invoke(query)
        if reranker:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return "\n\n".join(doc.page_content for doc in docs[:3])

    def get_docs(query: str):
        docs = retriever.invoke(query)
        if reranker:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = reranker.predict(pairs)
            docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return docs[:3]

    chain = (
        {"context": lambda q: get_context(q), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return chain, get_docs


def build_ragas_dataset(chain, get_docs_fn) -> Dataset:
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for qa in QA_PAIRS:
        q = qa["question"]
        docs = get_docs_fn(q)
        answer = chain.invoke(q)

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append([d.page_content for d in docs])
        data["ground_truth"].append(qa["ground_truth"])

    return Dataset.from_dict(data)


def run_ragas(dataset: Dataset) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings)
    )
    df = result.to_pandas()
    return {
        "faithfulness": df["faithfulness"].mean() if "faithfulness" in df else 0.0,
        "answer_relevancy": df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0,
        "context_precision": df["context_precision"].mean() if "context_precision" in df else 0.0,
    }


if __name__ == "__main__":
    console.print("[bold cyan]🔬 Lab 3.3: Hybrid Retrieval + RAGAS Evaluation[/bold cyan]\n")

    docs = build_documents()
    console.print(f"Created {len(docs)} chunks\n")

    strategies = [
        ("BM25 only",             "bm25",     False),
        ("Semantic only",         "semantic",  False),
        ("Hybrid (BM25+Semantic)","hybrid",    False),
        ("Hybrid + Re-ranking",   "hybrid",    True),
    ]

    all_results = []

    for label, strategy, rerank in strategies:
        console.print(f"[yellow]Testing: {label}...[/yellow]")
        retriever = build_retriever(strategy, docs)
        chain, get_docs_fn = build_chain(retriever, use_reranker=rerank)
        dataset = build_ragas_dataset(chain, get_docs_fn)

        console.print(f"  Running RAGAS evaluation...")
        scores = run_ragas(dataset)
        scores["strategy"] = label
        all_results.append(scores)
        console.print(f"  Faithfulness: {scores['faithfulness']:.3f} | "
                      f"Relevancy: {scores['answer_relevancy']:.3f} | "
                      f"Precision: {scores['context_precision']:.3f}")

    # Final comparison table
    table = Table(title="Strategy Comparison (RAGAS)", box=box.ROUNDED)
    table.add_column("Strategy", style="cyan")
    table.add_column("Faithfulness", justify="center", style="yellow")
    table.add_column("Answer Relevancy", justify="center", style="green")
    table.add_column("Context Precision", justify="center", style="blue")
    table.add_column("Avg Score", justify="center", style="bold")

    for r in all_results:
        avg = (r["faithfulness"] + r["answer_relevancy"] + r["context_precision"]) / 3
        avg_color = "green" if avg > 0.7 else "yellow" if avg > 0.5 else "red"
        table.add_row(
            r["strategy"],
            f"{r['faithfulness']:.3f}",
            f"{r['answer_relevancy']:.3f}",
            f"{r['context_precision']:.3f}",
            f"[{avg_color}]{avg:.3f}[/{avg_color}]"
        )

    console.print(table)
    console.print("\n[green]✅ Lab 3.3 Complete! Hybrid + RAGAS evaluation done.[/green]")