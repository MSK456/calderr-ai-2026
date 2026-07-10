"""
Day 3: Naive RAG Pipeline
Load → Split → Embed → Store → Retrieve → Generate
Chunk size experiments: 256 vs 512 vs 1024
"""

import os
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
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()
console = Console()

# ── SAMPLE KNOWLEDGE BASE ─────────────────────────────────────────────────────

KNOWLEDGE_BASE = """
# Artificial Intelligence and Machine Learning

## What is Machine Learning?
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Machine learning focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future.

## Types of Machine Learning
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm is trained on labeled data, where the correct output is provided for each input. Common examples include classification and regression tasks. In unsupervised learning, the algorithm must find patterns in unlabeled data. Clustering and dimensionality reduction are common unsupervised techniques. Reinforcement learning involves an agent learning to make decisions by receiving rewards or penalties for its actions.

## Deep Learning and Neural Networks
Deep learning is a subset of machine learning that uses neural networks with many layers. Neural networks are inspired by the human brain and consist of interconnected nodes organized in layers. Deep learning has revolutionized computer vision, natural language processing, and speech recognition. Convolutional neural networks (CNNs) are commonly used for image recognition tasks. Recurrent neural networks (RNNs) and transformers are used for sequential data like text.

## Transformer Architecture
The transformer architecture, introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017, has revolutionized natural language processing. Transformers use a self-attention mechanism that allows the model to weigh the importance of different words when processing a sequence. Unlike RNNs, transformers can process all tokens in parallel, making them much faster to train. BERT, GPT, and T5 are all based on the transformer architecture. The transformer's ability to capture long-range dependencies has made it the dominant architecture for NLP tasks.

## Large Language Models
Large language models (LLMs) are trained on massive amounts of text data using the transformer architecture. Models like GPT-4, Claude, and LLaMA have billions of parameters and can generate coherent, contextually appropriate text. LLMs are trained using next-token prediction: given a sequence of tokens, predict the next token. Through this simple objective, LLMs develop emergent capabilities like reasoning, code generation, and few-shot learning. The scale of training data and model parameters significantly impacts performance.

## Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. Instead of relying solely on knowledge stored in model parameters, RAG retrieves relevant documents from an external knowledge base and provides them as context to the language model. This allows LLMs to answer questions about documents they were never trained on, reduces hallucinations by grounding responses in retrieved facts, and enables knowledge to be updated without retraining. RAG consists of two main components: the retriever, which finds relevant documents, and the generator, which produces answers based on the retrieved context.

## Vector Databases
Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently. When text is converted to embeddings (numerical vector representations), similar texts have similar vectors. Vector databases use approximate nearest neighbor (ANN) algorithms to find the most similar vectors quickly. Popular vector databases include ChromaDB, FAISS, Pinecone, Weaviate, and Qdrant. ChromaDB is popular for local development and prototyping. FAISS is developed by Facebook AI and optimized for extremely fast similarity search. The choice of vector database depends on scale, latency requirements, and infrastructure constraints.

## Embeddings and Semantic Search
Embeddings are dense vector representations of text that capture semantic meaning. Unlike sparse keyword-based representations, embeddings place semantically similar texts close together in vector space. Models like sentence-transformers produce embeddings optimized for semantic similarity. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings and is a popular choice for its balance of speed and accuracy. Cosine similarity is the most common metric for comparing embeddings, measuring the angle between two vectors. A cosine similarity of 1.0 means identical direction (same meaning), while 0.0 means orthogonal (unrelated).

## Evaluation of RAG Systems
Evaluating RAG systems is challenging because we need to measure both retrieval quality and generation quality. RAGAS is a popular framework for evaluating RAG systems. Key metrics include faithfulness (whether the answer is supported by the retrieved context), answer relevancy (whether the answer addresses the question), context precision (whether the retrieved chunks are relevant), and context recall (whether all necessary information was retrieved). A good RAG system should score above 0.7 on all RAGAS metrics. Evaluation requires a test set of question-answer pairs and their corresponding ground truth context.
"""

TEST_QUESTIONS = [
    "What is machine learning?",
    "How do transformers work?",
    "What is RAG and why is it useful?",
    "What embedding models are commonly used?",
    "How are LLMs trained?",
]


# ── BUILD RAG ─────────────────────────────────────────────────────────────────

def build_rag(chunk_size: int, chunk_overlap: int = 50):
    """Build a complete RAG pipeline with given chunk size."""

    # 1. Create document
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "ai_knowledge_base"})

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    # 3. Embed + Store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"rag_chunk_{chunk_size}"
    )

    return vectorstore, chunks


def run_rag_query(vectorstore, question: str, k: int = 3) -> dict:
    """Run a single RAG query and return result + metadata."""

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."
Be concise and accurate."""),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.time()
    retrieved = retriever.invoke(question)
    answer = chain.invoke(question)
    elapsed = time.time() - start

    return {
        "question": question,
        "answer": answer,
        "num_chunks_retrieved": len(retrieved),
        "elapsed_ms": round(elapsed * 1000),
        "retrieved_chunks": [r.page_content[:100] for r in retrieved]
    }


# ── CHUNK SIZE EXPERIMENT ─────────────────────────────────────────────────────

def experiment_chunk_sizes():
    console.print("\n[bold cyan]EXPERIMENT: Chunk Size Comparison[/bold cyan]")
    console.print("[dim]Testing 256, 512, 1024 token chunks[/dim]\n")

    chunk_sizes = [256, 512, 1024]
    results = {}

    for size in chunk_sizes:
        console.print(f"[yellow]Building RAG with chunk_size={size}...[/yellow]")
        vectorstore, chunks = build_rag(size)
        console.print(f"  → {len(chunks)} chunks created")

        size_results = []
        for q in TEST_QUESTIONS[:3]:  # test first 3 questions
            result = run_rag_query(vectorstore, q, k=3)
            size_results.append(result)

        results[size] = {
            "num_chunks": len(chunks),
            "avg_ms": sum(r["elapsed_ms"] for r in size_results) // len(size_results),
            "results": size_results
        }

    # Display comparison
    table = Table(title="Chunk Size Impact", box=box.ROUNDED)
    table.add_column("Chunk Size", style="cyan", justify="center")
    table.add_column("# Chunks", style="yellow", justify="center")
    table.add_column("Avg Latency", style="green", justify="center")
    table.add_column("Tradeoff", style="dim")

    tradeoffs = {
        256: "More specific, may miss context",
        512: "Good balance (recommended)",
        1024: "More context, less precise retrieval"
    }

    for size, data in results.items():
        table.add_row(
            str(size),
            str(data["num_chunks"]),
            f"{data['avg_ms']}ms",
            tradeoffs[size]
        )

    console.print(table)

    # Show example answers
    console.print("\n[bold]Sample Answer Comparison (Q: 'What is RAG?')[/bold]")
    for size in chunk_sizes:
        rag_result = results[size]["results"][2]  # RAG question
        console.print(Panel(
            rag_result["answer"][:200] + "...",
            title=f"[cyan]chunk_size={size}[/cyan]",
            border_style="blue"
        ))


if __name__ == "__main__":
    console.print("[bold cyan]📚 Day 3: Naive RAG Pipeline[/bold cyan]\n")
    experiment_chunk_sizes()
    console.print("\n[green]✅ Day 3 Complete! RAG pipeline mastered.[/green]")