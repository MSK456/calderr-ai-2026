"""
Day 5: RAGAS Evaluation Framework
- Build evaluation dataset
- faithfulness, answer_relevancy, context_precision
- Using Groq as evaluator LLM
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

KNOWLEDGE = """
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
Python emphasizes code readability and simplicity. It uses indentation to define code blocks.
Python supports multiple programming paradigms including procedural, object-oriented, and functional.
Python is widely used in web development, data science, artificial intelligence, and automation.
The Python Package Index (PyPI) hosts over 400,000 packages for various use cases.

LangChain is an open-source framework designed for building LLM-powered applications.
LangChain provides components for chains, agents, memory, and retrieval systems.
LangChain Expression Language (LCEL) uses the pipe operator to compose components.
LangChain supports many LLM providers including OpenAI, Anthropic, and Groq.
LangGraph is an extension of LangChain for building stateful, multi-actor applications.

ChromaDB is an open-source vector database for AI applications.
ChromaDB can run in-memory, persist to disk, or run as a server.
ChromaDB supports metadata filtering, allowing queries to be constrained by document properties.
ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search.
ChromaDB collections can store embeddings, documents, and associated metadata.

RAG stands for Retrieval-Augmented Generation. It was introduced in a 2020 paper by Lewis et al.
RAG retrieves relevant documents from a knowledge base before generating an answer.
RAG reduces hallucinations by grounding the model's response in retrieved facts.
RAG allows LLMs to answer questions about documents not seen during training.
The two main components of RAG are the retriever and the generator.
"""

QA_PAIRS = [
    {
        "question": "Who created Python and when?",
        "ground_truth": "Python was created by Guido van Rossum in 1991."
    },
    {
        "question": "What is LangChain used for?",
        "ground_truth": "LangChain is an open-source framework for building LLM-powered applications."
    },
    {
        "question": "What algorithm does ChromaDB use for similarity search?",
        "ground_truth": "ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm."
    },
    {
        "question": "What does RAG stand for and who introduced it?",
        "ground_truth": "RAG stands for Retrieval-Augmented Generation, introduced in a 2020 paper by Lewis et al."
    },
    {
        "question": "What is LCEL in LangChain?",
        "ground_truth": "LangChain Expression Language (LCEL) uses the pipe operator to compose components."
    },
]


def build_simple_rag():
    """Build RAG pipeline for evaluation."""
    doc = Document(page_content=KNOWLEDGE)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents([doc])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="day5_eval")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the context provided. Be concise."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    return chain, retriever


def build_evaluation_dataset(chain, retriever):
    """Build dataset for RAGAS evaluation."""
    console.print("[yellow]Building evaluation dataset...[/yellow]")

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for qa in QA_PAIRS:
        q = qa["question"]
        docs = retriever.invoke(q)
        answer = chain.invoke(q)

        data["question"].append(q)
        data["answer"].append(answer)
        data["contexts"].append([d.page_content for d in docs])
        data["ground_truth"].append(qa["ground_truth"])

        console.print(f"  ✅ Q: {q[:50]}...")

    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset):
    """Run RAGAS evaluation with Groq LLM."""
    console.print("\n[yellow]Running RAGAS evaluation...[/yellow]")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    return result


def display_results(result):
    """Display RAGAS results."""
    table = Table(title="RAGAS Evaluation Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="yellow", justify="center")
    table.add_column("Meaning", style="dim")

    metrics_info = {
        "faithfulness": "Answer supported by retrieved context",
        "answer_relevancy": "Answer addresses the question",
        "context_precision": "Retrieved chunks are relevant",
    }

    scores = result.to_pandas()

    for metric, meaning in metrics_info.items():
        if metric in scores.columns:
            score = scores[metric].mean()
            color = "green" if score > 0.7 else "yellow" if score > 0.5 else "red"
            table.add_row(metric, f"[{color}]{score:.3f}[/{color}]", meaning)

    console.print(table)

    # Save results
    result.to_pandas().to_csv("week3/day5_rag_evaluation/ragas_results.csv", index=False)
    console.print("\n[green]✅ Results saved to ragas_results.csv[/green]")


if __name__ == "__main__":
    console.print("[bold cyan]📊 Day 5: RAGAS Evaluation[/bold cyan]\n")
    os.makedirs("week3/day5_rag_evaluation", exist_ok=True)

    chain, retriever = build_simple_rag()
    dataset = build_evaluation_dataset(chain, retriever)
    result = run_ragas_evaluation(dataset)
    display_results(result)

    console.print("\n[green]✅ Day 5 Complete! RAGAS evaluation mastered.[/green]")