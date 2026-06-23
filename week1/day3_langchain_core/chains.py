"""
Day 3: LangChain Core — 3 Chain Patterns + ChromaDB Q&A
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────────
# CHAIN PATTERN 1: Simple Q&A Chain
# ─────────────────────────────────────────────

def chain_pattern_1():
    print("\n" + "="*60)
    print("🔗 CHAIN 1: Simple Q&A (prompt → llm → parser)")
    print("="*60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI expert. Answer concisely in 2-3 sentences."),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": "What makes an AI agent different from a chatbot?"})
    print(f"Answer: {result}")


# ─────────────────────────────────────────────
# CHAIN PATTERN 2: Parallel Chain
# ─────────────────────────────────────────────

def chain_pattern_2():
    print("\n" + "="*60)
    print("🔗 CHAIN 2: Parallel (two analyses at once)")
    print("="*60)

    pros_prompt = ChatPromptTemplate.from_messages([
        ("system", "List 3 advantages only. Be brief."),
        ("user", "Topic: {topic}")
    ])

    cons_prompt = ChatPromptTemplate.from_messages([
        ("system", "List 3 disadvantages only. Be brief."),
        ("user", "Topic: {topic}")
    ])

    parallel_chain = RunnableParallel(
        pros=(pros_prompt | llm | StrOutputParser()),
        cons=(cons_prompt | llm | StrOutputParser()),
        original=RunnablePassthrough()
    )

    result = parallel_chain.invoke({"topic": "Agentic AI systems"})
    print(f"\n✅ PROS:\n{result['pros']}")
    print(f"\n❌ CONS:\n{result['cons']}")


# ─────────────────────────────────────────────
# CHAIN PATTERN 3: Chained transforms
# ─────────────────────────────────────────────

def chain_pattern_3():
    print("\n" + "="*60)
    print("🔗 CHAIN 3: Transform chain (topic → explain → simplify)")
    print("="*60)

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain this technical topic in detail."),
        ("user", "{topic}")
    ])

    simplify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Take this explanation and rewrite it like you're explaining to a 10-year-old."),
        ("user", "{explanation}")
    ])

    chain = (
        explain_prompt
        | llm
        | StrOutputParser()
        | (lambda x: {"explanation": x})
        | simplify_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"topic": "Vector embeddings in AI"})
    print(f"Kid-friendly explanation:\n{result}")


# ─────────────────────────────────────────────
# ChromaDB: Document Q&A
# ─────────────────────────────────────────────

def chromadb_qa():
    print("\n" + "="*60)
    print("📚 ChromaDB: Document Q&A System")
    print("="*60)

    # Sample documents (in real use, load from files)
    docs_text = """
    LangChain is a framework designed to simplify the creation of applications using large language models.
    It provides tools for chaining LLM calls, managing memory, and integrating with external data sources.

    ChromaDB is an open-source vector database designed for AI applications.
    It stores text as numerical embeddings and enables semantic similarity search.
    ChromaDB can run fully in-memory, perfect for prototyping and small applications.

    Groq is a company that built the LPU (Language Processing Unit), custom hardware for LLM inference.
    Groq's free tier provides approximately 14,400 requests per day across multiple models.
    Available models include llama-3.1-8b-instant, llama-3.3-70b-versatile, and mixtral-8x7b-32768.

    RAG stands for Retrieval Augmented Generation. It works by first retrieving relevant documents
    from a database, then passing them as context to an LLM to generate accurate, grounded answers.
    RAG reduces hallucinations and keeps answers up-to-date without retraining the model.
    """

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    docs = splitter.create_documents([docs_text])
    print(f"📄 Created {len(docs)} chunks from document")

    # Create embeddings (free, local, no API needed)
    print("🔄 Creating embeddings (this takes ~30 seconds first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Store in ChromaDB (in-memory)
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Build RAG chain
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
        ("user", "{question}")
    ])

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # Test questions
    questions = [
        "What is ChromaDB used for?",
        "How many requests does Groq provide per day?",
        "What does RAG stand for and how does it work?"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        answer = rag_chain.invoke(q)
        print(f"💬 {answer}")


if __name__ == "__main__":
    chain_pattern_1()
    chain_pattern_2()
    chain_pattern_3()
    chromadb_qa()
    print("\n✅ Day 3 Complete! You mastered LangChain chains + RAG.")