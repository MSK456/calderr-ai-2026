"""
RAG Pipeline builder — creates different RAG configurations from YAML config
"""

import os
import logging
import time
from dataclasses import dataclass
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

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    config_id: str
    label: str
    embedding_model: str
    chunk_size: int
    overlap: int
    k: int
    strategy: str


@dataclass
class PipelineResult:
    config_id: str
    label: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    latency_ms: float
    num_chunks_total: int


class RAGPipeline:
    """Configurable RAG pipeline for benchmarking."""

    def __init__(self, config: PipelineConfig, documents: list[Document]):
        self.config = config
        self.documents = documents
        self._build()

    def _build(self):
        """Build the pipeline based on configuration."""
        # Text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(self.documents)
        self.num_chunks = len(chunks)
        logger.debug(f"[{self.config.label}] Created {len(chunks)} chunks")

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": "cpu"}
        )

        # Vector store
        collection_name = f"bench_{self.config.config_id}_{self.config.embedding_model.replace('/', '_').replace('-', '_')}"
        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            collection_name=collection_name[:63]  # chroma has 63 char limit
        )
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.k}
        )

        # Strategy
        if self.config.strategy == "hybrid":
            bm25 = BM25Retriever.from_documents(chunks, k=self.config.k)
            self.retriever = EnsembleRetriever(
                retrievers=[bm25, semantic_retriever],
                weights=[0.4, 0.6]
            )
        else:
            self.retriever = semantic_retriever

        # LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question accurately using ONLY the provided context.
Be concise. If information is not in the context, state "Information not found."
Do not add information not present in the context."""),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def run(self, question: str, ground_truth: str) -> PipelineResult:
        """Run a single question through the pipeline."""
        start = time.time()

        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)

        elapsed = (time.time() - start) * 1000

        return PipelineResult(
            config_id=self.config.config_id,
            label=self.config.label,
            question=question,
            answer=answer,
            contexts=[doc.page_content for doc in retrieved_docs[:self.config.k]],
            ground_truth=ground_truth,
            latency_ms=round(elapsed, 2),
            num_chunks_total=self.num_chunks
        )