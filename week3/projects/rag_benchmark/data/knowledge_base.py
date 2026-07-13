"""
Knowledge base for RAG benchmark evaluation.
Contains rich, factual content across multiple domains.
"""

KNOWLEDGE_DOCUMENTS = {
    "python_advanced": """
Python Advanced Features and Best Practices

Python 3.11 introduced significant performance improvements through the Faster CPython project.
The new adaptive interpreter can be 10-60% faster than Python 3.10 for certain workloads.
Python type hints, introduced in PEP 484, allow static type checking with tools like mypy and pyright.
Dataclasses, introduced in Python 3.7, provide a decorator-based way to create classes with automatic methods.
Context managers using the 'with' statement ensure proper resource cleanup through __enter__ and __exit__ methods.
Async/await syntax, added in Python 3.5, enables efficient concurrent I/O without multi-threading complexity.
Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python bytecode across threads.
The asyncio library provides an event loop for running asynchronous code in a single thread.
Generator functions using 'yield' enable lazy evaluation, processing items one at a time to save memory.
List comprehensions are faster than equivalent for loops due to optimized C implementation in CPython.
Python decorators are functions that wrap other functions to extend behavior without modification.
The functools.lru_cache decorator provides memoization for expensive function calls.
Python's unittest and pytest frameworks provide comprehensive testing capabilities for Python applications.
Virtual environments created with venv or conda isolate project dependencies to avoid conflicts.
Python packaging with pyproject.toml (PEP 517/518) is the modern standard for distributing packages.
""",

    "rag_systems": """
RAG Systems: Architecture and Best Practices

Retrieval-Augmented Generation was introduced by Lewis et al. in 2020 as a solution for knowledge-intensive NLP tasks.
The core insight of RAG is that parametric memory (stored in model weights) can be supplemented with non-parametric memory (retrieved documents).
RAG consists of two main phases: the indexing phase and the querying phase.
During indexing, documents are loaded, split into chunks, embedded, and stored in a vector database.
During querying, the user's question is embedded, similar chunks are retrieved, and an LLM generates an answer using those chunks.
Chunk size is a critical hyperparameter: smaller chunks are more precise but may lack context; larger chunks have more context but lower retrieval precision.
Chunk overlap prevents important information from being split across chunk boundaries.
Naive RAG uses simple top-k semantic retrieval without any advanced techniques.
Advanced RAG techniques include HyDE, multi-query, re-ranking, and parent document retrieval.
HyDE (Hypothetical Document Embeddings) generates a hypothetical answer first, then uses it to retrieve real documents.
Multi-query retrieval generates multiple query variations to improve retrieval recall.
Cross-encoder re-ranking uses a more expensive model to re-score retrieved documents for better precision.
Parent document retrieval stores small chunks but retrieves their larger parent documents for more context.
Hybrid search combines BM25 keyword matching with dense semantic retrieval using ensemble weighting.
RAGAS provides automated evaluation metrics: faithfulness, answer relevancy, context precision, and context recall.
""",

    "vector_databases": """
Vector Databases: Comprehensive Guide

Vector databases are specialized data stores optimized for storing and querying high-dimensional vectors.
The primary operation in vector databases is approximate nearest neighbor (ANN) search.
Exact nearest neighbor search requires comparing a query vector against every stored vector, which is O(n).
ANN algorithms like HNSW, IVF, and ANNOY trade small accuracy losses for orders-of-magnitude speed improvements.
HNSW (Hierarchical Navigable Small World) builds a hierarchical graph structure enabling O(log n) search.
IVF (Inverted File Index) partitions vectors into clusters, searching only nearby clusters for a query.
Product Quantization (PQ) compresses vectors by encoding sub-spaces separately, reducing memory 8-32x.
ChromaDB uses HNSW by default and can run in-memory, persisted, or as a distributed server.
FAISS (Facebook AI Similarity Search) provides implementations of many ANN algorithms optimized for CPUs and GPUs.
Pinecone is a managed vector database with automatic scaling, replication, and global distribution.
Weaviate adds structured schema support, allowing hybrid queries combining vector and scalar filters.
Qdrant is written in Rust and designed for high-performance production deployments with filtering.
Metadata filtering allows constraining vector searches by document properties without post-filtering.
Namespaces or collections provide logical separation of document sets within a single vector database.
Index persistence allows reloading pre-built indexes without re-embedding documents from scratch.
""",

    "llm_evaluation": """
LLM and RAG Evaluation Frameworks

Evaluating LLMs and RAG systems requires both automated metrics and human evaluation.
RAGAS (Retrieval Augmented Generation Assessment) provides reference-free automated evaluation.
Faithfulness in RAGAS measures whether each claim in the answer is supported by the retrieved context.
Answer Relevancy in RAGAS measures how well the answer addresses the question asked.
Context Precision in RAGAS measures whether the retrieved chunks are relevant to the question.
Context Recall in RAGAS measures whether all necessary information was included in the retrieved context.
RAGAS uses an LLM as a judge, making it language-model dependent but human-like in evaluation.
TruLens provides an alternative evaluation framework with similar RAG triad metrics.
The RAG Triad (groundedness, context relevance, answer relevance) is a popular evaluation framework.
BLEU and ROUGE are traditional NLP metrics measuring n-gram overlap but are poor for open-ended generation.
BERTScore uses BERT embeddings to measure semantic similarity between generated and reference text.
Human evaluation remains the gold standard but is expensive, slow, and often inconsistent.
Evaluation datasets should cover diverse question types: factual, reasoning, multi-hop, and adversarial.
A/B testing in production with user feedback provides real-world effectiveness measurement.
Continuous evaluation pipelines automatically re-evaluate RAG systems as documents or models change.
""",

    "embeddings_theory": """
Embeddings: Theory and Practice

Embeddings are dense vector representations that encode semantic meaning in a continuous space.
Word2Vec, introduced by Mikolov et al. in 2013, was the pioneering neural embedding model.
Word2Vec uses two architectures: CBOW predicts target words from context; Skip-gram predicts context from target.
GloVe (Global Vectors) captures global co-occurrence statistics rather than local context windows.
Transformer-based models like BERT produce contextualized embeddings where the same word has different vectors in different contexts.
Sentence-Transformers fine-tune BERT-like models using siamese networks optimized for semantic similarity.
Contrastive learning trains embeddings so that similar pairs are close and dissimilar pairs are distant.
The all-MiniLM-L6-v2 model produces 384-dimensional embeddings with an excellent speed-accuracy balance.
BGE (BAAI General Embeddings) models are fine-tuned using RetroMAE and LLAMA-based distillation.
Larger embedding dimensions (e.g., 1536 for text-embedding-ada-002) capture more information but require more storage.
Cosine similarity computes the angle between vectors and is invariant to vector magnitude.
Dot product similarity is equivalent to cosine similarity for unit-normalized vectors.
Euclidean distance measures geometric distance and is less commonly used for text embeddings.
Semantic search using embeddings outperforms keyword search for paraphrase and synonym queries.
Batch encoding processes multiple texts simultaneously using GPU parallelism for faster embedding computation.
""",

    "langchain_advanced": """
LangChain Advanced Patterns

LangChain Expression Language (LCEL) provides a declarative way to compose AI pipelines using the pipe operator.
RunnablePassthrough passes input unchanged, useful for preserving the original query alongside processed results.
RunnableParallel executes multiple chains simultaneously and combines their outputs.
RunnableLambda wraps any Python function as a composable runnable in LCEL chains.
RunnableWithFallbacks provides graceful degradation when a primary chain fails.
RunnableBranch implements conditional logic, routing to different chains based on input conditions.
The .bind() method attaches fixed keyword arguments to a runnable for later invocation.
The .with_retry() method adds automatic retry logic to any runnable for handling transient failures.
Memory in LangChain stores conversation history for multi-turn interactions.
ConversationBufferMemory stores all messages; ConversationSummaryMemory stores a compressed summary.
Document loaders support PDF, DOCX, HTML, CSV, and over 100 other formats for ingestion.
RecursiveCharacterTextSplitter splits text preferring semantic boundaries like paragraphs and sentences.
MultiQueryRetriever generates multiple query variations and deduplicates results for better recall.
EnsembleRetriever combines multiple retrievers using Reciprocal Rank Fusion for hybrid search.
The RetrievalQA chain provides a simple interface for question-answering over retrieved documents.
"""
}


EVALUATION_QA_PAIRS = [
    # Python questions
    {"question": "What Python version introduced significant performance improvements through the Faster CPython project?", "ground_truth": "Python 3.11 introduced significant performance improvements through the Faster CPython project.", "domain": "python"},
    {"question": "What is the purpose of Python's Global Interpreter Lock?", "ground_truth": "The GIL prevents true parallel execution of Python bytecode across threads.", "domain": "python"},
    {"question": "What does the functools.lru_cache decorator provide?", "ground_truth": "functools.lru_cache provides memoization for expensive function calls.", "domain": "python"},

    # RAG questions
    {"question": "Who introduced RAG and in what year?", "ground_truth": "RAG was introduced by Lewis et al. in 2020.", "domain": "rag"},
    {"question": "What is HyDE in the context of RAG?", "ground_truth": "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer first, then uses it to retrieve real documents.", "domain": "rag"},
    {"question": "What is the difference between naive RAG and advanced RAG?", "ground_truth": "Naive RAG uses simple top-k semantic retrieval while advanced RAG includes techniques like HyDE, multi-query, re-ranking, and parent document retrieval.", "domain": "rag"},

    # Vector DB questions
    {"question": "What algorithm does ChromaDB use by default for similarity search?", "ground_truth": "ChromaDB uses HNSW (Hierarchical Navigable Small World) by default.", "domain": "vector_db"},
    {"question": "What is Product Quantization in vector databases?", "ground_truth": "Product Quantization compresses vectors by encoding sub-spaces separately, reducing memory 8-32x.", "domain": "vector_db"},
    {"question": "What is FAISS and who developed it?", "ground_truth": "FAISS (Facebook AI Similarity Search) provides implementations of ANN algorithms optimized for CPUs and GPUs, developed by Facebook AI Research.", "domain": "vector_db"},

    # Evaluation questions
    {"question": "What does faithfulness measure in RAGAS?", "ground_truth": "Faithfulness measures whether each claim in the answer is supported by the retrieved context.", "domain": "evaluation"},
    {"question": "What is context precision in RAG evaluation?", "ground_truth": "Context Precision measures whether the retrieved chunks are relevant to the question.", "domain": "evaluation"},
    {"question": "What is the RAG Triad?", "ground_truth": "The RAG Triad consists of groundedness, context relevance, and answer relevance.", "domain": "evaluation"},

    # Embeddings questions
    {"question": "What are the two architectures in Word2Vec?", "ground_truth": "CBOW predicts target words from context; Skip-gram predicts context from target.", "domain": "embeddings"},
    {"question": "What dimension does all-MiniLM-L6-v2 produce?", "ground_truth": "The all-MiniLM-L6-v2 model produces 384-dimensional embeddings.", "domain": "embeddings"},
    {"question": "How does cosine similarity differ from Euclidean distance for embeddings?", "ground_truth": "Cosine similarity computes the angle between vectors and is invariant to magnitude, while Euclidean distance measures geometric distance.", "domain": "embeddings"},

    # LangChain questions
    {"question": "What is RunnableParallel in LangChain?", "ground_truth": "RunnableParallel executes multiple chains simultaneously and combines their outputs.", "domain": "langchain"},
    {"question": "What does EnsembleRetriever use for combining retrievers?", "ground_truth": "EnsembleRetriever combines multiple retrievers using Reciprocal Rank Fusion for hybrid search.", "domain": "langchain"},
    {"question": "What is the difference between ConversationBufferMemory and ConversationSummaryMemory?", "ground_truth": "ConversationBufferMemory stores all messages while ConversationSummaryMemory stores a compressed summary.", "domain": "langchain"},
]