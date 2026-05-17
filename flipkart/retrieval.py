"""
Retrieval Strategies for the RAG Pipeline

Provides multiple retriever configurations:
1. Vector-only (baseline) — Pure semantic search via AstraDB
2. BM25-only — Keyword matching on local documents
3. Hybrid — Combines vector + BM25 via EnsembleRetriever

Why hybrid search?
- Vector search finds semantically similar text but misses exact keywords
- BM25 finds exact keyword matches but misses semantic meaning
- Combining both catches queries like "price of BoAt" (keyword) AND
  "affordable wireless earbuds" (semantic)

Usage:
    from flipkart.retrieval import get_retriever
    
    retriever = get_retriever(vector_store, strategy="hybrid")
"""
import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # type: ignore
from flipkart.data_preprocessing import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

DATA_PATH = "data/flipkart_product_review.csv"
PRICE_PATH = "data/product_prices.csv"


def _load_documents_for_bm25() -> list:
    """Load documents from CSV for BM25 retriever.
    
    BM25 needs local documents (it doesn't use a vector store).
    We load the same data that's in AstraDB so both retrievers
    search the same content.
    """
    df = pd.read_csv(DATA_PATH)
    
    # Load prices
    try:
        prices_df = pd.read_csv(PRICE_PATH)
        price_map = dict(zip(prices_df["product_title"], prices_df["price"]))
    except FileNotFoundError:
        price_map = {}
    
    documents = []
    for _, row in df.iterrows():
        title = str(row.get("product_title", ""))
        rating = row.get("rating", "N/A")
        summary = str(row.get("summary", ""))
        review = clean_text(str(row.get("review", "")))
        price = price_map.get(title, "N/A")
        
        content = (
            f"Product: {title}\n"
            f"Price: ₹{price}\n"
            f"Rating: {rating}/5\n"
            f"Summary: {summary}\n"
            f"Review: {review}"
        )
        
        doc = Document(
            page_content=content,
            metadata={"product_title": title, "rating": rating, "price": price}
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents for BM25")
    return documents


def get_retriever(vector_store, strategy: str = "hybrid", k: int = 3):
    """Get a retriever based on the specified strategy.
    
    Args:
        vector_store: AstraDB vector store instance.
        strategy: One of 'vector', 'bm25', or 'hybrid'.
        k: Number of documents to retrieve.
    
    Returns:
        A LangChain retriever instance.
    """
    if strategy == "vector":
        logger.info(f"Using vector-only retriever (k={k})")
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    elif strategy == "bm25":
        logger.info(f"Using BM25-only retriever (k={k})")
        docs = _load_documents_for_bm25()
        return BM25Retriever.from_documents(docs, k=k)
    
    elif strategy == "hybrid":
        logger.info(f"Using hybrid retriever (vector + BM25, k={k})")
        
        # Vector retriever (semantic search)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # BM25 retriever (keyword search)
        docs = _load_documents_for_bm25()
        bm25_retriever = BM25Retriever.from_documents(docs, k=k)
        
        # Combine: 60% vector + 40% BM25
        # Vector gets more weight because semantic understanding is more
        # important than keyword matching for most queries
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
        return ensemble
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'vector', 'bm25', or 'hybrid'.")
