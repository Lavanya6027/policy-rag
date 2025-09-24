from typing import List, Tuple, Dict, Optional
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# -----------------------------
# Custom Exceptions
# -----------------------------
class RetrievalError(Exception):
    """Base class for errors in retrieval operations."""
    pass

class VectorSearchError(RetrievalError):
    """Raised when vector search fails."""
    pass

class KeywordSearchError(RetrievalError):
    """Raised when keyword search fails."""
    pass

class RerankError(RetrievalError):
    """Raised when reranking fails."""
    pass


# -------------------------
# 2. Reranker
# -------------------------
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
 
    def rerank(self, query, documents, top_n=5):
        pairs = [(query, d.page_content) for d in documents]
        scores = self.model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
        return sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)[:top_n]
 

# -----------------------------
# PolicyRetriever Class
# -----------------------------
class PolicyRetriever:
    """
    Hybrid retrieval system combining semantic (vector) search and 
    lexical (keyword) search with reranking.

    Attributes:
        vector_store (Chroma): Pre-initialized Chroma vector store.
        reranker (object): Reranker object with rerank(query, docs, top_n) method.
    """

    def __init__(self, vector_store: Chroma, reranker):
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve_context(
        self, 
        query: str, 
        all_docs: List[Document], 
        initial_k: int = 10, 
        final_k: int = 3
    ) -> Tuple[str, List[Dict]]:
        """
        Performs hybrid retrieval with reranking.

        Args:
            query (str): The user query to retrieve context for.
            all_docs (List[Document]): List of all available documents.
            initial_k (int, optional): Number of top documents to fetch per search method. Defaults to 10.
            final_k (int, optional): Number of top documents to return after reranking. Defaults to 3.

        Returns:
            Tuple[str, List[Dict]]: 
                - context: Combined string of top documents with metadata.
                - serialized_chunks: List of dicts with 'text' and 'source'.

        Raises:
            VectorSearchError: If vector search fails.
            KeywordSearchError: If keyword search fails.
            RerankError: If reranking fails.
        """
        try:
            # 1. Vector Search (Semantic)
            print("[INFO] Performing vector search...")
            vector_docs = self.vector_store.similarity_search(query, k=initial_k)
            print(f"[INFO] Vector search retrieved {len(vector_docs)} documents.")
        except Exception as e:
            raise VectorSearchError(f"Vector search failed: {e}")

        try:
            # 2. Keyword Search (Lexical)
            print("[INFO] Performing keyword search (BM25)...")
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = initial_k
            keyword_docs = bm25_retriever.invoke(query)
            print(f"[INFO] Keyword search retrieved {len(keyword_docs)} documents.")
        except Exception as e:
            raise KeywordSearchError(f"Keyword search failed: {e}")

        try:
            # 3. Combine and Deduplicate
            print("[INFO] Combining and deduplicating results...")
            combined_docs = vector_docs + keyword_docs
            unique_docs = list({doc.page_content: doc for doc in combined_docs}.values())
            print(f"[INFO] {len(unique_docs)} unique documents after deduplication.")

            # 4. Rerank
            print("[INFO] Reranking documents...")
            reranked_docs = self.reranker.rerank(query, unique_docs, top_n=final_k)
            print(f"[INFO] Returning top {len(reranked_docs)} reranked documents.")
        except Exception as e:
            raise RerankError(f"Reranking failed: {e}")

        # 5. Serialize results
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source_name', 'N/A')}\nContent: {doc.page_content}"
            for doc in reranked_docs
        )

        serialized_chunks = [
            {"text": doc.page_content, "source": doc.metadata.get("source_name", "N/A")}
            for doc in reranked_docs
        ]

        return context, serialized_chunks

# -----------------------------
# Function to initialize PolicyRetriever
# -----------------------------
def init_policy_retriever(
    vector_store,
    reranker: Optional[Reranker] = None
) -> PolicyRetriever:
    """
    Initialize the PolicyRetriever with optional custom reranker.

    Args:
        vector_store_path (str): Path to Chroma vector store.
        embedding_function: Optional embedding function for Chroma.
        reranker (Reranker, optional): Custom reranker. Defaults to None.

    Returns:
        PolicyRetriever: Initialized retriever object.
    """
    if reranker is None:
        print("[INFO] No custom reranker provided. Using default Reranker...")
        reranker = Reranker()
    else:
        print("[INFO] Using custom reranker provided.")

    retriever = PolicyRetriever(vector_store=vector_store, reranker=reranker)
    print("[INFO] PolicyRetriever initialized successfully.")
    return retriever
