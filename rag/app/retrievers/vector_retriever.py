import logging
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
# Assuming document_loader and vector_store_manager are available via relative or absolute imports
from app.core.document_loader import DocumentProcessor 
from app.core.vector_store_manager import VectorStoreManager 

logger = logging.getLogger(__name__)

class VectorRetrieverLoader:
    """
    Standard RAG retrieval: loads documents, chunks them, indexes in ChromaDB, 
    and exposes a simple similarity search retriever.
    """
    def __init__(
        self,
        content_path: str,
        embedding_model: Embeddings,
        collection_name: str = "master",
        vector_store_path: str = "vector_store/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cleanup_existing: bool = True
    ):
        self.content_path = content_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cleanup_existing = cleanup_existing
        
        self.doc_processor = DocumentProcessor(content_path=self.content_path)
        self.store_manager = VectorStoreManager(
            embedding_model=embedding_model,
            collection_name=collection_name,
            vector_store_path=vector_store_path
        )

    def load_retriever(self) -> BaseRetriever:
        """Loads or creates the index and returns the retriever."""
        
        # 1. Attempt to load existing index
        vector_store = self.store_manager.load_or_create_vector_store(force_rebuild=False)

        # Check if the store was successfully loaded and contains data
        # Note: Accessing private attribute `_collection` is a workaround for ChromaDB count
        if vector_store and hasattr(vector_store, '_collection') and vector_store._collection.count() > 0 and not self.cleanup_existing:
            logger.info("Existing index loaded successfully.")
            return vector_store.as_retriever()
        
        # 2. Rebuild is required (or forced)
        logger.info("Starting index rebuild for VectorRetrieverLoader.")
        raw_documents = self.doc_processor.load_raw_documents()
        
        if not raw_documents:
            logger.warning("No documents loaded to index. Creating empty vector store.")
            vector_store = self.store_manager.load_or_create_vector_store(force_rebuild=True)
            return vector_store.as_retriever()

        chunks = self.doc_processor.chunk_documents(
            documents=raw_documents, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # 3. Create the new index and get the retriever
        vector_store = self.store_manager.load_or_create_vector_store(
            documents_to_index=chunks, 
            force_rebuild=True
        )
        logger.info("Retriever successfully initialized.")
        return vector_store.as_retriever()
