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
            """
            Loads or creates the index and returns the retriever.
            
            Note: This is modified to always force a rebuild/reindexing,
            clearing any existing data.
            """
            
            # 1. Force a rebuild/reindex right away by passing force_rebuild=True
            # This clears any existing index based on the store_manager's implementation.
            logger.info("Starting index rebuild for VectorRetrieverLoader (forced).")
            
            # Calling load_or_create_vector_store with force_rebuild=True here
            # ensures the old data is cleared before we attempt to index new data.
            # This is a good practice to ensure the vector store is properly reset.
            _ = self.store_manager.load_or_create_vector_store(force_rebuild=True) 

            # 2. Load and process documents
            raw_documents = self.doc_processor.load_raw_documents()
            
            if not raw_documents:
                logger.warning("No documents loaded to index. Creating empty vector store.")
                # Still call load_or_create_vector_store with force_rebuild=True 
                # to ensure an empty, fresh store is returned if no docs are found.
                vector_store = self.store_manager.load_or_create_vector_store(force_rebuild=True)
                return vector_store.as_retriever()

            chunks = self.doc_processor.chunk_documents(
                documents=raw_documents, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"chunks: {chunks[0]}")
            
            # 3. Create the new index and get the retriever by indexing the chunks
            # force_rebuild=True is already ensured here, but it's good practice
            # to pass it again if load_or_create_vector_store handles the actual 
            # indexing logic and might implicitly clear/rebuild.
            vector_store = self.store_manager.load_or_create_vector_store(
                documents_to_index=chunks, 
                force_rebuild=True
            )
            logger.info("Retriever successfully initialized with new index.")
            return vector_store.as_retriever()
