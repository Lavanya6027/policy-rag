import logging
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.document_loader import DocumentProcessor 
from app.core.vector_store_manager import VectorStoreManager 

logger = logging.getLogger(__name__)

class ParentDocumentRetrieverLoader:
    """
    Implements the Parent Document Retriever (PDR) strategy.
    Requires a persistent doc_store (e.g., LocalFileStore) for production use 
    to prevent context loss on restart.
    """
    def __init__(
        self,
        content_path: str,
        embedding_model: Embeddings,
        doc_store: BaseStore,
        collection_name: str = "pdr_master",
        vector_store_path: str = "vector_store/pdr_chroma_db",
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        chunk_overlap: int = 200,
        cleanup_existing: bool = False
    ):
        if not isinstance(doc_store, BaseStore):
             raise TypeError("doc_store must be an instance of LangChain BaseStore.")
             
        self.content_path = content_path
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.cleanup_existing = cleanup_existing
        
        self.doc_processor = DocumentProcessor(content_path=self.content_path)
        self.store_manager = VectorStoreManager(
            embedding_model=embedding_model,
            collection_name=collection_name,
            vector_store_path=vector_store_path
        )
        # FIX: doc_store is expected to be persistent (LocalFileStore) and passed in.
        self.doc_store = doc_store

    def load_retriever(self) -> BaseRetriever:
        """Builds and returns the ParentDocumentRetriever, loading from persistence if possible."""
        try:
            # 1. Instantiate Splitters
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.parent_chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.child_chunk_size, 
                chunk_overlap=self.chunk_overlap
            )

            # 2. Get the Vector Store (handles cleanup/load logic internally)
            vector_store = self.store_manager.load_or_create_vector_store(force_rebuild=self.cleanup_existing)
            
            # 3. Instantiate the ParentDocumentRetriever
            pdr_retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=self.doc_store, 
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )

            # 4. Check if a full rebuild is necessary
            rebuild_required = self.cleanup_existing or vector_store._collection.count() == 0

            if rebuild_required:
                logger.info("Starting fresh PDR indexing process.")
                raw_documents = self.doc_processor.load_raw_documents()
                
                if raw_documents:
                    # This method populates both the persistent vector store (child chunks) 
                    # and the persistent doc store (parent documents).
                    pdr_retriever.add_documents(raw_documents)
                    logger.info("Parent Document Retriever index built successfully.")
                else:
                    logger.warning("No documents loaded to index. Retriever remains empty.")
            else:
                logger.info("Loaded existing ParentDocumentRetriever components successfully.")

            return pdr_retriever
            
        except Exception as e:
            logger.error(f"Error during ParentDocumentRetriever loading/building: {e}")
            raise RuntimeError("Failed to build ParentDocumentRetriever.") from e