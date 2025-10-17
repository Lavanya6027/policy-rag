class VectorStoreError(Exception):
    """Custom exception raised for critical vector store management errors."""
    pass

# vector_store_manager.py

import os
import shutil
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
# Removed: from chromadb.errors import InvalidCollection (Caused the ImportError)
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# Define the custom error class here, before VectorStoreManager
class VectorStoreError(Exception):
    """Custom exception raised for critical vector store management errors."""
    pass

class VectorStoreManager:
    """Manages the creation, loading, and persistence of the ChromaDB vector store."""

    def __init__(
        self,
        embedding_model: Embeddings,
        collection_name: str,
        vector_store_path: str,
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_store_path = vector_store_path
        self.vector_store: Optional[Chroma] = None

    def _cleanup_directory(self) -> None:
        """Removes the entire vector store directory (used only as a fallback for file lock issues)."""
        if os.path.exists(self.vector_store_path):
            logger.warning(f"Attempting to remove entire ChromaDB directory at: {self.vector_store_path} as fallback.")
            try:
                # Explicitly delete the LangChain object to help release file handles before rmtree
                if self.vector_store:
                    del self.vector_store
                    self.vector_store = None
                
                shutil.rmtree(self.vector_store_path)
                logger.info("ChromaDB directory cleaned successfully via rmtree.")
            except OSError as e:
                # Raise the custom error for critical failure
                error_msg = (
                    f"CRITICAL ERROR: Failed to remove ChromaDB directory {self.vector_store_path}: {e}. "
                    "The index cannot be rebuilt cleanly. A server restart may be required."
                )
                logger.error(error_msg)
                raise VectorStoreError(error_msg) from e

    def _clear_existing_collection(self) -> None:
            """Clears all documents from the existing ChromaDB collection using the database's internal method."""
            if self.vector_store:
                logger.info(f"Clearing all documents from collection: '{self.collection_name}' using internal delete.")
                
                try:
                    # FIX: Access the underlying Chroma client using ._client attribute
                    client = self.vector_store._client
                    
                    # Check if the collection exists before attempting to delete it
                    try:
                        client.get_collection(name=self.collection_name)
                        client.delete_collection(name=self.collection_name)
                        logger.info("ChromaDB collection cleared and deleted successfully.")
                    except Exception as e:
                        # Catch the error where the collection doesn't exist (e.g., if database file is empty)
                        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                            logger.warning(f"Collection '{self.collection_name}' not found. No clear needed.")
                        else:
                            raise e # Re-raise other errors
                    
                    # The LangChain object is now invalid, so set it to None.
                    self.vector_store = None 
                    
                except Exception as e:
                    logger.error(
                        f"Error clearing ChromaDB collection internally: {e}. "
                        "Falling back to full directory removal (rmtree)."
                    )
                    self._cleanup_directory() # This will raise VectorStoreError if it fails again
            elif os.path.exists(self.vector_store_path):
                logger.warning("VectorStore object is None but path exists. Forcing directory cleanup as a fallback.")
                self._cleanup_directory()


    def load_or_create_vector_store(
        self, 
        documents_to_index: List[Document] = None, 
        force_rebuild: bool = False
    ) -> Chroma:
        """Loads existing ChromaDB or creates a new one from documents."""
        
        # --- Handle Force Rebuild ---
        if force_rebuild:
            # Attempt to load the existing store first if it's not set, so _clear_existing_collection can work
            if not self.vector_store and os.path.exists(self.vector_store_path):
                try:
                    self.vector_store = Chroma(
                        collection_name=self.collection_name, 
                        embedding_function=self.embedding_model, 
                        persist_directory=self.vector_store_path
                    )
                except Exception as e:
                    logger.error(f"Failed to instantiate temporary Chroma object for clearing: {e}")
                    # Continue, _clear_existing_collection handles the None case
            
            # The _clear_existing_collection will now raise VectorStoreError on critical failure.
            self._clear_existing_collection()
        
        # --- Attempt to Load Existing ---
        if not force_rebuild and os.path.exists(self.vector_store_path):
            try:
                logger.info("Attempting to load existing ChromaDB index from disk.")
                vector_store = Chroma(
                    collection_name=self.collection_name, 
                    embedding_function=self.embedding_model, 
                    persist_directory=self.vector_store_path
                )
                # Check if the collection is actually populated
                collection: Collection = vector_store.get()
                if collection["ids"]: # Checks if the collection is non-empty
                    self.vector_store = vector_store
                    logger.info(f"ChromaDB loaded from existing index with {len(collection['ids'])} items.")
                    return self.vector_store
                
                logger.warning("Loaded ChromaDB is empty or corrupted. Forcing index rebuild.")
                force_rebuild = True # Fall through to rebuild logic
                self.vector_store = None # Ensure object is cleared before rebuild
                
            except Exception as e:
                logger.error(f"Failed to load existing ChromaDB index: {e}. Forcing index rebuild.")
                force_rebuild = True # Fall through to rebuild logic
                self.vector_store = None # Ensure object is cleared before rebuild

        # --- Create New Index (Rebuild) ---
        if documents_to_index and len(documents_to_index) > 0:
            logger.info(f"Creating new ChromaDB index with {len(documents_to_index)} chunks.")
            
            # If force_rebuild was true, the old collection was cleared. 
            # Now, create a new collection with the new documents.
            self.vector_store = Chroma.from_documents(
                documents=documents_to_index, 
                embedding=self.embedding_model, 
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            logger.info(f"New ChromaDB index saved to disk at: {self.vector_store_path}")
        else:
            # Create empty store if no documents are provided for indexing
            if not self.vector_store: 
                logger.warning("Creating an empty collection for initialization.")
                self.vector_store = Chroma(
                    collection_name=self.collection_name, 
                    embedding_function=self.embedding_model, 
                    persist_directory=self.vector_store_path
                )

        return self.vector_store