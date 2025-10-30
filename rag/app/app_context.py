import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Annotated
from fastapi import Depends, status, Request
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings 
from langchain_core.retrievers import BaseRetriever
from langchain.storage import LocalFileStore 

from app.db.manager import JsonFileManager 
from app.db.repository import JsonRepository
from app.data_models import ResourceModel 
from app.core.embeddings import load_embedding_model 
from app.core.llm import AIService 
from app.retrievers.vector_retriever import VectorRetrieverLoader 
from app.data_models import QueryRequest, RAGResponse 
from app.core.rag_chain_service import process_rag_query 
from app.core.chat_logger import ChatLogger

logger = logging.getLogger(__name__)

DB_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db', 'dev_db.json')

RAG_CONTENT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'corpora')
RAG_VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vector_store', 'chroma_db') # Adjusted path name for standard vector store
RAG_COLLECTION_NAME = "vector_collection"
RAG_CLEANUP_EXISTING = False 

STATE: Dict[str, Any] = {}

def initialize_rag_retriever(embedding_model: Embeddings, cleanup_existing: bool = RAG_CLEANUP_EXISTING) -> BaseRetriever:
    """Instantiates the VectorRetrieverLoader and calls its load_retriever."""
    print("Initializing RAG Retriever (Standard Vector Retriever)...")
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    try:
        retriever_loader = VectorRetrieverLoader(
            content_path=RAG_CONTENT_PATH,
            embedding_model=embedding_model,
            collection_name=RAG_COLLECTION_NAME,
            vector_store_path=RAG_VECTOR_STORE_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            cleanup_existing=cleanup_existing 
        )
        
        rag_retriever = retriever_loader.load_retriever()
        
        print("RAG Retriever successfully initialized.")
        return rag_retriever
    except Exception as e:
        logger.critical(f"FATAL ERROR during RAG Retriever initialization: {e}")
        raise RuntimeError("Application failed to initialize RAG components.") from e

def update_rag_retriever(new_retriever: BaseRetriever):
    """Updates the global STATE with a newly built RAG retriever instance."""
    STATE['RAG_RETRIEVER'] = new_retriever
    logger.info("RAG Retriever instance in STATE successfully updated.")

@asynccontextmanager
async def lifespan_startup_shutdown(app: Any):
    """Initializes DAL, the embedding model, and the RAG retriever on startup."""
    print("Executing application startup sequence...")

    try:
        db_manager = JsonFileManager(DB_FILE_PATH)
        resource_repo = JsonRepository(db_manager, 'resources', ResourceModel)
        embedding_model = load_embedding_model() 
        ai_service = AIService() 
        rag_retriever = initialize_rag_retriever(embedding_model=embedding_model, cleanup_existing=RAG_CLEANUP_EXISTING)

        chat_logger = ChatLogger()
        app.state.chat_logger = chat_logger

        STATE['DB_MANAGER'] = db_manager
        STATE['RESOURCE_REPO'] = resource_repo
        STATE['EMBEDDING_MODEL'] = embedding_model 
        STATE['RAG_RETRIEVER'] = rag_retriever 
        STATE['AI_SERVICE'] = ai_service
        
        print("All necessary resources and RAG system successfully initialized.")
    except Exception:
        STATE.clear()
        raise
    
    yield 
    
    print("Executing application shutdown sequence...")
    STATE.clear() 


def get_resource_repository() -> JsonRepository:
    """Provides the initialized ResourceRepository instance."""
    if 'RESOURCE_REPO' not in STATE:
        raise RuntimeError("Resource repository not initialized. Check FastAPI lifespan setup.")
    return STATE['RESOURCE_REPO']

def get_embedding_model() -> Embeddings:
    """Provides the initialized LangChain Embeddings model instance."""
    if 'EMBEDDING_MODEL' not in STATE:
        raise RuntimeError("Embedding model not initialized. Check FastAPI lifespan setup.")
    return STATE['EMBEDDING_MODEL']

def get_rag_retriever() -> BaseRetriever:
    """Provides the initialized RAG BaseRetriever instance."""
    if 'RAG_RETRIEVER' not in STATE:
        raise RuntimeError("RAG Retriever not initialized. Check FastAPI lifespan setup.")
    return STATE['RAG_RETRIEVER']

def get_ai_service() -> AIService:
    """Provides the initialized AI_SERVICE instance."""
    if 'AI_SERVICE' not in STATE:
        raise RuntimeError("AI_SERVICE not initialized. Check FastAPI lifespan setup.")
    return STATE['AI_SERVICE']

def get_chat_logger(request: Request):
    return request.app.state.chat_logger

EmbeddingModelDep = Annotated[Embeddings, Depends(get_embedding_model)]
RetrieverDep = Annotated[BaseRetriever, Depends(get_rag_retriever)]
AIServiceDep = Annotated[AIService, Depends(get_ai_service)]
