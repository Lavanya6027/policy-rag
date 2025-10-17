from fastapi import FastAPI, Depends, status
from fastapi.responses import JSONResponse
import logging
from app.routers import resources 
from app.app_context import (
    lifespan_startup_shutdown, 
    initialize_rag_retriever, 
    update_rag_retriever,     
    EmbeddingModelDep,
    RetrieverDep,
    AIServiceDep         
)
from app.data_models import QueryRequest, RAGResponse 
from app.core.rag_chain_service import process_rag_query 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HR Assistant Lia",
    version="1.0.0",
    lifespan=lifespan_startup_shutdown
)

app.include_router(resources.router)

@app.get("/", tags=["Root"])
def read_root():
    """Simple root endpoint to confirm API is running."""
    return {"message": "HR Assistant Lia API is running. Check /docs for endpoints."}

@app.get("/refresh-corpora", tags=["Admin"])
def refresh_corpora(embedding_model: EmbeddingModelDep):
    """
    Forces a full rebuild and re-indexing of the RAG corpora and updates 
    the active RAG retriever instance in the application state.
    """
    logger.info("Manual RAG corpora refresh initiated.")
    try:
        new_retriever = initialize_rag_retriever(
            embedding_model=embedding_model, 
            cleanup_existing=True
        )
        
        update_rag_retriever(new_retriever)
        
        return {
            "status": "success", 
            "message": "Corpora successfully re-indexed and RAG retriever updated."
        }
    except Exception as e:
        logger.error(f"Failed to refresh corpora: {e}")
        # Updated to use JSONResponse for proper API error handling
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error", 
                "message": f"Corpora refresh failed: {str(e)}"
            }
        )

@app.post(
    "/chat/", 
    response_model=RAGResponse, 
    tags=["RAG"], 
    summary="Chat with the HR Assistant/Query the policy knowledge base."
)
async def chat_with_assistant(
    request: QueryRequest,
    retriever: RetrieverDep,
    ai_service: AIServiceDep
):
    """
    Executes a RAG query/chat turn:
    1. Embeds the user question.
    2. Retrieves relevant policy documents from the vector store using the RAG retriever.
    3. Engineers a prompt with the context and question.
    4. Sends the final prompt to the LLM (Ollama/Gemini) for a grounded answer.
    """
    response = await process_rag_query(request, retriever, ai_service)
    
    if response.error:
        logger.error(f"Chat endpoint returned an error: {response.error}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.model_dump()
        )
    
    return response
