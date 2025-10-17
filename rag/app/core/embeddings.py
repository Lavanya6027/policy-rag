import logging
from langchain_huggingface import HuggingFaceEmbeddings
from app.custom_errors import ModelInitializationError 
from app.core.config import EMBEDDING_MODEL 

logger = logging.getLogger(__name__)

def load_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    Raises ModelInitializationError on failure.
    """
    try:
        logger.info(f"Attempting to load embedding model: {EMBEDDING_MODEL}")
        
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} 
        )
        logger.info(f"Successfully loaded embedding model: {EMBEDDING_MODEL}")
        return model
    except Exception as e:
        logger.error(f"FATAL: Failed to load embedding model {EMBEDDING_MODEL}: {e}", exc_info=True)
        raise ModelInitializationError(EMBEDDING_MODEL, message="LangChain HuggingFace model failed to load.") from e