from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (rag/)

class Config:
    # AI Model
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

    # Vectorstore & embeddings
    VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", BASE_DIR / "chroma_store"))
    JSON_PATH = Path(os.getenv("JSON_PATH", BASE_DIR / "data/processed_chunks_with_ids.json"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Retriever settings
    TOP_K_CHUNKS_ON_RETRIEVER = int(os.getenv("TOP_K_CHUNKS_ON_RETRIEVER", 5))
    TOP_K_CHUNKS_ON_RERANKER = int(os.getenv("TOP_K_CHUNKS_ON_RERANKER", 3))

    @classmethod
    def validate(cls):
        if not cls.OLLAMA_MODEL:
            raise ValueError("OLLAMA_MODEL is not set in environment")
        if not cls.OLLAMA_API_URL:
            raise ValueError("OLLAMA_API_URL is not set in environment")
        if not cls.JSON_PATH.exists():
            raise FileNotFoundError(f"JSON_PATH does not exist: {cls.JSON_PATH}")
