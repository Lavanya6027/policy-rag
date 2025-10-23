import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "127.0.0.1:11435")
OLLAMA_TIMEOUT = os.getenv("OLLAMA_TIMEOUT", 120)