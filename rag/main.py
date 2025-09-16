import os
import logging
from datetime import datetime
from extractors import ExtractorDispatcher, PDFExtractor, DocxExtractor, TxtExtractor
from preprocessing import Normalizer, Chunker
from embeddings import SBERTEmbedder, FAISSVectorStore
from rag import RAGPipeline
from llm import GeminiAIClient, AIAnswerService
from flask import Flask, request, jsonify
from config import Config

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_pipeline_{timestamp}.log")
    
    # Use handlers for clean CLI and file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    return logging.getLogger(__name__)

logger = setup_logging()

def clear_console():
    """Clear console based on OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

# Initialize RAG components
logger.info("Initializing RAG Pipeline")
try:
    ExtractorDispatcher.register(".pdf", PDFExtractor())
    ExtractorDispatcher.register(".docx", DocxExtractor())
    ExtractorDispatcher.register(".txt", TxtExtractor())
    logger.info("Document extractors registered.")

    normalizer = Normalizer()
    chunker = Chunker()
    embedder = SBERTEmbedder()
    store = FAISSVectorStore(embedding_dim=384)
    rag = RAGPipeline(embedder, store, chunker, normalizer)
    gemini = GeminiAIClient()
    ai_service = AIAnswerService(gemini)
    logger.info("RAG pipeline components initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize RAG components: {e}")
    raise

# Ingest documents
def ingest_documents(docs_folder, rag_pipeline):
    """Ingest documents from a specified folder into the RAG pipeline."""
    logger.info(f"Starting document ingestion from: {docs_folder}")
    if not os.path.exists(docs_folder):
        logger.critical(f"Documents folder not found: {docs_folder}")
        raise FileNotFoundError(f"Folder not found: {docs_folder}")

    doc_count, skipped_count = 0, 0
    for f in os.listdir(docs_folder):
        path = os.path.join(docs_folder, f)
        try:
            logger.info(f"Processing document: {f}")
            text = ExtractorDispatcher.extract(path)
            rag_pipeline.add_document(text, source=f)
            doc_count += 1
            logger.info(f"Successfully ingested {f}")
        except Exception as e:
            skipped_count += 1
            logger.warning(f"Skipped {f} due to error: {e}")
    
    logger.info(f"Document ingestion complete. Success: {doc_count}, Skipped: {skipped_count}.")

# Main execution
if __name__ == "__main__":
    clear_console()
    
    docs_folder = Config.DOCS_FOLDER
    ingest_documents(docs_folder, rag)

    # Flask app
    app = Flask(__name__)

    @app.route("/chat", methods=["POST"])
    def chat():
        """Handle chat queries via API."""
        try:
            data = request.json
            query = data.get("query", "")
            if not query.strip():
                return jsonify({"error": "Query is required"}), 400

            chunks = rag.retrieve(query, top_k=10)
            response = ai_service.get_answer(query, chunks)
            return jsonify({"answer": response["answer"]})
        except Exception as e:
            logger.error(f"Error during chat request: {e}")
            return jsonify({"error": "An internal error occurred"}), 500

    logger.info("Starting Flask application...")
    app.run(debug=True)