from flask import Flask, request, jsonify
from pathlib import Path
import json
from datetime import datetime
from typing import List, Optional
from app.document_loaders import JSONChunkLoader
from app.retriever import PolicyRetriever, init_policy_retriever
from app.llm import AIAnswerService, OllamaAIClient
from app.policy_corpus_builder import build_vectorstore_from_json
from app.config import Config

app = Flask(__name__)

# -----------------------------
# Initialize RAG components
# -----------------------------
VECTORSTORE_PATH = Config.VECTORSTORE_PATH
JSON_PATH = Config.JSON_PATH
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
TOP_K_CHUNKS_ON_RETRIEVER = Config.TOP_K_CHUNKS_ON_RETRIEVER
TOP_K_CHUNKS_ON_RERANKER = Config.TOP_K_CHUNKS_ON_RERANKER

print("[INFO] Initializing vector store...")
vectorstore = build_vectorstore_from_json(
    json_path=JSON_PATH,
    persist_dir=VECTORSTORE_PATH,
    embedding_model=EMBEDDING_MODEL
)
print("[INFO] Vector store ready.")

print("[INFO] Initializing retriever...")
hybrid_retriever = init_policy_retriever(vector_store=vectorstore, reranker=None)
print("[INFO] Retriever ready.")

doc_loader = JSONChunkLoader(json_path=JSON_PATH)
docs = doc_loader.load()

ollama_client = OllamaAIClient()
ai_service = AIAnswerService(ollama_client)

BACKUP_LOG_FILE = Path("rag_query_logs.json")

def log_interaction(query: str, chunk_ids: list, response: str):
    # Create a log entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "chunk_ids": chunk_ids,
        "response": response
    }

    # If file exists, load it, otherwise start with empty list
    if BACKUP_LOG_FILE.exists():
        with open(BACKUP_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Append new entry
    logs.append(entry)

    # Save back
    with open(BACKUP_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# -----------------------------
# Helper: generate answer
# -----------------------------
def generate_answer(question: str):
    context, chunks = hybrid_retriever.retrieve_context(
        query=question,
        all_docs=docs,
        initial_k=TOP_K_CHUNKS_ON_RERANKER,
        final_k=TOP_K_CHUNKS_ON_RETRIEVER
    )
    print(f"[DEBUG] Retrieved context: {context}")

    query_template = f"""
You are an AI assistant specialized in answering company policy and HR-related queries.
- Always use the provided context to answer.
- If the answer requires reasoning (e.g., calculating allowance), explain your steps clearly.
- If context is insufficient, say "I could not find this information in the documents."
- Be concise but cover key details.
- Provide references to source documents when possible.

Context:
{context}

Question:
{question}
"""
    response = ai_service.get_answer(query=query_template)

    # backup log
    chunk_ids = [chunk["id"] for chunk in chunks]
    log_interaction(query=question, chunk_ids=chunk_ids, response=response)
    return response

# -----------------------------
# Flask API endpoint
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("query", "").strip()
        if not question:
            return jsonify({"error": "Query is required"}), 400

        response_data = generate_answer(question)
        return jsonify({
            "answer": response_data["answer"],
            "tokens": response_data["tokens"],
            "response_time": response_data["response_time"],
            "error": response_data["error"]
        })


    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
