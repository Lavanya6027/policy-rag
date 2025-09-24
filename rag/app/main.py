from flask import Flask, request, jsonify
from pathlib import Path
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

        answer = generate_answer(question)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
