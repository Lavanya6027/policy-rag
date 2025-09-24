import time
import json
import requests
import logging
from app.config import Config
from google import genai

# -----------------------------
# Setup Logging (Console Only)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("AIService")

# -----------------------------
# Ollama AI Client (Local API)
# -----------------------------
class OllamaAIClient:
    def __init__(self, model: str = Config.OLLAMA_MODEL, base_url: str = Config.OLLAMA_API_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _send_request(self, prompt: str) -> tuple[str, float]:
        """Send chat request to Ollama and stream response back."""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_response = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "message" in data and "content" in data["message"]:
                            full_response.append(data["message"]["content"])
                        elif "response" in data:
                            full_response.append(data["response"])
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            elapsed = round(time.time() - start_time, 3)
            return "".join(full_response).strip(), elapsed
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            raise RuntimeError(f"Ollama error: {e}") from e

    def generate_answer(self, query: str, chunks: list = None, prompt_template: str = None) -> dict:
        """Generate answer using Ollama AI with optional custom prompt."""
        if chunks is None:
            chunks = []

        if not prompt_template:
            prompt_template = (
                "You are a helpful assistant for company travel policy questions. "
                "Context:\n{context}\n\nQuery: {query}\n\nFinal Answer:"
            )

        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)

        try:
            answer, elapsed = self._send_request(prompt)
            logger.info(f"[Ollama] Model={self.model} | Time={elapsed}s | Tokens={len(prompt.split())}")
            return {
                "answer": answer or "[No response]",
                "tokens": len(prompt.split()),
                "response_time": elapsed,
                "error": None
            }
        except Exception as e:
            logger.error(e)
            return {
                "answer": None,
                "tokens": 0,
                "response_time": 0,
                "error": str(e)
            }


# -----------------------------
# AI Answer Service (Common Interface)
# -----------------------------
class AIAnswerService:
    def __init__(self, ai_client):
        self.ai_client = ai_client

    def get_answer(self, query: str, chunks: list = None, prompt_template: str = None) -> dict:
        """Get answer from AI client with optional custom prompt."""
        return self.ai_client.generate_answer(query=query, chunks=chunks, prompt_template=prompt_template)
