from config import Config
from google import genai
import requests
import json
import time


# -----------------------------
# Gemini AI Client (Cloud API)
# -----------------------------
class GeminiAIClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = model

    def generate_answer(self, query: str, chunks: list, prompt_template: str = None) -> dict:
        if not prompt_template:
            prompt_template = (
                "You are a helpful assistant for company travel policy questions. "
                "You will receive a context document and a query.\n\n"
                "Your task:\n"
                "1. If the query asks for factual classification (like employee category, city class, travel type, conduct rules), "
                "answer directly from the context.\n"
                "2. If the query requires reasoning (like calculating allowance, comparing entitlements, eligibility checks), "
                "work through the logic step by step and then provide the final answer.\n"
                "3. Always base answers only on the given context. "
                "If the context doesn’t cover the query, reply with: 'I don’t know based on the provided policy.'\n\n"
                "Guidelines:\n"
                "- Show calculations clearly if math is involved.\n"
                "- Be concise but complete.\n"
                "- Never invent rules that are not in the context.\n\n"
                "Context:\n{context}\n\n"
                "Query: {query}\n\n"
                "Final Answer:"
            )


        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)

        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            elapsed = round(time.time() - start_time, 3)

            answer = getattr(response, "text", str(response))
            return {
                "answer": answer,
                "tokens": len(prompt.split()),
                "response_time": elapsed,
                "error": None
            }
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return {
                "answer": None,
                "tokens": 0,
                "response_time": elapsed,
                "error": f"Gemini error: {e}"
            }


# -----------------------------
# Ollama AI Client (Local/Server API)
# -----------------------------
class OllamaAIClient:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://10.16.7.20:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_answer(self, query: str, chunks: list, prompt_template: str = None) -> dict:
        if not prompt_template:
            prompt_template = (
                "You are a helpful assistant for company travel policy questions. "
                "You will receive a context document and a query.\n\n"
                "Your task:\n"
                "1. If the query asks for factual classification (like employee category, city class, travel type, conduct rules), "
                "answer directly from the context.\n"
                "2. If the query requires reasoning (like calculating allowance, comparing entitlements, eligibility checks), "
                "work through the logic step by step and then provide the final answer.\n"
                "3. Always base answers only on the given context. "
                "If the context doesn’t cover the query, reply with: 'I don’t know based on the provided policy.'\n\n"
                "Guidelines:\n"
                "- Show calculations clearly if math is involved.\n"
                "- Be concise but complete.\n"
                "- Never invent rules that are not in the context.\n\n"
                "Context:\n{context}\n\n"
                "Query: {query}\n\n"
                "Final Answer:"
            )


        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)

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
            answer = "".join(full_response).strip() if full_response else "[No response]"
            return {
                "answer": answer,
                "tokens": len(prompt.split()),
                "response_time": elapsed,
                "error": None
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return {
                "answer": None,
                "tokens": 0,
                "response_time": elapsed,
                "error": f"Ollama error: {e}"
            }

    def test_generate_answer(self, prompt: str) -> dict:

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
            answer = "".join(full_response).strip() if full_response else "[No response]"
            return {
                "answer": answer,
                "tokens": len(prompt.split()),
                "response_time": elapsed,
                "error": None
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            return {
                "answer": None,
                "tokens": 0,
                "response_time": elapsed,
                "error": f"Ollama error: {e}"
            }

# -----------------------------
# AI Answer Service (Common Interface)
# -----------------------------
class AIAnswerService:
    def __init__(self, ai_client):
        self.ai_client = ai_client

    def get_answer(self, query: str, chunks: list) -> dict:
        return self.ai_client.generate_answer(query, chunks)
    
    def test_get_answer(self, prompt: str) -> dict:
        if hasattr(self.ai_client, 'test_generate_answer'):
            return self.ai_client.test_generate_answer(prompt)
        else:
            return {"answer": None, "tokens": 0, "response_time": 0, "error": "Test method not implemented for this client."}
