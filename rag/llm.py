from config import Config
from google import genai
import requests


# -----------------------------
# Gemini AI Client (Cloud API)
# -----------------------------
class GeminiAIClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = model

    def generate_answer(self, query: str, chunks: list, prompt_template: str = None) -> str:
        if not prompt_template:
            prompt_template = (
                "You are a helpful assistant. "
                "Use the following context to answer the query.\n\n"
                "Context:\n{context}\n\n"
                "Query: {query}\n\n"
                "Answer in a clear, concise way. "
                "If the answer is not in the context, say you don't know."
            )

        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

# -----------------------------
# AI Answer Service (Common Interface)
# -----------------------------
class AIAnswerService:
    def __init__(self, ai_client):
        self.ai_client = ai_client

    def get_answer(self, query: str, chunks: list) -> dict:
        try:
            answer = self.ai_client.generate_answer(query, chunks)
            return {"answer": answer}
        except Exception as e:
            return {"error": f"AI answer generation failed: {e}"}
