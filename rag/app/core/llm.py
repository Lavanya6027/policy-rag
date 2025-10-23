import time
import json
import requests
import logging
from typing import Protocol, Any, Dict, Literal
# from google import genai 
# from google.genai.errors import APIError
from app.core.config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

logger = logging.getLogger("AIService")

class BaseAIService(Protocol):
    """
    Protocol defining the common interface for all AI generation services.
    Services must only handle LLM communication, not prompt engineering.
    """
    def query(self, prompt: str) -> Dict[str, Any]:
        """
        Sends the final, fully engineered prompt to the LLM and returns the structured response.

        :param prompt: The complete text prompt to send to the model.
        :return: A dictionary containing the answer and metadata.
        """
        ...


class OllamaAIService:
    """Handles direct communication with a self-hosted Ollama endpoint."""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info(f"Ollama Client initialized (Model: {self.model}, URL: {self.base_url})")

    def _send_request(self, prompt: str) -> tuple[str, float]:
        """Internal logic to send request to Ollama and stream the response."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True 
                },
                stream=True,
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()

            full_response = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            full_response.append(data["response"])
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            elapsed = round(time.time() - start_time, 3)
            return "".join(full_response).strip(), elapsed
        except Exception as e:
            raise RuntimeError(f"Ollama connectivity or request error: {e}") from e


    def query(self, prompt: str) -> Dict[str, Any]:
        """Sends the prompt to Ollama and returns the answer and metadata."""
        try:
            answer, elapsed = self._send_request(prompt)
            tokens = len(prompt.split())
            
            logger.info(f"[Ollama] Model={self.model} | Time={elapsed}s | Tokens={tokens}")
            
            return {
                "answer": answer or "[No response]",
                "tokens": tokens,
                "response_time": elapsed,
                "error": None
            }
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return {
                "answer": None,
                "tokens": 0,
                "response_time": 0,
                "error": str(e)
            }

# class GeminiAIService:
#     """Handles direct communication with the Google Gemini API."""
    
#     def __init__(self, model: str = Config.GEMINI_MODEL, api_key: str = Config.GEMINI_API_KEY):
#         self.model = model
        
#         if not api_key:
#             logger.error("GEMINI_API_KEY is not configured.")
#             raise ValueError("GEMINI_API_KEY is required for GeminiAIService.")
            
#         self.client = genai.Client(api_key=api_key)
#         logger.info(f"Gemini Client initialized (Model: {self.model})")

#     def query(self, prompt: str) -> Dict[str, Any]:
#         """Sends the prompt to Gemini and returns the answer and metadata."""
#         start_time = time.time()
        
#         try:
#             response = self.client.models.generate_content(
#                 model=self.model,
#                 contents=[prompt]
#             )
            
#             elapsed = round(time.time() - start_time, 3)
#             answer = response.text.strip()
#             tokens = len(prompt.split()) 
#             logger.info(f"[Gemini] Model={self.model} | Time={elapsed}s | Tokens={tokens}")
            
#             return {
#                 "answer": answer,
#                 "tokens": tokens,
#                 "response_time": elapsed,
#                 "error": None
#             }

#         except APIError as e:
#             logger.error(f"Gemini API Error: {e}")
#             return {
#                 "answer": None,
#                 "tokens": 0,
#                 "response_time": 0,
#                 "error": f"Gemini API Error: {str(e)}"
#             }
#         except Exception as e:
#             logger.error(f"Gemini unknown error: {e}")
#             return {
#                 "answer": None,
#                 "tokens": 0,
#                 "response_time": 0,
#                 "error": str(e)
#             }


MODEL_TYPE = Literal['ollama', 'gemini']

class AIService:
    """
    Unified service that abstracts model selection and uses the common query interface.
    """
    def __init__(self, model_type: MODEL_TYPE = 'ollama'):
        """
        Initializes the service with the specified model type.

        :param model_type: The LLM backend to use ('ollama' or 'gemini').
        """
        self.model_type = model_type
        self._client: BaseAIService
        
        if model_type == 'ollama':
            self._client = OllamaAIService()
        # elif model_type == 'gemini':
        #     self._client = GeminiAIService()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'ollama' or 'gemini'.")
        
        logger.info(f"AIService initialized: {self.model_type}")

    def query(self, prompt: str) -> Dict[str, Any]:
        """
        Delegates the query to the underlying chosen LLM client.

        :param prompt: The complete text prompt to send to the model.
        :return: A dictionary containing the answer and metadata from the client.
        """
        return self._client.query(prompt)
    

def run_ai_query(backend: str, user_prompt: str):
    print(f"\n--- Running query against {backend.upper()} backend ---")
    
    try:
        # 2. Initialize the service, specifying the model type
        ai_service = AIService(model_type=backend)
        
        # 3. Call the unified query method
        response = ai_service.query(user_prompt)
        
        # 4. Process the structured dictionary response
        if response['error']:
            print(f"Error: {response['error']}")
        else:
            print(f"Model: {ai_service._client.model}")
            print(f"Response Time: {response['response_time']:.2f}s")
            print(f"Answer:\n{response['answer']}")
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":

    # Define the final prompt to be sent (fully engineered by the caller)
    QUERY = "What are the three main benefits of using a Parent Document Retriever in a RAG system?"
    # --- USAGE 1: Ollama Backend ---
    # Assumes Ollama is running locally and configured in app.config
    run_ai_query('ollama', QUERY)