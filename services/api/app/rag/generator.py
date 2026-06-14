from abc import ABC, abstractmethod
from typing import Optional
import httpx
import logging
logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        pass

class GroqClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"

    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

def get_llm_client(settings) -> BaseLLMClient:
    return GroqClient(
        api_key=settings.GROQ_API_KEY,
        model=settings.LLM_MODEL
    )
