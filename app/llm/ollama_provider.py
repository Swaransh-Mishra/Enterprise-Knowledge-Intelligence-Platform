"""
Ollama LLM provider.
"""

import requests

from app.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Generates responses using a local Ollama model.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434"
    ):

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

        print(
            f"\nOllama provider ready."
            f"\nModel: {self.model_name}\n"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2
    ) -> str:

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()