"""
LLM provider factory.

Creates the appropriate provider based
on application configuration.
"""

from app.config.settings import settings

from app.llm.ollama_provider import OllamaProvider
from app.llm.watsonx_provider import WatsonXProvider


def create_llm_provider():

    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":

        return OllamaProvider(
            model_name=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

    if provider == "watsonx":

        return WatsonXProvider(
            model_name=settings.LLM_MODEL,
            url=settings.WATSONX_URL,
            api_key=settings.WATSONX_APIKEY,
            project_id=settings.WATSONX_PROJECT_ID
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )