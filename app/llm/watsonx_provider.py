"""
IBM watsonx.ai LLM provider.
"""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from app.llm.base import BaseLLMProvider


class WatsonXProvider(BaseLLMProvider):
    """
    Generates responses using IBM watsonx.ai.
    """

    def __init__(
        self,
        model_name: str,
        url: str,
        api_key: str,
        project_id: str
    ):

        self.model_name = model_name

        credentials = Credentials(
            url=url,
            api_key=api_key
        )

        self.model = ModelInference(
            model_id=model_name,
            credentials=credentials,
            project_id=project_id
        )

        print(
            f"\nWatsonx provider ready."
            f"\nModel: {self.model_name}\n"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2
    ) -> str:

        response = self.model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
            }
        )

        return response.strip()