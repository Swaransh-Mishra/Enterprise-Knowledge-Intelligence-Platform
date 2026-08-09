"""
Base interface for LLM providers.
"""


class BaseLLMProvider:
    """
    Common interface used by all LLM providers.
    """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.2
    ) -> str:
        """
        Generate a response from the language model.
        """

        raise NotImplementedError(
            "LLM providers must implement generate()."
        )