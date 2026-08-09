"""
LLM provider package.
"""

from app.llm.factory import create_llm_provider


def get_llm():
    """
    Create the configured LLM provider.
    """

    return create_llm_provider()