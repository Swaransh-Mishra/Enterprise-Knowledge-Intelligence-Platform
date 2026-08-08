"""
Pydantic models for semantic search.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Search request model.
    """

    query: str = Field(
        ...,
        min_length=3,
        description="Question or search query."
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return."
    )