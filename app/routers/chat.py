"""
Chat Routes.

Provides an endpoint for asking questions
about uploaded enterprise documents.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.chat_engine import ChatEngine


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


chat_engine = ChatEngine()


class ChatRequest(BaseModel):

    question: str

    history: list = []


@router.post("/")
async def chat(request: ChatRequest):

    """
    Ask a question using the enterprise
    knowledge base and conversation history.
    """

    result = chat_engine.ask(

        question=request.question,

        history=request.history

    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }