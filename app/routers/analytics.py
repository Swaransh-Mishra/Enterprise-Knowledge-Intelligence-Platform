"""
Analytics API.

Provides live platform statistics
and active AI configuration.
"""

from pathlib import Path
import pickle

from fastapi import APIRouter

from app.config.settings import settings

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


@router.get("/")
async def get_analytics():

    vector_folder = Path("data/vector_store")

    # --------------------------------
    # Read stored knowledge base
    # --------------------------------

    documents = set()
    chunks = 0

    documents_file = (
        vector_folder / "documents.pkl"
    )

    if documents_file.exists():

        try:

            with open(
                documents_file,
                "rb"
            ) as file:

                stored_chunks = pickle.load(file)

            chunks = len(stored_chunks)

            for chunk in stored_chunks:

                document_name = chunk.get(
                    "document_name"
                )

                if document_name:
                    documents.add(document_name)

        except Exception:

            documents = set()
            chunks = 0

    # --------------------------------
    # Platform statistics
    # --------------------------------

    return {

        "documents": len(documents),

        "chunks": chunks,

        "status": "Operational",

        "embedding_model":
            "all-MiniLM-L6-v2",

        "vector_database":
            "FAISS",

        "llm_provider":
            settings.LLM_PROVIDER,

        "llm":
            settings.LLM_MODEL
    }