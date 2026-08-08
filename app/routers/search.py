"""
Semantic and Hybrid Search Routes.
"""

from fastapi import APIRouter, HTTPException

from app.models import SearchRequest
from app.hybrid_search import HybridSearch


router = APIRouter(
    prefix="/search",
    tags=["Search"]
)


search_engine = HybridSearch()


@router.post("/")
async def search_documents(request: SearchRequest):
    """
    Search indexed documents using hybrid retrieval.
    """

    try:

        results = search_engine.search(
            query=request.query,
            top_k=request.top_k
        )

        formatted_results = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            formatted_results.append(
                {
                    "rank": rank,
                    "document": result["document_name"],
                    "chunk_id": result["chunk_id"],
                    "chunk_number": result["chunk_number"],
                    "preview": result["text"][:300],
                    "combined_score": round(
                        result["combined_score"],
                        3
                    )
                }
            )

        return {
            "query": request.query,
            "results_found": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )