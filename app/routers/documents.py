"""
Document Management Routes.
"""

from fastapi import APIRouter, File, UploadFile

from app.services.document_service import DocumentService
from app.vector_store import VectorStore


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


document_service = DocumentService()


@router.get("/")
async def list_documents():
    """
    Return documents currently stored in the knowledge base.
    """

    vector_store = VectorStore()

    try:
        vector_store.load()
    except Exception:
        return {
            "documents": [],
            "total_documents": 0
        }

    documents = {}

    for chunk in vector_store.documents:

        document_name = chunk.get(
            "document_name",
            "Unknown"
        )

        if document_name not in documents:

            documents[document_name] = {
                "filename": document_name,
                "type": chunk.get(
                    "document_type",
                    ""
                ),
                "chunks": chunk.get(
                    "total_chunks",
                    0
                ),
                "status": "Ready"
            }

    document_list = list(documents.values())

    return {
        "documents": document_list,
        "total_documents": len(document_list)
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload and process a document.
    """

    result = document_service.process_upload(file)

    return result