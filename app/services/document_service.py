"""
Document Processing Service.

Coordinates document upload,
processing, chunking, embedding,
and vector storage.
"""

import shutil
from pathlib import Path

from fastapi import UploadFile

from app.document_loader import DocumentLoader
from app.text_cleaner import TextCleaner
from app.chunker import DocumentChunker
from app.embeddings import DocumentEmbedder
from app.vector_store import VectorStore


class DocumentService:
    """
    Handles complete document ingestion.
    """

    def __init__(self):

        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()

        self.embedder = DocumentEmbedder()

        self.vector_store = VectorStore()

        # Load existing vector database
        # if it already exists.

        vector_folder = Path(
            "data/vector_store"
        )

        index_file = (
            vector_folder / "faiss.index"
        )

        documents_file = (
            vector_folder / "documents.pkl"
        )

        if (
            index_file.exists()
            and documents_file.exists()
        ):

            self.vector_store.load()

            print(
                "Existing vector database loaded."
            )

    def process_upload(
        self,
        file: UploadFile,
    ):

        # --------------------------------
        # Save uploaded file
        # --------------------------------

        upload_folder = Path(
            "data/uploads"
        )

        upload_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = (
            upload_folder / file.filename
        )

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # --------------------------------
        # Load document
        # --------------------------------

        raw_text = self.loader.load_document(
            str(file_path)
        )

        # --------------------------------
        # Clean text
        # --------------------------------

        clean_text = self.cleaner.clean(
            raw_text
        )

        documents = [

            {
                "filename": file.filename,

                "extension":
                    file_path.suffix.lower(),

                "size_kb":
                    round(
                        file_path.stat().st_size / 1024,
                        2
                    ),

                "text": clean_text,
            }

        ]

        # --------------------------------
        # Create chunks
        # --------------------------------

        chunks = self.chunker.split_documents(
            documents
        )

        # --------------------------------
        # Generate embeddings
        # --------------------------------

        embedded_chunks = (
            self.embedder.embed_chunks(chunks)
        )

        embeddings = [

            chunk["embedding"]

            for chunk in embedded_chunks

        ]

        # --------------------------------
        # Add to existing FAISS index
        # --------------------------------

        self.vector_store.add_embeddings(
            embeddings,
            embedded_chunks
        )

        # --------------------------------
        # Save updated vector database
        # --------------------------------

        self.vector_store.save()

        return {

            "status": "success",

            "filename": file.filename,

            "chunks_created":
                len(chunks),

            "total_chunks":
                len(
                    self.vector_store.documents
                ),

            "message":
                "Document processed successfully."

        }