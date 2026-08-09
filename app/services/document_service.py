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

        # --------------------------------
        # Load Existing Vector Database
        # --------------------------------

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

    # --------------------------------
    # Process Upload
    # --------------------------------

    def process_upload(
        self,
        file: UploadFile
    ):

        # --------------------------------
        # Validate Filename
        # --------------------------------

        if not file.filename:

            raise ValueError(
                "Uploaded file has no filename."
            )

        filename = Path(
            file.filename
        ).name

        extension = Path(
            filename
        ).suffix.lower()

        # --------------------------------
        # Validate Extension
        # --------------------------------

        if extension not in (
            self.loader.SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported file type: "
                f"{extension}"
            )

        # --------------------------------
        # Upload Folder
        # --------------------------------

        upload_folder = Path(
            "data/uploads"
        )

        upload_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path = (
            upload_folder / filename
        )

        # --------------------------------
        # Save Uploaded File
        # --------------------------------

        try:

            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            # --------------------------------
            # Check Empty File
            # --------------------------------

            if file_path.stat().st_size == 0:

                raise ValueError(
                    "The uploaded file is empty."
                )

            # --------------------------------
            # Load Document
            # --------------------------------

            raw_text = (
                self.loader.load_document(
                    str(file_path)
                )
            )

            # --------------------------------
            # Clean Text
            # --------------------------------

            clean_text = (
                self.cleaner.clean(
                    raw_text
                )
            )

            if not clean_text.strip():

                raise ValueError(
                    "No readable text was found "
                    "in the uploaded document."
                )

            # --------------------------------
            # Create Document Object
            # --------------------------------

            documents = [

                {
                    "filename": filename,

                    "extension":
                        extension,

                    "size_kb":
                        round(
                            file_path.stat().st_size
                            / 1024,
                            2
                        ),

                    "text": clean_text,
                }

            ]

            # --------------------------------
            # Create Chunks
            # --------------------------------

            chunks = (
                self.chunker.split_documents(
                    documents
                )
            )

            if not chunks:

                raise ValueError(
                    "No document chunks were "
                    "created from the uploaded file."
                )

            # --------------------------------
            # Generate Embeddings
            # --------------------------------

            embedded_chunks = (
                self.embedder.embed_chunks(
                    chunks
                )
            )

            if not embedded_chunks:

                raise ValueError(
                    "No embeddings were created "
                    "for the uploaded document."
                )

            embeddings = [

                chunk["embedding"]

                for chunk in embedded_chunks

            ]

            # --------------------------------
            # Replace Existing Document
            # --------------------------------

            removed_chunks = (
                self.vector_store.remove_document(
                    filename
                )
            )

            if removed_chunks:

                print(
                    f"Removed {removed_chunks} "
                    f"existing chunks for "
                    f"{filename}."
                )

            # --------------------------------
            # Add New Document
            # --------------------------------

            self.vector_store.add_embeddings(
                embeddings,
                embedded_chunks
            )

            # --------------------------------
            # Save Vector Database
            # --------------------------------

            self.vector_store.save()

            return {

                "status": "success",

                "filename": filename,

                "chunks_created":
                    len(chunks),

                "total_chunks":
                    len(
                        self.vector_store.documents
                    ),

                "message":
                    "Document processed successfully."
            }

        except Exception:

            # --------------------------------
            # Remove failed upload
            # --------------------------------

            if file_path.exists():

                try:

                    file_path.unlink()

                except OSError:

                    pass

            raise