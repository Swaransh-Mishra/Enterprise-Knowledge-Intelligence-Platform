"""
Document Chunking Module.

This module splits cleaned document text into smaller overlapping chunks
that can later be converted into embeddings and stored in a vector database.
"""

from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits documents into overlapping chunks for semantic search.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, documents: list) -> list:
        """
        Split multiple documents into chunks.

        Parameters
        ----------
        documents : list
            List containing loaded document dictionaries.

        Returns
        -------
        list
            List of chunk dictionaries.
        """

        all_chunks = []

        for document in documents:

            text = document["text"]

            text_chunks = self.text_splitter.split_text(text)

            total_chunks = len(text_chunks)

            for chunk_index, chunk in enumerate(text_chunks):

                chunk_data = {
                    "chunk_id": f"{document['filename']}_{chunk_index}",

                    "document_name": document["filename"],

                    "document_type": document["extension"],

                    "page_number": None,

                    "chunk_number": chunk_index + 1,

                    "total_chunks": total_chunks,

                    "characters": len(chunk),

                    "created_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "text": chunk
                }

                all_chunks.append(chunk_data)

        return all_chunks