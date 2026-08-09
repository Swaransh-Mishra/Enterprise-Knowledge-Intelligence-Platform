"""
FAISS Vector Store Module.

This module stores document embeddings inside a FAISS index
and performs semantic similarity search.
"""

import os
import faiss
import numpy as np
import pickle


class VectorStore:
    """
    Handles creation, saving, loading,
    and searching of the FAISS vector database.
    """

    def __init__(self, embedding_dimension=384):

        self.embedding_dimension = embedding_dimension

        self.index = faiss.IndexFlatL2(
            embedding_dimension
        )

        self.documents = []

    # --------------------------------
    # Add Embeddings
    # --------------------------------

    def add_embeddings(
        self,
        embeddings,
        chunks
    ):
        """
        Adds embeddings and corresponding text chunks.
        """

        embeddings = np.array(
            embeddings
        ).astype("float32")

        self.index.add(
            embeddings
        )

        self.documents.extend(
            chunks
        )

    # --------------------------------
    # Remove Existing Document
    # --------------------------------

    def remove_document(
        self,
        document_name
    ):
        """
        Removes all chunks belonging
        to a specific document.
        """

        if not self.documents:

            return 0

        document_name = document_name.lower()

        remaining_documents = [
            document
            for document in self.documents
            if document.get(
                "document_name",
                ""
            ).lower() != document_name
        ]

        removed_count = (
            len(self.documents)
            -
            len(remaining_documents)
        )

        if removed_count == 0:

            return 0

        # --------------------------------
        # Collect Remaining Embeddings
        # --------------------------------

        remaining_embeddings = []

        for document in remaining_documents:

            embedding = document.get(
                "embedding"
            )

            if embedding is not None:

                remaining_embeddings.append(
                    embedding
                )

        # --------------------------------
        # Rebuild FAISS Index
        # --------------------------------

        self.index = faiss.IndexFlatL2(
            self.embedding_dimension
        )

        if remaining_embeddings:

            embeddings = np.array(
                remaining_embeddings
            ).astype("float32")

            self.index.add(
                embeddings
            )

        self.documents = remaining_documents

        return removed_count

    # --------------------------------
    # Search
    # --------------------------------

    def search(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Returns the most similar chunks.
        """

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for idx in indices[0]:

            if idx != -1:

                results.append(
                    self.documents[idx]
                )

        return results

    # --------------------------------
    # Save
    # --------------------------------

    def save(
        self,
        folder="data/vector_store"
    ):
        """
        Saves FAISS index and chunk metadata.
        """

        os.makedirs(
            folder,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            os.path.join(
                folder,
                "faiss.index"
            )
        )

        with open(
            os.path.join(
                folder,
                "documents.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )

    # --------------------------------
    # Load
    # --------------------------------

    def load(
        self,
        folder="data/vector_store"
    ):
        """
        Loads existing FAISS index.
        """

        self.index = faiss.read_index(
            os.path.join(
                folder,
                "faiss.index"
            )
        )

        with open(
            os.path.join(
                folder,
                "documents.pkl"
            ),
            "rb"
        ) as f:

            self.documents = pickle.load(
                f
            )