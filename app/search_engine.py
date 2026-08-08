"""
Semantic Search Engine.

Converts a user query into an embedding and retrieves
the most relevant document chunks from the FAISS index.
"""

import numpy as np

from sentence_transformers import SentenceTransformer

from app.vector_store import VectorStore


class SemanticSearch:
    """
    Semantic retrieval using Sentence Transformers + FAISS.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):

        print("\nLoading search model...")

        self.model = SentenceTransformer(model_name)

        self.vector_store = VectorStore()

        self.reload()

        print("Search engine ready.\n")

    # --------------------------------
    # Reload Vector Database
    # --------------------------------

    def reload(self):

        self.vector_store.load()

    # --------------------------------
    # Search
    # --------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        self.reload()

        if not self.vector_store.documents:

            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = self.vector_store.index.search(
            query_embedding.astype(np.float32),
            top_k
        )

        results = []

        for rank, idx in enumerate(
            indices[0]
        ):

            if idx == -1:
                continue

            chunk = self.vector_store.documents[idx].copy()

            chunk["rank"] = rank + 1

            chunk["distance"] = float(
                distances[0][rank]
            )

            results.append(chunk)

        return results