"""
Keyword Search Engine.

Uses TF-IDF to find document chunks
that match important words from the query.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.vector_store import VectorStore


class KeywordSearch:
    """
    Keyword retrieval using TF-IDF.
    """

    def __init__(self):

        print("\nLoading keyword search...")

        self.vector_store = VectorStore()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        self.documents = []
        self.matrix = None

        self.reload()

        print("Keyword search ready.\n")

    # --------------------------------
    # Reload Indexed Documents
    # --------------------------------

    def reload(self):

        self.vector_store.load()

        self.documents = self.vector_store.documents

        texts = [
            document["text"]
            for document in self.documents
        ]

        if texts:

            self.matrix = self.vectorizer.fit_transform(
                texts
            )

        else:

            self.matrix = None

    # --------------------------------
    # Search
    # --------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        self.reload()

        if not self.documents or self.matrix is None:

            return []

        query_vector = self.vectorizer.transform(
            [query]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        ranked_indexes = scores.argsort()[::-1]

        results = []

        for rank, idx in enumerate(
            ranked_indexes[:top_k],
            start=1
        ):

            if scores[idx] <= 0:
                continue

            chunk = self.documents[idx].copy()

            chunk["rank"] = rank

            chunk["keyword_score"] = float(
                scores[idx]
            )

            results.append(chunk)

        return results