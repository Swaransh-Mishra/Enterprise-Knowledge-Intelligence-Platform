"""
Hybrid Search Engine.

Combines semantic search and keyword search
with document-aware retrieval.
"""

import re

import numpy as np

from app.search_engine import SemanticSearch
from app.keyword_search import KeywordSearch


class HybridSearch:
    """
    Hybrid retrieval using semantic similarity
    and keyword matching.
    """

    def __init__(self):

        print("\nLoading hybrid search...")

        self.semantic_search = SemanticSearch()
        self.keyword_search = KeywordSearch()

        print("Hybrid search ready.\n")

    # --------------------------------
    # Main Search
    # --------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        requested_document = self.find_document(query)

        if requested_document:

            return self.search_document(
                query=query,
                document_name=requested_document,
                top_k=top_k
            )

        return self.search_all_documents(
            query=query,
            top_k=top_k
        )

    # --------------------------------
    # Search Specific Document
    # --------------------------------

    def search_document(
        self,
        query: str,
        document_name: str,
        top_k: int = 5
    ) -> list:

        self.semantic_search.reload()

        documents = (
            self.semantic_search.vector_store.documents
        )

        document_chunks = [
            document
            for document in documents
            if document.get(
                "document_name",
                ""
            ).lower() == document_name.lower()
        ]

        if not document_chunks:

            return []

        # --------------------------------
        # Semantic Scores
        # --------------------------------

        query_embedding = (
            self.semantic_search.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        texts = [
            chunk.get("text", "")
            for chunk in document_chunks
        ]

        chunk_embeddings = (
            self.semantic_search.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        semantic_scores = np.dot(
            chunk_embeddings,
            query_embedding[0]
        )

        # --------------------------------
        # Keyword Scores
        # --------------------------------

        keyword_vectorizer = (
            self.keyword_search.vectorizer
        )

        keyword_matrix = (
            keyword_vectorizer.transform(texts)
        )

        query_vector = (
            keyword_vectorizer.transform(
                [query]
            )
        )

        keyword_scores = (
            query_vector @ keyword_matrix.T
        ).toarray().flatten()

        # --------------------------------
        # Normalize Scores
        # --------------------------------

        normalized_semantic = (
            self.normalize_scores(
                semantic_scores.tolist()
            )
        )

        normalized_keyword = (
            self.normalize_scores(
                keyword_scores.tolist()
            )
        )

        # --------------------------------
        # Combine Scores
        # --------------------------------

        results = []

        for index, chunk in enumerate(
            document_chunks
        ):

            semantic_score = (
                normalized_semantic[index]
            )

            keyword_score = (
                normalized_keyword[index]
            )

            combined_score = (
                0.7 * semantic_score
                +
                0.3 * keyword_score
            )

            result = chunk.copy()

            result["semantic_score"] = float(
                semantic_score
            )

            result["keyword_score"] = float(
                keyword_score
            )

            result["combined_score"] = float(
                combined_score
            )

            results.append(result)

        # --------------------------------
        # Sort
        # --------------------------------

        results.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # --------------------------------
        # Remove Duplicate Chunks
        # --------------------------------

        final_results = []

        seen_chunks = set()

        for result in results:

            chunk_id = result.get(
                "chunk_id"
            )

            if chunk_id in seen_chunks:
                continue

            seen_chunks.add(chunk_id)

            result["rank"] = (
                len(final_results) + 1
            )

            final_results.append(result)

            if len(final_results) >= top_k:
                break

        return final_results

    # --------------------------------
    # Search All Documents
    # --------------------------------

    def search_all_documents(
        self,
        query: str,
        top_k: int
    ) -> list:

        semantic_results = (
            self.semantic_search.search(
                query=query,
                top_k=10
            )
        )

        keyword_results = (
            self.keyword_search.search(
                query=query,
                top_k=10
            )
        )

        # --------------------------------
        # Normalize Semantic Scores
        # --------------------------------

        semantic_scores = [
            result.get(
                "distance",
                0.0
            )
            for result in semantic_results
        ]

        normalized_semantic = (
            self.normalize_scores(
                semantic_scores
            )
        )

        combined = {}

        for result, score in zip(
            semantic_results,
            normalized_semantic
        ):

            chunk_id = result["chunk_id"]

            combined[chunk_id] = result.copy()

            combined[chunk_id][
                "semantic_score"
            ] = score

            combined[chunk_id][
                "keyword_score"
            ] = 0.0

        # --------------------------------
        # Normalize Keyword Scores
        # --------------------------------

        keyword_scores = [
            result.get(
                "keyword_score",
                0.0
            )
            for result in keyword_results
        ]

        normalized_keyword = (
            self.normalize_scores(
                keyword_scores
            )
        )

        for result, score in zip(
            keyword_results,
            normalized_keyword
        ):

            chunk_id = result["chunk_id"]

            if chunk_id not in combined:

                combined[chunk_id] = result.copy()

                combined[chunk_id][
                    "semantic_score"
                ] = 0.0

            combined[chunk_id][
                "keyword_score"
            ] = score

        # --------------------------------
        # Calculate Hybrid Score
        # --------------------------------

        results = []

        for chunk in combined.values():

            semantic_score = chunk.get(
                "semantic_score",
                0.0
            )

            keyword_score = chunk.get(
                "keyword_score",
                0.0
            )

            combined_score = (
                0.7 * semantic_score
                +
                0.3 * keyword_score
            )

            chunk["combined_score"] = (
                combined_score
            )

            results.append(chunk)

        # --------------------------------
        # Entity-Aware Retrieval
        # --------------------------------

        results = self.apply_entity_filter(
            query=query,
            results=results
        )

        # --------------------------------
        # Sort Results
        # --------------------------------

        results.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # --------------------------------
        # Final Results
        # --------------------------------

        final_results = []

        seen_chunks = set()

        for result in results:

            chunk_id = result["chunk_id"]

            if chunk_id in seen_chunks:
                continue

            seen_chunks.add(chunk_id)

            result["rank"] = (
                len(final_results) + 1
            )

            final_results.append(result)

            if len(final_results) >= top_k:
                break

        return final_results

    # --------------------------------
    # Extract Query Entities
    # --------------------------------

    def extract_query_entities(
        self,
        query: str
    ) -> list:

        entities = []

        # --------------------------------
        # Project names
        # Example:
        # Project Phoenix
        # Project Atlantis
        # --------------------------------

        project_matches = re.findall(
            r"\bProject\s+[A-Za-z0-9_-]+",
            query,
            flags=re.IGNORECASE
        )

        for entity in project_matches:

            entity = entity.strip()

            if entity.lower() not in [
                item.lower()
                for item in entities
            ]:

                entities.append(entity)

        # --------------------------------
        # Two-word proper names
        # Example:
        # Neha Gupta
        # Strategic Layer
        # --------------------------------

        proper_name_matches = re.findall(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            query
        )

        for entity in proper_name_matches:

            entity = entity.strip()

            if entity.lower() in [
                item.lower()
                for item in entities
            ]:
                continue

            entities.append(entity)

        return entities

    # --------------------------------
    # Apply Entity Filter
    # --------------------------------

    def apply_entity_filter(
        self,
        query: str,
        results: list
    ) -> list:

        if not results:

            return results

        entities = self.extract_query_entities(
            query
        )

        # --------------------------------
        # No specific entity
        # Keep normal hybrid retrieval
        # --------------------------------

        if not entities:

            return results

        # --------------------------------
        # Find chunks containing entities
        # --------------------------------

        entity_results = []

        for result in results:

            text = result.get(
                "text",
                ""
            )

            document_name = result.get(
                "document_name",
                ""
            )

            searchable_text = (
                f"{document_name} {text}"
            ).lower()

            matched_entities = 0

            for entity in entities:

                if entity.lower() in searchable_text:

                    matched_entities += 1

            if matched_entities > 0:

                result = result.copy()

                result["entity_match"] = (
                    matched_entities
                )

                # --------------------------------
                # Small ranking boost
                # --------------------------------

                result["combined_score"] = (
                    result.get(
                        "combined_score",
                        0.0
                    )
                    +
                    0.15 * matched_entities
                )

                entity_results.append(
                    result
                )

        # --------------------------------
        # If entity was found in results,
        # use entity-matching results only.
        # --------------------------------

        if entity_results:

            return entity_results

        # --------------------------------
        # If no entity was found,
        # return original retrieval results.
        #
        # This prevents the filter from
        # destroying normal semantic search.
        # --------------------------------

        return results

    # --------------------------------
    # Find Document
    # --------------------------------

    def find_document(
        self,
        query: str
    ):

        query = query.lower()

        documents = (
            self.semantic_search.vector_store.documents
        )

        document_names = sorted(
            set(
                document.get(
                    "document_name",
                    ""
                ).lower()
                for document in documents
            ),
            key=len,
            reverse=True
        )

        for document_name in document_names:

            if document_name and document_name in query:

                return document_name

        return None

    # --------------------------------
    # Normalize Scores
    # --------------------------------

    def normalize_scores(
        self,
        scores
    ):

        if not scores:

            return []

        minimum = min(scores)

        maximum = max(scores)

        if maximum == minimum:

            return [
                1.0
                for _ in scores
            ]

        return [
            (score - minimum)
            /
            (maximum - minimum)
            for score in scores
        ]