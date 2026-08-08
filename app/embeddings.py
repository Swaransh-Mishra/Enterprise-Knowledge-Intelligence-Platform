"""
Document Embedding Module.

This module converts text chunks into dense vector embeddings
using a Sentence Transformer model.

These embeddings are later stored inside a FAISS vector database
to enable semantic search.
"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DocumentEmbedder:
    """
    Generates vector embeddings for document chunks.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize embedding model.

        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model.
        """

        print(f"\nLoading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded successfully.\n")

    def embed_chunks(
        self,
        chunks: list
    ) -> list:
        """
        Generate embeddings for all chunks.

        Parameters
        ----------
        chunks : list
            List containing chunk dictionaries.

        Returns
        -------
        list
            Chunk dictionaries with embeddings added.
        """

        embedded_chunks = []

        for chunk in tqdm(
            chunks,
            desc="Generating Embeddings"
        ):

            embedding = self.model.encode(
                chunk["text"],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            chunk["embedding"] = embedding

            embedded_chunks.append(chunk)

        return embedded_chunks