import cohere
from typing import List, Union
from ..utils.settings import settings


class EmbeddingService:
    def __init__(self):
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = "embed-english-v3.0"

    def embed_texts(
        self,
        texts: List[str],
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere API
        input_type: "search_document" for documents, "search_query" for queries
        """
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type
        )
        return response.embeddings

    def embed_single_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embedding for a single text
        """
        embeddings = self.embed_texts([text], input_type)
        return embeddings[0]

    def embed_batch(self, texts: List[str], input_type: str = "search_document", batch_size: int = 96) -> List[List[float]]:
        """
        Process embeddings in batches for efficiency (process 96 texts per batch for Cohere API efficiency)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch, input_type)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# Global instance
embedding_service = EmbeddingService()