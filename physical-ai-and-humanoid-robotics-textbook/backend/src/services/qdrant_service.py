from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
from ..utils.settings import settings


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = "textbook_embeddings"

    def create_collection(self):
        """
        Create the textbook_embeddings collection with 1024 dimensions for Cohere embed-english-v3.0
        using Cosine distance and payload indexes for 'module' field
        """
        # Check if collection already exists
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection {self.collection_name} already exists")
            return
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,  # Cohere embed-english-v3.0 produces 1024-dimensional vectors
                    distance=models.Distance.COSINE
                )
            )

            # Create payload index for 'module' field to improve search performance
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="module",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            print(f"Collection {self.collection_name} created successfully")

    def upsert_vectors(self, points: List[Dict[str, Any]]):
        """
        Upsert vectors to the collection
        Each point should have: id, vector, payload
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_vectors(self, query_vector: List[float], top_k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        """
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )

        # Format results
        formatted_results = [
            {
                "content": hit.payload["content"],
                "module": hit.payload["module"],
                "section": hit.payload["section"],
                "url": hit.payload["url"],
                "score": hit.score
            }
            for hit in search_results
        ]

        return formatted_results

    def delete_collection(self):
        """
        Delete the collection (useful for testing/refreshing)
        """
        self.client.delete_collection(self.collection_name)


# Global instance
qdrant_service = QdrantService()