"""
Qdrant retriever — converts a question to a vector and searches for similar document chunks.
"""

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "/app/models/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self, host: str, port: int, collection: str) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection
        self._embedder = SentenceTransformer(EMBED_MODEL)
        logger.info("Retriever initialized (model=%s, collection=%s)", EMBED_MODEL, collection)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self._embedder.encode(query).tolist()

        results: list[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
        )

        chunks = []
        for point in results:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    score=point.score,
                )
            )

        logger.info("Retrieved %d chunks for query (top score=%.3f)", len(chunks), chunks[0].score if chunks else 0.0)
        return chunks

    def is_healthy(self) -> bool:
        try:
            self._client.get_collection(self._collection)
            return True
        except Exception:
            return False
