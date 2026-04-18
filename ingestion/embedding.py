from __future__ import annotations
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
from openai import AsyncOpenAI
from qdrant_client.models import SparseVector

if TYPE_CHECKING:
    from ingestion.chunker import Chunk
    from config import Settings

logger = logging.getLogger(__name__)

_BM25_N_FEATURES: int = 2**17

def _tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

async def embed_texts(
        texts: list[str],
        client: AsyncOpenAI,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100        
) -> list[list[float]]:
    
    sanitised = [text if text.strip() else "[empty]" for text in texts]

    embeddings: list[list[float]] = []
    for i in range(0, len(sanitised), batch_size):
        batch = sanitised[i:i + batch_size]
        response = await client.embeddings.create(input=batch, model=model,dimensions=dimensions)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

def compute_sparse_vector(
        texts: list[str],
        n_features: int = _BM25_N_FEATURES
) -> list[SparseVector]:
    
    vector : list[SparseVector] = []

    for text in texts:
        tokens = _tokenizer(text)
        if not tokens:
            vector.append(SparseVector(indices=[], values=[]))
            continue
        
        tf=Counter(tokens)
        total_terms = len(tokens)

        bucket_weights: dict[int, float] = {}
        for term, count in tf.items():
            bucket = hash(term) % n_features
            weight = count / total_terms
            bucket_weights[bucket] = weight

        sorted_buckets = sorted(bucket_weights.items())
        indices = [idx for idx, _ in sorted_buckets]
        values = [weight for _, weight in sorted_buckets]
        vector.append(SparseVector(indices=indices, values=values))
    return vector

class BaseEmbedder(ABC):

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        pass

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self,settings: Settings):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(
            texts=texts,
            client=self.client,
            model=self.model,
            dimensions=self.dimensions
        )
    
async def embed_chunks(chunks: list[Chunk], 
                       embedder: BaseEmbedder,
                       settings: Settings) -> tuple[list[list[float]], list[SparseVector]]:
    texts = [chunk.text for chunk in chunks]
    dense_embeddings = await embedder.embed(texts)
    sparse_embeddings = compute_sparse_vector(texts)
    return dense_embeddings, sparse_embeddings

if __name__ == "__main__":

    from dataclasses import dataclass

    @dataclass
    class TestChunk:
        text: str
    
    async def main():
        from config import Settings
        settings = Settings()
        embedder = OpenAIEmbedder(settings)
        test_chunks = [TestChunk(text="This is a test."), TestChunk(text="what is capital of india")]
        dense, sparse = await embed_chunks(test_chunks, embedder, settings)
        print("Dense embeddings:", dense)
        print("Sparse embeddings:", sparse)

    asyncio.run(main())