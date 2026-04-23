from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    HnswConfig,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams
)

from ingestion.embedding import BaseEmbedder, compute_sparse_vector
if TYPE_CHECKING:
    from config import Settings
    from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

class QdrantDocumentStore:

    def __init__(self,settings: Settings):
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        )
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
            prefer_grpc=True
        )
        self.collection_name = settings.qdrant_collection_name
        self.settings = settings

    async def create_collection(self,overwrite: bool = False) -> None:
        response = await self.client.get_collections()
        existing = {c.name for c in response.collections}

        if self.collection_name in existing:
            if not overwrite:
                logger.info(f"Collection '{self.collection_name}' already exists. Skipping creation.")
                return
            logger.info(f"Collection '{self.collection_name}' already exists. Deleting for overwrite.")
            await self.client.delete_collection(collection_name=self.collection_name)

        logger.info(f"Creating collection '{self.collection_name}' with vector and sparse index.")
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_dense": VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfig(
                        m=16,
                        ef_construction=200,
                        full_scan_threshold=1000
                    )
                )
            },
            sparse_vectors_config={
                "bm25_sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

    async def delete_collection(self, collection_name: str) -> None:

        response = await self.client.get_collections()
        existing = {c.name for c in response.collections}
        if collection_name not in existing:
            logger.warning("Collection '%s' does not exist.", collection_name)
            return False
        await self.client.delete_collection(collection_name)
        logger.info("Deleted collection '%s'", collection_name)
        return True
    
    async def upsert_chunks(self, 
                            chunks: list[Chunk], 
                            dense_embedding: list[list[float]],
                            sparse_embedding: list[SparseVector],
                            batch_size: int =64) -> int:
        
        if len(chunks) != len(dense_embedding) or len(chunks) != len(sparse_embedding):
            raise ValueError("Length of chunks, dense_embedding, and sparse_embedding must be the same.")
        
        points : list[PointStruct] = []
        for chunk,dense,sparse in zip(chunks,dense_embedding,sparse_embedding,strict=False):
            payload = {
                "text": chunk.text,
                "chunk_id": chunk.id,
                "soucre_file": chunk.source_file,
                "page": chunk.page,
                "element_types": chunk.element_types,
                "bbox": chunk.bbox,
                "is_atomic": chunk.is_atomic,
                "modality": chunk.modality,
                "image_base64": chunk.image_base64,
                "caption": chunk.caption
            }
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={"text_dense": dense,"bm25_sparse": sparse},
                payload=payload
            )   
            )
        total =0
        for i in range(0,len(points),batch_size):
            batch = points[i:i+batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            total += len(batch)
            logger.info(f"Upserted {total}/{len(points)} chunks.")
        return total
        
    async def search(self, 
                     query: str, 
                     embedder: BaseEmbedder,
                     settings: Settings,
                     top_k: int = 10,
                     filter_modality :str | None = None,
    ) -> list[dict]:
        query_dense = (await embedder.embed([query]))[0]
        querysparse = compute_sparse_vector([query])[0]

        query_filter = None
        if filter_modality is not None:
            from qdrant_client.models import FieldCondition, Filter,MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="modality",
                        match=MatchValue(
                            value=filter_modality
                        )
                    )
                ]
            )
        result = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch = [
                Prefetch(query=query_dense,using="text_dense",limit=top_k*2),
                Prefetch(query=querysparse,using="bm25_sparse",limit=top_k*2),
            ],
            query = FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            filter=query_filter
        
        )
        return [point.payload for point in result.points]

if __name__ == "__main__":

    import asyncio
    from config import get_settings
    settings = get_settings()
    from qdrant_client.models import VectorParams, Distance

    async def full_test():
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value(),
            prefer_grpc=True
        )

        # Create test collection
        await client.create_collection(
            collection_name="test_connection",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )

        print("✅ Collection created")

        # Insert point
        await client.upsert(
            collection_name="test_connection",
            points=[
                PointStruct(
                    id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "my_id")),
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"test": "ok"}
                )
            ]
        )
        print("✅ Insert successful")

        # Search
        result = await client.query_points(
                collection_name="test_connection",
                query=[0.1, 0.2, 0.3, 0.4],
                limit=1
            )

        print("✅ Search successful:", result)

    asyncio.run(full_test())