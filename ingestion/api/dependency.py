from __future__ import annotations
from functools import lru_cache
from openai import AsyncOpenAI

from config import Settings
from ingestion.embedding import BaseEmbedder,get_embedder
from ingestion.vdb import QdrantDocumentStore
from ingestion.reranker import BaseRanker,get_reranker

@lru_cache
def get_openai_client() -> AsyncOpenAI:
    settings = Settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return AsyncOpenAI(api_key=api_key)

@lru_cache
def get_store() -> QdrantDocumentStore:
    settings = Settings()
    return QdrantDocumentStore(settings)    

@lru_cache
def get_reranker() -> BaseRanker:
    settings = Settings()
    return get_reranker(settings)

@lru_cache
def get_embedder() -> BaseEmbedder:
    settings = Settings()
    return get_embedder(settings)