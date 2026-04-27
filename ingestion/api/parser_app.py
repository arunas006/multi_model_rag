from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from ingestion.api.middleware import LoggingMiddleware
from ingestion.api.route.parse import router as parse_router

from config import get_settings
from utils.logging_config import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and report startup/shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    logger.info(
        "Starting doc-parser API | parser={} | backend={} | collection={}",
        settings.parser_backend,
        settings.reranker_backend,
        settings.qdrant_collection_name,
    )
    yield
    logger.info("Shutting down doc-parser API")

def create_app() -> FastAPI:
    """Construct and return the FastAPI application."""
    app = FastAPI(
        title="doc-parser RAG API",
        description="Multimodal RAG pipeline: PDF ingestion, hybrid search, and reranking.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(LoggingMiddleware)
    app.include_router(parse_router, prefix="/parse", tags=["parse"])


    return app


app = create_app