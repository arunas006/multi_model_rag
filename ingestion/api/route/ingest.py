from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Form,UploadFile
from loguru import logger
from qdrant_client.models import SparseVector

from ingestion.api.dependency import get_embedder_dep, get_openai_client, get_store
from ingestion.api.schema import IngestRequest, IngestResponse
from ingestion.chunker import Chunk, document_aware_chunk
from config import get_settings
from ingestion.embedding import embed_chunks
from ingestion.image_caption import enrich_chunk
from ingestion.pipeline import DocumentParser

_CHUNKS_OUTPUT_DIR = Path("data/chunks")

router = APIRouter()

def _save_chunks_to_disk(
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse: list[SparseVector],
    source_name: str,
) -> None:
    
    try:
        _CHUNKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(source_name).stem
        out_path = _CHUNKS_OUTPUT_DIR / f"{stem}.json"

        records = []

        for chunk, d_emb, s_emb in zip(chunks, dense, sparse, strict=False):
            records.append({
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "modality": chunk.modality,
                "element_types": chunk.element_types,
                "is_atomic": chunk.is_atomic,
                "bbox": chunk.bbox,
                "source_file": chunk.source_file,
                "text": chunk.text,
                "caption": chunk.caption,
                "dense_embedding": d_emb,
                "sparse_embedding": {
                    "indices": s_emb.indices,
                    "values": s_emb.values,
                },
            })

        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        logger.info("Saved {} chunks to {}", len(records), out_path)
    except Exception:
        logger.warning("Failed to save chunks debug file for {}", source_name, exc_info=True)

async def _run_ingest(
        pdf_path:Path,
        collection_override: str | None,
        overwrite:bool,
        max_chunk_tokens :int,
        caption:bool,
        display_name:str | None = None
) -> IngestResponse:
    
    settings = get_settings()
    client = get_openai_client()
    embedder = get_embedder_dep()
    store = get_store()

    source_name = display_name or pdf_path.name

    if collection_override:
        store.collection_name = collection_override
    collection=store.collection_name

    t0 = time.perf_counter()
    
    try:
        parser = DocumentParser()
        loop = asyncio.get_running_loop()
        parse_result = await loop.run_in_executor(None, parser.parse, pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Parsing failed for {}: {}", source_name, exc)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc
    
    chunks = await asyncio.to_thread(
        document_aware_chunk,
        [(p.page_num, p.elements) for p in parse_result.pages],
        source_name,    
        max_chunk_tokens
    )

    logger.info("Chunked {} → {} chunks", source_name, len(chunks))

    if caption and settings.image_caption_enabled:
        chunks = await enrich_chunk(
            chunks,
            pdf_path=pdf_path,
            client=client,
            semaphore = asyncio.Semaphore(3)
            
        )

    semaphore = asyncio.Semaphore(5)
    async def embed_chunk(batch):
        async with semaphore:
            return await embed_chunks(batch,embedder, settings)
    
    batches = [
        chunks[i:i+32] for i in range(0, len(chunks), 32)   
    ]

    embed_tasks = [embed_chunk(batch) for batch in batches]

    result = await asyncio.gather(*embed_tasks)

    dense_all, sparse_all = [], []
    for dense, sparse in result:
        dense_all.extend(dense)
        sparse_all.extend(sparse)

    _save_chunks_to_disk(chunks, dense_all, sparse_all, source_name)

    await store.create_collection(overwrite=overwrite)
    upserted = await store.upsert_chunks(
        chunks,
        dense_all,
        sparse_all,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    modality_counts = dict(Counter(c.modality for c in chunks))

    return IngestResponse(
        source_file=source_name,
        collection=collection,
        chunks_upserted=upserted,
        modality_counts=modality_counts,
        latency_ms=round(latency_ms, 2),
    )

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"})

@router.post("/file", response_model=IngestResponse, summary="Ingest document via file upload")
async def ingest_file(
    file: UploadFile = File(..., description="Document file to ingest (PDF or image)."),
    collection: str | None = Form(None, description="Override collection name. Leave blank to use the default from QDRANT_COLLECTION_NAME env var.", example=None),
    overwrite: bool = Form(False, description="Recreate collection before ingesting."),
    max_chunk_tokens: int = Form(512, ge=64, le=4096, description="Max tokens per chunk."),
    caption: bool = Form(True, description="Run GPT-4o captioning on image chunks."),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if not file.filename or suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return await _run_ingest(
            tmp_path, collection, overwrite, max_chunk_tokens, caption,
            display_name=file.filename,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

@router.post("", response_model=IngestResponse, summary="Ingest document by file path")
async def ingest_by_path(req: IngestRequest) -> IngestResponse:
    """Ingest a document referenced by its local file path."""
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return await _run_ingest(file_path, req.collection, req.overwrite, req.max_chunk_tokens, req.caption)

   
  