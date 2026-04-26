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
from ingestion.pipeline import ParseElement

_CHUNKS_OUTPUT_DIR = Path("data/chunks")

settings = get_settings()
META_FILE = Path(settings.META_FILE)

router = APIRouter()

def update_source_metadata(chunks):
    new_sources = {c.source_file for c in chunks if c.source_file}

    if META_FILE.exists():
        try:
            existing = set(json.loads(META_FILE.read_text()))
        except:
            existing = set()
    else:
        existing = set()

    updated = existing.union(new_sources)

    META_FILE.write_text(json.dumps(sorted(updated), indent=2))

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
        parsed_file_path:Path,
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

    source_name = display_name or (pdf_path.name if pdf_path else parsed_file_path.name)

    if collection_override:
        store.collection_name = collection_override
    collection=store.collection_name

    t0 = time.perf_counter()

    if parsed_file_path:
    # Load parsed JSON
        with open(parsed_file_path) as f:
            parsed_json = json.load(f)

        logger.info("Loaded parsed JSON from {}", parsed_file_path)
        pages_data = []
        for p in parsed_json["pages"]:
            elements = [
                ParseElement(
                    label=el["label"],
                    text=el["text"],
                    bbox=el["bbox"],
                    score=el.get("score", 1.0),
                    reading_order=el["reading_order"],
                )
                for el in p["elements"]
            ]

            pages_data.append((p["page_num"], elements))
         
    else:
        if not pdf_path:
            raise HTTPException(status_code=400, detail="file_path required if parsed_file not provided")
        try:
            parser = DocumentParser()
            loop = asyncio.get_running_loop()
            parse_result = await loop.run_in_executor(None, parser.parse, pdf_path)
            pages_data = [(p.page_num, p.elements) for p in parse_result.pages]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Parsing failed for {}: {}", source_name, exc)
            raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc
    
    chunks = await asyncio.to_thread(
        document_aware_chunk,
        pages_data,
        source_name,    
        max_chunk_tokens
    )

    logger.info("Chunked {} → {} chunks", source_name, len(chunks))

    if caption and settings.image_caption_enabled and pdf_path:
        chunks = await enrich_chunk(
            chunks,
            pdf_path=pdf_path,
            client=client,
            semaphore = asyncio.Semaphore(3)
            
        )

    update_source_metadata(chunks)

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
    file: UploadFile | None = File(..., description="Document file to ingest (PDF or image)."),
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
            pdf_path=tmp_path,parsed_file_path=None, collection_override=collection,
            overwrite=overwrite,max_chunk_tokens=max_chunk_tokens, caption=caption,
            display_name=file.filename,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

@router.post("", response_model=IngestResponse, summary="Ingest document by file path")
async def ingest_by_path(req: IngestRequest) -> IngestResponse:
    """Ingest a document referenced by its local file path."""
    file_path = Path(req.file_path) if req.file_path else None
    parsed_file = Path(req.parsed_file) if req.parsed_file else None

    if not file_path and not parsed_file:
        raise HTTPException(
            status_code=400,
            detail="Either file_path or parsed_file must be provided"
        )
    if file_path and not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}"
        )
    
    if parsed_file and not parsed_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Parsed file not found: {parsed_file}"
        )
 
    return await _run_ingest(
        pdf_path=file_path,
        parsed_file_path=parsed_file,
        collection_override=req.collection,
        overwrite=req.overwrite,
        max_chunk_tokens=req.max_chunk_tokens,
        caption=req.caption)

   
  