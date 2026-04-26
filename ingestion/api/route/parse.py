from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
import json

from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from ingestion.pipeline import DocumentParser
from ingestion.api.schema import ParseResponse, ParseRequest

router = APIRouter()

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
)

output_dir = Path("data/parsed")
output_dir.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def _run_parse(file_path: Path, display_name: str | None = None) -> ParseResponse:
    t0 = time.perf_counter()

    try:
        parser = DocumentParser()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, parser.parse, file_path)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Parsing failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    base_name = Path(display_name or result.source_file).stem
    result.save(output_dir, file_name=base_name)
    json_path = output_dir / f"{base_name}.json"

    return ParseResponse(
        source_file=display_name or result.source_file,
        pages=len(result.pages),
        total_elements=result.total_elements,
        latency_ms=round(latency_ms, 2),
        full_markdown=result.full_markdown,
        parsed_file=str(json_path),
        source_file_path=str(file_path)
    )


# 🔹 FILE UPLOAD VERSION (like /ingest/file)
@router.post("/file", response_model=ParseResponse, summary="Parse document via file upload")
async def parse_file(file: UploadFile = File(...)) -> ParseResponse:
    suffix = Path(file.filename).suffix.lower() if file.filename else ""

    if not file.filename or suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    import time
    base_name = Path(file.filename).stem
    unique_name = f"{base_name}_{int(time.time())}{suffix}"

    dest_path = UPLOAD_DIR / unique_name

    # ✅ save uploaded file permanently
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    # ✅ parse using saved file
    return await _run_parse(dest_path, display_name=file.filename)


# 🔹 PATH VERSION (like /ingest)
@router.post("", response_model=ParseResponse, summary="Parse document by file path")
async def parse_by_path(req: ParseRequest) -> ParseResponse:
    file_path = Path(req.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    return await _run_parse(file_path)