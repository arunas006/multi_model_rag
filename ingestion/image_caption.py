from __future__ import annotations

import asyncio
from asyncio import tasks
import base64
import io
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from torch import chunk

from ingestion.chunker import Chunk
from ingestion.pdf_utils import pdf_to_images

logger = logging.getLogger(__name__)

_MIN_CROP_SIZE_PX: int = 50
_TABLE_MAX_INPUT_CHARS: int = 12000
_TABLE_MAX_TOKENS: int = 2000
_IMAGE_MAX_TOKENS: int = 800

def _parse_image_response(text:str) -> tuple[str, str]:

    caption =""
    for line in text.splitlines():
        if line.startswith("Caption:"):
            caption = line[len("Caption:"):].strip()
            break
    if not caption:
        caption = text.strip()[:200]
    return caption, text.strip()

def _parse_text_response(raw_original: str,enrich:str | None) -> tuple[str, str]:

    enriched_clean = (enrich or "").strip()
    return raw_original, enriched_clean if enriched_clean else raw_original

def _parse_table_json_response(raw_ocr :str, json_str:str) -> tuple[str, str]:

    try:
        table_data = json.loads(json_str)
    except (json.JSONDecodeError) as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return raw_ocr, raw_ocr
    
    markdown_table = table_data.get("markdown", "")
    summary = table_data.get("summary", "")

    if not markdown_table and not summary:
        logger.warning("JSON response does not contain 'markdown' or 'summary' fields.")
        return raw_ocr, raw_ocr
    
    caption = markdown_table if markdown_table else raw_ocr
    text = summary if summary else raw_ocr

    return caption, text

def _validate_table_extraction(
        num_rows_reported: int,
        markdown_table: str
) -> bool:
    
    if not markdown_table or num_rows_reported <= 0:    
        logger.warning("Markdown table is empty.")
        return True
    
    md_lines =[]

    for line in markdown_table.splitlines():
        if line.strip() and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
            md_lines.append(line.strip())

    actual_rows = max(0,len(md_lines) - 1)

    if num_rows_reported <=0:
        return True
    
    row_ratio = actual_rows / num_rows_reported
    if row_ratio < 0.7 and row_ratio > 1.5  :
        logger.warning(f"Low row ratio: Reported {num_rows_reported}, Actual {actual_rows} ({row_ratio:.2%}).")
        return False
    
    return True

def _get_surrounding_text(chunks: list[Chunk], idx:int, max_chars: int = 400) -> str:
    target = chunks[idx]
    parts : list[str] = []

    #look backward 
    for i in range(max(0, idx-2), idx):
        c = chunks[i]
        if c.modality == "text" and abs(c.page-target.page) <=1:
            parts.append(c.text[:max_chars])
    
    # Look forward
    for i in range(idx+1, min(len(chunks), idx+3)):
        c = chunks[i]
        if c.modality == "text" and abs(c.page-target.page) <=1:
            parts.append(c.text[:max_chars])
    
    combined = "....".join(parts)
    return combined[:max_chars * 2] if combined else ""

def _crop_chunk_to_base64(
        pdf_path: Path,
        chunk: Chunk,
        min_crop_size_px: int = _MIN_CROP_SIZE_PX
):
    if chunk.bbox is None:
        return None
    
    page_img = pdf_to_images(pdf_path, chunk.page-1,dpi=150)
    w,h =page_img.size

    bbox = chunk.bbox
    x1=int(bbox[0]*w/1000)
    y1=int(bbox[1]*h/1000)
    x2=int(bbox[2]*w/1000)
    y2=int(bbox[3]*h/1000)

    crop = page_img.crop((x1,y1,x2,y2))

    if crop.size[0] < min_crop_size_px or crop.size[1] < min_crop_size_px:
        logger.debug(f"Crop size {crop.size} is smaller than minimum {min_crop_size_px}px, skipping image extraction.")
        return None
    buffered = io.BytesIO()
    crop.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

_IMAGE_SYSTEM_PROMPT = """\
You are a scientific figure analysis assistant for a document retrieval system.

First, classify the figure into one of these types:
CHART — bar charts, line graphs, scatter plots, pie charts, heatmaps
DIAGRAM — flowcharts, architecture diagrams, block diagrams, network diagrams
PHOTO — photographs, microscopy images, medical scans
SCREENSHOT — UI screenshots, code screenshots, terminal output
OTHER — any figure that does not fit the above categories

Then analyze the figure and respond in EXACTLY this format with no extra text:

TYPE: <CHART | DIAGRAM | PHOTO | SCREENSHOT | OTHER>
CAPTION: <1-2 sentence description of what the figure shows overall — for semantic search.>
DETAIL:
- For CHART: describe chart type, all axis labels, data series names, key data points, and the main trend or comparison.
- For DIAGRAM: describe all components, their labels, connections, and the overall flow or hierarchy.
- For PHOTO: describe the subject, setting, notable features, and any annotations or labels.
- For SCREENSHOT: describe the UI elements, visible text, layout, and what operation is shown.
- For OTHER: describe the key visual components, their arrangement, and purpose.
STRUCTURE: <Grouping and containment relationships — which components belong to which group or module. Use dashes for sub-items.>

Be specific and technical. Reference labels, numbers, and text visible in the figure. Do not invent information not visible in the figure.\
"""

async def _enrich_single_image(
        chunk: Chunk,
        pdf_path: Path,
        surrounding_text: str,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        model: str    
):
    async with semaphore:
        try:
            b64 = _crop_chunk_to_base64(pdf_path, chunk)
            if not b64:
                logger.debug(f"Skipping image enrichment for chunk {chunk.id} due to small crop size.")
                chunk.text ="[figure]" 
                return 

            user_content: list[dict] =[]

            if surrounding_text:
                user_content.append(
                    {
                        "type":"text",
                        "text":f"Surrounding text from the document that may provide context about the figure:\n{surrounding_text}"
                    }
                )
            user_content.append(
                {
                    "type":"image_url",
                    "image_url":{"url": f"data:image/png;base64,{b64}"},
                }
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":_IMAGE_SYSTEM_PROMPT},
                    {"role":"user","content":user_content}
                ],
                max_tokens=_IMAGE_MAX_TOKENS,
                temperature=0.0
    
            )
            raw_response = (response.choices[0].message.content or "").strip()
            caption, full_text = _parse_image_response(raw_response)
            chunk.caption = caption
            chunk.text= full_text
            chunk.image_base64=b64

            logger.debug(f"Enriched image chunk {chunk.id} with caption: {caption}")

        except Exception:
            logger.warning("Image enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)
            chunk.text = "[figure]"

_TABLE_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given a table from a document, you MUST respond with valid JSON only — no text outside the JSON object.

Think step by step:
1. Count the number of columns (including row-label columns).
2. Count the number of data rows (excluding the header row).
3. Reproduce the COMPLETE table in markdown format with | delimiters. Include EVERY row and EVERY column — do not summarise, skip, or truncate any data. Use exact values from the original.
4. Write a 1-2 sentence semantic summary of what the table shows, for search indexing.

Respond in this exact JSON schema:
{
  "num_columns": <integer>,
  "num_rows": <integer, excluding header>,
  "markdown_table": "<complete markdown table with | delimiters — ALL rows, ALL columns, exact values>",
  "summary": "<1-2 sentence description of what this table shows, measures, or compares>"
}

Rules:
- For merged or spanning cells, repeat the value across all affected columns/rows.
- For empty cells, use "-" as a placeholder.
- Escape any pipe characters within cell values as \\|.
- Do not round, paraphrase, or abbreviate any numbers or text.\
"""

async def _enrich_single_table(
        chunk: Chunk,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        model: str,
        pdf_path: Path
):
    async with semaphore:
        try:
            raw = chunk.text
            if len(raw) > _TABLE_MAX_INPUT_CHARS:
                logger.warning(f"Table chunk {chunk.id} text length {len(raw)} exceeds max {_TABLE_MAX_INPUT_CHARS}, truncating for enrichment.")
                table_text = raw[:_TABLE_MAX_INPUT_CHARS] + "\n...[truncated]"
            else:
                table_text = raw

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":_TABLE_SYSTEM_PROMPT},
                    {"role":"user","content":f"Here is the extracted table text:\n{table_text}"}
                ],
                max_tokens=_TABLE_MAX_TOKENS,
                temperature=0.0,
                response_format={"type":"json_object"}
            )
            json_str = response.choices[0].message.content or ""
            caption, full_text = _parse_table_json_response(raw, json_str)

            try:
                data = json.loads(json_str)
                num_rows_reported = data.get("num_rows", 0)
                markdown_table = data.get("markdown_table", "")
                num_columns = data.get("num_columns", 0)

                if not _validate_table_extraction(num_rows_reported, markdown_table):
                    logger.info("Table validation failed for %s, retrying with correction",
                        chunk.chunk_id,)
                    caption, text = await _retry_table_extraction(
                        raw, table_text, num_rows_reported, client, model, semaphore,
                    )
            except json.JSONDecodeError as e:
              pass

            chunk.caption = caption
            chunk.text = full_text

            if pdf_path is not None and chunk.bbox is not None:
                try:
                    b64 = _crop_chunk_to_base64(pdf_path, chunk)
                    chunk.image_base64 = b64
                except Exception as e:
                    logger.error(f"Error cropping image for chunk {chunk.id}: {e}")


        except Exception:
            logger.warning("Table enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)

async def _retry_table_extraction(
        raw_ocr: str,
        table_text: str,
        prev_num_rows: int,
        client: AsyncOpenAI,
        model: str,
        semaphore: asyncio.Semaphore,
):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Here is a table from a research document:\n\n{table_text}\n\n"
                        f"IMPORTANT: A previous extraction reported {prev_num_rows} rows "
                        f"but the markdown output was incomplete. Please carefully count "
                        f"ALL rows and reproduce the COMPLETE table. Do not skip any rows."
                    ),
                },
            ],
            max_tokens=_TABLE_MAX_TOKENS,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        json_str = response.choices[0].message.content or ""
        caption, full_text = _parse_table_json_response(raw_ocr, json_str)
        return caption, full_text
    except Exception as e:
        logger.warning("Table retry also failed, using raw OCR")
        return raw_ocr, raw_ocr

_FORMULA_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given a mathematical formula or equation in LaTeX, respond in EXACTLY this format:

SUMMARY: <One sentence in plain English: what the formula computes or represents, its domain (e.g. probability, optimisation, signal processing), and where it typically appears.>
DETAIL: <Define each symbol or variable. List key properties and what the formula is used for.>

Use precise mathematical language but prefer plain English where equivalent.\
"""

_ALGORITHM_SYSTEM_PROMPT = """\
You are a scientific document analysis assistant for a document retrieval system.
Given pseudocode or an algorithm from a research paper, respond in EXACTLY this format:

SUMMARY: <One paragraph describing what the algorithm does, its purpose, and the problem it solves.>
DETAIL: <Cover: (1) inputs and outputs, (2) main steps or phases, (3) time and space complexity if determinable, (4) notable design decisions or properties.>

Use the variable names and terminology from the algorithm itself.\
"""

async def _enrich_formula_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    pdf_path: Path | None = None,
) -> None:
    """Generate a verbal formula description in-place."""
    async with semaphore:
        try:
            raw = chunk.text

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _FORMULA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is a formula from a research document:\n\n{raw}\n\n"
                            "Provide a verbal description for document retrieval."
                        ),
                    },
                ],
                max_tokens=350,
                temperature=0.0,
            )

            enriched = (response.choices[0].message.content or "").strip()
            chunk.caption, chunk.text = _parse_text_response(raw, enriched)

            # Also crop and store the visual for multimodal generation
            if pdf_path is not None and chunk.bbox is not None:
                try:
                    chunk.image_base64 = _crop_chunk_to_base64(pdf_path, chunk)
                except Exception:
                    logger.warning(
                        "Formula crop failed for chunk %s", chunk.chunk_id, exc_info=True
                    )

            logger.debug("Enriched formula chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning("Formula enrichment failed for chunk %s", chunk.chunk_id, exc_info=True)


async def _enrich_algorithm_single(
    chunk: Chunk,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    pdf_path: Path | None = None,
) -> None:
    """Generate a semantic algorithm description in-place."""
    async with semaphore:
        try:
            raw = chunk.text

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _ALGORITHM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Here is an algorithm from a research document:\n\n{raw}\n\n"
                            "Provide a semantic description for document retrieval."
                        ),
                    },
                ],
                max_tokens=450,
                temperature=0.0,
            )

            enriched = (response.choices[0].message.content or "").strip()
            chunk.caption, chunk.text = _parse_text_response(raw, enriched)

            # Also crop and store the visual for multimodal generation
            if pdf_path is not None and chunk.bbox is not None:
                try:
                    chunk.image_base64 = _crop_chunk_to_base64(pdf_path, chunk)
                except Exception:
                    logger.warning(
                        "Algorithm crop failed for chunk %s", chunk.chunk_id, exc_info=True
                    )

            logger.debug("Enriched algorithm chunk %s", chunk.chunk_id)

        except Exception:
            logger.warning(
                "Algorithm enrichment failed for chunk %s", chunk.chunk_id, exc_info=True
            )

async def enrich_chunk(
        chunks: list[Chunk],
        pdf_path: Path,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
        model: str = "gpt-4o",
        max_concurrent: int = 5
) -> list[Chunk]:
    
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    counts : dict[str,int] = defaultdict(int)
    for idx, chunk in enumerate(chunks):
        if chunk.modality == "image":
            if chunk.bbox is not None:
                context = _get_surrounding_text(chunks, idx)
                tasks.append(
                    _enrich_single_image(chunk, 
                                         pdf_path,
                                         context,
                                         client,
                                         semaphore,
                                         model,
                                        )
                )
                counts["image"] +=1
            else:
               logger.debug("Image chunk %s has no bbox; setting text='[figure]'", chunk.chunk_id)
               chunk.text = "[figure]"
        elif chunk.modality == "table":
            tasks.append(_enrich_single_table(chunk, client, semaphore, model, pdf_path))
            counts["table"] +=1
        elif chunk.modality == "formula":
            tasks.append(_enrich_formula_single(chunk, client, semaphore, model, pdf_path))
            counts["formula"] +=1
        elif chunk.modality == "algorithm":
            tasks.append(_enrich_algorithm_single(chunk, client, semaphore, model, pdf_path))
            counts["algorithm"] +=1
        
    if not tasks:
        logger.info("No chunks to enrich.")
        return chunks

    await asyncio.gather(*tasks)
   
    logger.info(
        "Enriched %d image / %d table / %d formula / %d algorithm chunks from %s",
        counts["image"], counts["table"], counts["formula"], counts["algorithm"],
        pdf_path.name
    )
    return chunks

async def enrich_image_chunks(
    chunks: list[Chunk],
    pdf_path: Path,
    client: AsyncOpenAI,
    max_concurrent: int = 5,
) -> list[Chunk]:
    
    return await enrich_chunk(
        chunks, pdf_path=pdf_path, client=client, max_concurrent=max_concurrent
    )
   
if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    # --- Mock Chunk class (since yours is imported) ---
    class Chunk:
        def __init__(self, chunk_id, modality, text, page=1, bbox=None):
            self.chunk_id = chunk_id
            self.modality = modality
            self.text = text
            self.page = page
            self.bbox = bbox

            # Outputs
            self.caption = None
            self.image_base64 = None

        def __repr__(self):
            return f"Chunk(id={self.chunk_id}, modality={self.modality}, text={self.text[:30]})"

    # --- Mock OpenAI client ---
    class MockResponse:
        def __init__(self, content):
            self.choices = [
                type("obj", (), {
                    "message": type("msg", (), {"content": content})
                })
            ]

    class MockClient:
        class chat:
            class completions:
                @staticmethod
                async def create(*args, **kwargs):
                    # Return dummy response based on prompt type
                    if "json_object" in str(kwargs):
                        return MockResponse('{"num_columns":2,"num_rows":2,"markdown_table":"|A|B|\\n|1|2|","summary":"Test table"}')
                    return MockResponse("Caption: Test caption\nDetail: Test detail")

    async def main():
        # --- Create sample chunks ---
        chunks = [
            Chunk("1", "image", "image content", bbox=(0, 0, 500, 500)),
            Chunk("2", "table", "col1 col2\n1 2\n3 4"),
            Chunk("3", "formula", "E = mc^2"),
            Chunk("4", "algorithm", "for i in range(n): pass"),
            Chunk("5", "text", "This is surrounding text")
        ]

        # --- Inputs ---
        pdf_path = Path("dummy.pdf")  # won't be used in mock
        client = MockClient()
        semaphore = asyncio.Semaphore(2)

        # --- Run enrichment ---
        enriched_chunks = await enrich_chunk(
            chunks=chunks,
            pdf_path=pdf_path,
            client=client,
            semaphore=semaphore,
            model="mock-model"
        )

        # --- Print results ---
        print("\n=== FINAL OUTPUT ===")
        for c in enriched_chunks:
            print(f"\nID: {c.chunk_id}")
            print(f"Modality: {c.modality}")
            print(f"Caption: {c.caption}")
            print(f"Text: {c.text}")
            print(f"Image present: {c.image_base64 is not None}")

    # --- Run async main ---
    asyncio.run(main())



   


  