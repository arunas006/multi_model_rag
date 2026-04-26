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

def _merge_nearby_bboxes(chunks: list[Chunk], y_threshold: int = 120, x_threshold: int = 500) -> list[Chunk]:
    """
    Merge nearby IMAGE chunks (charts/images split into fragments)
    - Only merges modality == "image"
    - Uses both vertical + horizontal proximity
    - Preserves text/table chunks
    """

    merged_chunks = []
    used = set()

    for i, c1 in enumerate(chunks):

        if (
            i in used
            or c1.bbox is None
            or (c1.modality or "").lower() != "image"
        ):
            continue

        x1, y1, x2, y2 = map(int, c1.bbox)
        used.add(i)

        for j, c2 in enumerate(chunks[i + 1:], start=i + 1):

            if (
                j in used
                or c2.bbox is None
                or (c2.modality or "").lower() != "image"
                or c1.page != c2.page
            ):
                continue

            x1b, y1b, x2b, y2b = map(int, c2.bbox)

            # 🔥 KEY: both vertical + horizontal proximity
            if (
                abs(y1 - y1b) < y_threshold and
                abs((x1 + x2) / 2 - (x1b + x2b) / 2) < x_threshold
            ):
                x1 = min(x1, x1b)
                y1 = min(y1, y1b)
                x2 = max(x2, x2b)
                y2 = max(y2, y2b)
                used.add(j)

        # update merged bbox
        c1.bbox = [x1, y1, x2, y2]
        merged_chunks.append(c1)

    # ✅ add remaining chunks (text, table, etc.)
    for i, c in enumerate(chunks):
        if i not in used:
            merged_chunks.append(c)

    # ✅ preserve reading order (VERY IMPORTANT)
    merged_chunks.sort(
        key=lambda c: (
            c.page,
            c.bbox[1] if c.bbox else 0
        )
    )

    return merged_chunks

import numpy as np

def _score_crop(crop):
    """
    Score crop based on visual richness
    Higher = more useful (charts, tables)
    Lower = blank / whitespace
    """
    gray = crop.convert("L")
    arr = np.array(gray)

    # standard deviation = variation
    return arr.std()

def _crop_chunk_to_base64(
        pdf_path: Path,
        chunk: Chunk,
        min_crop_size_px: int = _MIN_CROP_SIZE_PX
):
    
    page_img = pdf_to_images(pdf_path, chunk.page-1,dpi=150)
    w,h =page_img.size

    if chunk.bbox is None:
        crop = page_img
    else:
        x1, y1, x2, y2 = map(float, chunk.bbox)
        
    x1=int(x1*w/1000)
    y1=int(y1*h/1000)
    x2=int(x2*w/1000)
    y2=int(y2*h/1000)
    
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid bbox for chunk {chunk.chunk_id}: {chunk.bbox}")
        return None
        
    crop = page_img.crop((x1,y1,x2,y2))

    # print(f"""
    #         DEBUG CROP:
    #         Page size: {w}x{h}
    #         BBox: {bbox}
    #         Crop: ({x1},{y1},{x2},{y2})
    #         Width: {x2-x1}, Height: {y2-y1}
    #         """)

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
                logger.debug(f"Skipping image enrichment for chunk {chunk.chunk_id} due to small crop size.")
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

            logger.debug(f"Enriched image chunk {chunk.chunk_id} with caption: {caption}")

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

            table_text = re.sub(r"<.*?>", " ", raw)
            table_text = re.sub(r"\s+", " ", table_text).strip()

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
                    logger.error(f"Error cropping image for chunk {chunk.chunk_id}: {e}")


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

    chunks = _merge_nearby_bboxes(chunks)
    
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
        pdf_path.name if pdf_path else "parsed_json"
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

from PIL import ImageDraw
from ingestion.pdf_utils import pdf_to_images


def save_page_with_all_bboxes(chunks, pdf_path, debug_dir):
    debug_dir.mkdir(exist_ok=True)

    COLOR_MAP = {
        "image": "red",
        "table": "blue",
        "chart": "green",
    }

    pages = {}
    for c in chunks:
        if c.bbox is None:
            continue
        pages.setdefault(c.page, []).append(c)

    for page_num, page_chunks in pages.items():
        page_img = pdf_to_images(pdf_path, page_num - 1, dpi=150)
        draw = ImageDraw.Draw(page_img)

        w, h = page_img.size

        for c in page_chunks:
            x1, y1, x2, y2 = map(float, c.bbox)

            # ✅ NORMALIZED SCALE
            x1 = int(x1 * w / 1000)
            x2 = int(x2 * w / 1000)
            y1 = int(y1 * h / 1000)
            y2 = int(y2 * h / 1000)

            color = COLOR_MAP.get(c.modality, "yellow")

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 12)), f"{c.modality}-{c.chunk_id}", fill=color)

        file_path = debug_dir / f"page_{page_num}_debug.png"
        page_img.save(file_path)

        print(f"Saved → {file_path}")

def save_cropped_chunk_image(chunk: Chunk, pdf_path: Path, output_dir: Path):

    crop_dir = output_dir / "crops_from_func"
    crop_dir.mkdir(parents=True,exist_ok=True)

    for c in chunks:
        if c.bbox is None:
            continue

        try:
            b64 = _crop_chunk_to_base64(pdf_path, c)
            if not b64:
                print(f"Skipping chunk {c.chunk_id} (empty crop)")
                continue
            img_bytes = base64.b64decode(b64)
            file_path = crop_dir / f"chunk_{c.chunk_id}_{c.modality}_p{c.page}.png"

            with open(file_path, "wb") as f:
                f.write(img_bytes)

            print(f"Saved crop → {file_path}")

        except Exception as e:
            print(f"Error for chunk {c.chunk_id}: {e}")
   
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    import json

    from openai import AsyncOpenAI
    from config import get_settings

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    pdf_path = Path(r"C:\dev\multi_model_rag\ingestion\test_data\data\data_pdf.pdf")
    json_path = Path(r"C:\dev\multi_model_rag\ingestion\test_output\data_pdf_elements.json")
    output_md = Path(r"C:\dev\multi_model_rag\ingestion\test_output\captions.md")

    semaphore = asyncio.Semaphore(2)

    with open(json_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    chunks = []
    for i, el in enumerate(elements):
        label = el.get("labels", "").lower()
        text = el.get("texts", "")
        bbox = el.get("bboxes", None)



        if any(x in label for x in ["figure", "chart", "image"]):
            modality = "image"
        elif label in ["table"]:
            modality = "table"
        else:
            continue  # skip text

        chunk = Chunk(
            chunk_id=str(i),
            modality=modality,
            text=text,
            page=el.get("page", 1),
            bbox=bbox,
            element_types=[label],
            source_file=str(pdf_path),
            is_atomic=True
        )

        chunks.append(chunk)

    print("\n=== DEBUG BEFORE ENRICH ===")
    for c in chunks[:10]:
        print("ID:", c.chunk_id, "MOD:", c.modality, "BBOX:", c.bbox)

    print(f"Total chunks created: {len(chunks)}")

    # DEBUG_DIR = Path(r"C:\dev\multi_model_rag\ingestion\test_output\debug_pages")

    # # print("\n=== DRAWING ALL BBOXES (TABLE / IMAGE / CHART) ===")

    # # save_page_with_all_bboxes(chunks, pdf_path, DEBUG_DIR)
    # save_cropped_chunk_image(chunks, pdf_path, DEBUG_DIR)

    async def run():
        enriched = await enrich_chunk(
            chunks=chunks,
            pdf_path=pdf_path,
            client=client,
            semaphore=semaphore,
            model="gpt-4o"
        )
        return enriched

    enriched_chunks = asyncio.run(run())

    md_lines = ["# Extracted Captions\n"]
    for c in enriched_chunks:
        md_lines.append(f"## Chunk {c.chunk_id} ({c.modality})\n")

        if c.caption:
            md_lines.append(f"**Caption:** {c.caption}\n")

        if c.text:
            md_lines.append(f"**Detail:**\n{c.text}\n")

        md_lines.append("---\n")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n✅ Captions saved → {output_md}")




   


  