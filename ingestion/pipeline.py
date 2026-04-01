from __future__ import annotations

from curses import raw
import logging
from dataclasses import dataclass,field
from pathlib import Path
from typing import List,Any

from config import get_settings
from ingestion.pdf_utils import pdf_to_images,count_pdf_pages
from ingestion.post_processor import assemble_markdown

logger = logging.getLogger(__name__)

try:
    from glmocr import GLM_OCR
    _GLM_OCR_AVAILABLE = True
except ImportError:
    _GLM_OCR_AVAILABLE = False
    logger.warning("GLM_OCR library not found. OCR functionality will be unavailable.")

@dataclass
class ParseElement:
    label: str
    text: str
    bbox: List[float]
    score: float
    reading_order: int

@dataclass
class PageResult:
    page_num: int
    elements: List[ParseElement] = field(default_factory=list)
    markdown: str = ""

@dataclass
class ParseResult:
    source_file: str
    pages: List[PageResult] = field(default_factory=list)
    total_elements: int = 0
    full_markdown: str = ""

    @classmethod
    def from_sdk_result(cls,raw: Any,source_file: str) -> ParseResult:
        pages: List[PageResult] = []
        raw_pages : List[List[dict]] = getattr(raw,"json_result",[])
        full_markdown :str =getattr(raw,"markdown_result","") or ""

        for page_idx, raw_elements in enumerate(raw_pages):
            page_num = page_idx + 1
            elements: list[ParseElement] = []

            for raw_el in raw_elements:
                bbox_2d = raw_el.get("bbox_2d", [0, 0, 1, 1])
                el = ParseElement(
                    label=raw_el.get("label", "paragraph"),
                    text=raw_el.get("content", ""),
                    bbox=[float(v) for v in bbox_2d],
                    score=1.0,  # SDK does not provide a confidence score
                    reading_order=raw_el.get("index", len(elements)),
                )
                elements.append(el)

            markdown = assemble_markdown(elements)
            pages.append(PageResult(page_num=page_num, elements=elements, markdown=markdown))

        total_elements = sum(len(page.elements) for page in pages)
        return cls(
            source_file=source_file,
            pages=pages,
            total_elements=total_elements,
            full_markdown=full_markdown
        )



