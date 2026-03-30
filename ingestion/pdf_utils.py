from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

supported_extensions: frozenset[str] = frozenset({".pdf",".jpg",".jpeg",".png",".tiff",".gif"})

# --------------------------
# Converting PDF into Images
# --------------------------    

def pdf_to_images(pdf_path: Path,page_num: int,dpi: int=300) -> List[Image.Image]:

    """
    Convert a PDF file into a list of PIL Image objects, one per page.
    """
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    try:

        if page_num > len(doc):
            logger.error(f"Requested page number {page_num} exceeds total pages {len(doc)} in PDF.")
            raise ValueError(f"Requested page number {page_num} exceeds total pages {len(doc)} in PDF.")

        page = doc.load_page(page_num)

        # Scale resolution 
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is the default DPI for PDFs
        pix = page.get_pixmap(matrix=mat)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return image
    finally:
        doc.close()

#--------------------------
# Count Pages
#--------------------------

def count_pdf_pages(pdf_path: Path) -> int:

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc= fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()

