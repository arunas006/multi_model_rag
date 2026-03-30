from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from ingestion.pdf_utils import count_pdf_pages, pdf_to_images
from ingestion.validator import validated_input_file

logger = logging.getLogger(__name__)

def load_pdf_as_images(pdf_path: Path) -> List[Image.Image]:

    validated_input_file(pdf_path)

    ext = pdf_path.suffix.lower()
    if ext == ".pdf":
        total_pages = count_pdf_pages(pdf_path)
        logger.info(f"PDF file has {total_pages} pages")

        images : List[Image.Image] = []
        for i in range(total_pages):
            try:
                
                img = pdf_to_images(pdf_path, i)

                if img is None:
                    raise ValueError(f"Failed to convert page {i} of PDF to image.")
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image for page {i} has invalid dimensions: {img.size}")
                
                images.append(img)
            except Exception as e:
                logger.error(f"Error processing page {i} of PDF: {e}")
                raise
            if len(images) != total_pages:
                logger.warning(f"Expected {total_pages} images but got {len(images)}. Some pages may have failed to convert.")

            logger.info(f"Successfully loaded {len(images)} images from PDF.")
        return images
    
    else:
        logger.info("Image detected. Loading as single image.")

        try:
            img = Image.open(pdf_path)
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Image has invalid dimensions: {img.size}")
            return [img]
        except Exception as e:
            logger.error(f"Error loading image file: {e}")

if __name__ == "__main__":

    from pathlib import Path
    # data_path = Path(r"C:\dev\multi_model_rag\ingestion\test_data\sample1.pdf")
    data_path = Path(r"C:\dev\multi_model_rag\ingestion\test_data\graph.jpg")
    output_path = Path(r"C:\dev\multi_model_rag\ingestion\test_data\output_images")

    print("PDF pages:", count_pdf_pages(data_path))

    images = load_pdf_as_images(data_path)

    print("Images returned:", len(images))

    output_path.mkdir(exist_ok=True)

    for i, img in enumerate(images):
        img.save(output_path / f"page_{i+1}.png")

    print(f"Saved {len(images)} images to {output_path}")

            

            
                