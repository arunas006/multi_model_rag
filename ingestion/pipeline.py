from __future__ import annotations


import logging
from dataclasses import dataclass,field
from pathlib import Path
from typing import List,Any,Union
from PIL import Image
from tqdm import tqdm

from config import Settings, get_settings
from ingestion.pdf_utils import pdf_to_images,count_pdf_pages
from ingestion.post_processor import assemble_markdown,save_to_json

logger = logging.getLogger(__name__)

supported_extensions: frozenset[str] = frozenset({".pdf",".jpg",".jpeg",".png",".tiff",".gif"})


try:
    from glmocr import GlmOcr
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

    def save(self,output_dir:Path) -> None:
        save_to_json(self,output_dir)   

class DocumentParser:

    def __init__(self):

        settings = get_settings()
        if _GLM_OCR_AVAILABLE==False:
            raise ImportError("GLM_OCR library is required for DocumentParser but was not found.")  
        
        api_key = (
            settings.z_ai_api_key.get_secret_value() if settings.z_ai_api_key and settings.parser_backend == "cloud" else None
        )

        self._parser = GlmOcr(api_key=api_key, 
                               config_path=settings.config_yaml_path)
        
        logger.info(f"Initialized DocumentParser with backend: {settings.parser_backend}")

    def parse(self,file_path: Union[str, Path, Image.Image, List[Image.Image]]) -> ParseResult:

        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
            settings = get_settings()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if file_path.suffix.lower() not in supported_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            logger.info(f"Parsing file: {file_path}")
            parse_kwargs : dict[str, Any] ={}
            if file_path.suffix.lower() == ".pdf":
                total_page = count_pdf_pages(file_path)
                if settings.parser_backend == "cloud":
                    parse_kwargs["start_page_id"] = 0
                    parse_kwargs["end_page_id"] = total_page-1
                    logger.info(f"PDF has {total_page} pages. Sending all pages to cloud parser.")
                else:
                    logger.info("PDF has %d pages (PyMuPDF) — Ollama mode uses pypdfium2 "
                    "internally; page count may differ slightly",
                    total_page,
                )
            
                if settings.parser_backend == "ollama":
                    parse_kwargs["save_layout_visualization"] = False

                raw = self._parser.parse(str(file_path),**parse_kwargs)
                result = ParseResult.from_sdk_result(raw,source_file=str(file_path))
                if len(result.pages) != total_page:
                    logger.warning(f"Page count mismatch: PDF has {total_page} pages but parser returned {len(result.pages)} pages.")
                else:
                    logger.info(f"Successfully parsed PDF with {len(result.pages)} pages.")
                return result
            
        # Parsing Single Image   
            elif file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".gif"}:

                print("Parsing Single image file")
            
                logger.info("Parsing image from PIL Image object")

                parse_kwargs: dict[str, Any] = {}
                if settings.parser_backend == "ollama":
                    parse_kwargs["save_layout_visualization"] = False
                
                raw = self._parser.parse([str(file_path)],**parse_kwargs)
                source_name = getattr(file_path, "filename", None)
                source_name = Path(source_name).name if source_name else "image"
                return ParseResult.from_sdk_result(raw,source_file=source_name)
            
            
        else:   
            raise ValueError(f"Invalid input type: {type(file_path)}")
        
    def _pil_to_bytes(self,image: Image.Image) -> bytes:
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def parse_batch(self,file_paths: List[Union[str, Path, Image.Image]],output_dir: Path) -> List[ParseResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[ParseResult] = []
        for fp in tqdm(file_paths, desc="Parsing documents", unit="file"):
            try:
                result = self.parse(fp)
                result.save(output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error parsing {fp}: {e}", exc_info=True)
        return results
    
if __name__ == "__main__":

    file_path = r"C:\dev\multi_model_rag\ingestion\test_data\data"
    # file_path = r"C:\dev\multi_model_rag\ingestion\test_data\graph.jpg"
    output_dir = Path(r"C:\dev\multi_model_rag\ingestion\test_output")
    pdf_files = list(Path(file_path).glob("*.pdf"))
    parser = DocumentParser()
    try:
        result = parser.parse_batch(pdf_files, output_dir)

        for res in result:
            print("\n===== PARSE SUMMARY =====")
            print("Source:", res.source_file)
            print("Pages:", len(res.pages))
            print("Elements:", res.total_elements)
    
    except Exception as e:
        logger.error(f"Error during parsing: {e}", exc_info=True)



           


