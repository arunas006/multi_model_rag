
"""Structure-aware and document-aware chunkers for RAG-ready document chunks."""

from __future__ import annotations
from dataclasses import dataclass,field
from typing import List,Any,Union
import logging

from ingestion.post_processor import Elementlike

logger = logging.getLogger(__name__)

_TOKEN_WORD_RATIO = 1.3

ATOMIC_LABELS: frozenset[str] = frozenset(
    {"table", "formula", "inline_formula", "algorithm", "image", "figure"}
)
TITLE_LABELS: frozenset[str] = frozenset(
    {"document_title", "paragraph_title", "figure_title"}
)
# Modality classification sets
_IMAGE_TYPES: frozenset[str] = frozenset({"image", "figure"})
_TABLE_TYPES: frozenset[str] = frozenset({"table"})
_FORMULA_TYPES: frozenset[str] = frozenset({"formula", "inline_formula"})
_ALGORITHM_TYPES: frozenset[str] = frozenset({"algorithm"})

def _infer_modality(element_type:List[str]) -> str:

    label = frozenset(element_type) 

    if label & _IMAGE_TYPES:
        return "image"
    elif label & _TABLE_TYPES:
        return "table"
    elif label & _FORMULA_TYPES:
        return "formula"
    elif label & _ALGORITHM_TYPES:
        return "algorithm"
    else:
        return "text"

@dataclass
class chunk:
    text:str
    chunk_id:str
    page:int
    element_types:List[str]
    bbox:List[float] | None 
    source_file:str
    is_atomic:bool
    modality:str = field(default="text")
    image_base64:str | None = field(default=None)
    caption:str | None =field(default=None)

def _estimate_tokens(text:str) -> int:
    return int(len(text.split()) * _TOKEN_WORD_RATIO)

def _split_text_into_chunks(text:str,chunk_size:int,overlap:int) -> List[str]:

    words=text.split()
    words_per_chunk = max(1, int(chunk_size / _TOKEN_WORD_RATIO))
    overlap_words = max(0, int(overlap / _TOKEN_WORD_RATIO))
    sub_chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        sub_chunks.append(" ".join(chunk_words))
        start = end - overlap_words

        if start <=0:
            start = end
    return sub_chunks


  

   




