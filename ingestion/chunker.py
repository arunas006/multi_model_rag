
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
class Chunk:
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

def document_aware_chunk(element:list[tuple[int,List[Elementlike]]],
                         source_file:str,
                         max_chunk_size:int = 512,
                         overlap:int = 50) -> list[Chunk]:

    all_pairs : List[tuple[int,str]] = []

    for page_num,ele in element:
        for el in ele:
            all_pairs.append((page_num,el))

    if not all_pairs:
        return []
    
    all_pairs.sort(key=lambda x: (x[0], x[1].reading_order))

    chunks: list[Chunk] = []
    chunk_id=0

    current_texts : list[str]=[]
    current_labels : list[str]=[]
    current_tokens: int = 0
    current_page: int = all_pairs[0][0]

    pending_title: str | None = None
    pending_title_label: str | None = None
    pending_title_page: int = current_page

    def flush_current() -> None:
        nonlocal current_texts, current_labels, current_page, current_tokens,chunk_id
        nonlocal pending_title, pending_title_label, pending_title_page

        if not current_texts and not pending_title:
            return
        
        texts_to_flush : list[str] = []
        labels_to_flush : list[str] = []

        page_to_use = pending_title_page if (pending_title and not current_texts) else current_page

        if pending_title is not None:
            texts_to_flush.append(pending_title)
            labels_to_flush.append(pending_title_label or "paragraph_title")
            pending_title = None
            pending_title_label = None
            
        texts_to_flush.extend(current_texts)
        labels_to_flush.extend(current_labels)

        if not texts_to_flush:
            return  
        
        chunk= Chunk(
            text=" ".join(texts_to_flush),
            chunk_id=f"{source_file}_{page_to_use}_{chunk_id}",
            page=page_to_use,
            element_types=labels_to_flush,
            bbox=None,
            source_file=source_file,
            is_atomic=False,
            modality=_infer_modality(labels_to_flush)
        )
        chunks.append(chunk)
        chunk_id +=1
        current_texts = []
        current_labels = []
        current_tokens = 0

    for page_num, ele in all_pairs:
        label= ele.label
        text = ele.text.strip()

        if label in ATOMIC_LABELS:
            figure_caption: str | None = None
            if pending_title is not None and pending_title_label == "figure_title":
                figure_caption = pending_title
                pending_title = None
                pending_title_label = None
            
            flush_current()

            if figure_caption:
                atomic_text = f"{figure_caption}\n\n{text}" if text else figure_caption
                atomic_labels =["figure_title", label] 
            else:
                atomic_text = text
                atomic_labels = [label]

            atomic_chunk = Chunk(
                text=atomic_text,
                chunk_id=f"{source_file}_{page_num}_{chunk_id}",
                page=page_num,
                element_types=atomic_labels,
                bbox=ele.bbox,
                source_file=source_file,
                is_atomic=True,
                modality=_infer_modality(atomic_labels),
            )
            chunks.append(atomic_chunk)
            chunk_id +=1
            continue

        if not text:
            continue

        
        if label in TITLE_LABELS:
            if current_texts:
                flush_current()
            elif pending_title is not None:
                flush_current()
            pending_title = text
            pending_title_label = label
            pending_title_page = page_num
            continue

        token_estimate = _estimate_tokens(text)
        pending_tokens = _estimate_tokens(pending_title) if pending_title else 0

        if token_estimate > max_chunk_size:
            flush_current()
            sub_chunks = _split_text_into_chunks(text, max_chunk_size, overlap=overlap)
            for sub_chunk in sub_chunks:
                chunk = Chunk(
                    text=sub_chunk,
                    chunk_id=f"{source_file}_{page_num}_{chunk_id}",
                    page=page_num,
                    element_types=[label],
                    bbox=None,
                    source_file=source_file,
                    is_atomic=False,
                    modality=_infer_modality([label]),
                )
                chunks.append(chunk)
                chunk_id +=1
            continue

        if current_tokens + token_estimate + pending_tokens > max_chunk_size:
            flush_current()
            
        if pending_title is not None:
            if not current_texts:
                current_page = pending_title_page
            current_texts.append(pending_title)
            current_labels.append(pending_title_label)
            current_tokens += _estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None
        
        if not current_texts:
            current_page = page_num
        current_texts.append(text)
        current_labels.append(label)
        current_tokens += token_estimate

        if current_tokens >= max_chunk_size:
            flush_current()
    flush_current()
    return chunks

def structure_aware_chunk(elements: list[Elementlike], 
                          source_file:str,
                          page:int,
                          max_chunk_size: int, overlap: int) -> list[chunk]:
    
    final_chunk = document_aware_chunk([(page,elements)],
                                       source_file,max_chunk_size, overlap)
    
    return final_chunk

if __name__ == "__main__":
    import json
    from pathlib import Path


    from dataclasses import dataclass

    @dataclass
    class TestElement:
        label: str
        text: str
        bbox: list
        score: float
        reading_order: int

    test_elements = [
    TestElement(label="document_title", text="Test Document", bbox=[0,0,100,20], score=0.9, reading_order=1),
    TestElement(label="paragraph_title", text="Introduction", bbox=[0,30,100,50], score=0.9, reading_order=2),
    TestElement(label="paragraph", text="This is the first paragraph of the document.", bbox=[0,60,100,80], score=0.9, reading_order=3),
    TestElement(label="paragraph", text="This is the second paragraph of the document.", bbox=[0,90,100,110], score=0.9, reading_order=4),
    TestElement(label="table", text="Table content here", bbox=[0,120,100,150], score=0.9, reading_order=5)
    ]

    chunks = structure_aware_chunk(test_elements, "test_file.pdf", page=1, max_chunk_size=50, overlap=10)
    for c in chunks:
        print(json.dumps(c.__dict__, indent=2)) 



        
    

   




