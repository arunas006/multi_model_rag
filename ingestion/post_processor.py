from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any,Protocol,runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable
class Elementlike(Protocol):
    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int

SKIP_LABELS : frozenset[str] = frozenset({"image", "seal", "page_number"})
PROMPT_MAP : dict[str,str] = {
    "document_title": lambda t: f"# {t}",
    "paragraph_title": lambda t: f"## {t}",
    "abstract": lambda t: f"**Abstract:** {t}",
    "table": lambda t: t,
    "formula": lambda t: f"\n$$\n{t}\n$$\n",
    "inline_formula": lambda t: f"\n$$\n{t}\n$$\n",
    "code_block": lambda t: f"```\n{t}\n```",
    "footnotes": lambda t: f"\n---\n{t}",
    "algorithm": lambda t: f"```\n{t}\n```"
}

def assemble_markdown(elements: list[Elementlike]) -> str:
    if not elements:
        return ""
    sorted_elements = sorted(elements, key=lambda el: el.reading_order)
    
    markdown_parts: list[str] = []
    for el in sorted_elements:
        if el.label in SKIP_LABELS:
            logger.debug(f"Skipping element with label '{el.label}' and text '{el.text[:30]}...'")
            continue
        prompt_func = PROMPT_MAP.get(el.label)
        if prompt_func:
            markdown_parts.append(prompt_func(el.text))
        else:
            logger.debug(f"No prompt mapping for label '{el.label}'. Using raw text.")
            markdown_parts.append(el.text)
    return "\n\n".join(markdown_parts).strip()