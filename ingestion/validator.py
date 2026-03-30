from __future__ import annotations

from pathlib import Path

supported_extensions: frozenset[str] = frozenset({".pdf",".jpg",".jpeg",".png",".tiff",".gif"})


def validated_input_file(file_path: Path) -> None:

    """Validate that the input file exists and is of a supported type."""

    if not file_path.exists():

        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if file_path.suffix.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Supported types are: {supported_extensions}")