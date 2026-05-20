"""Phase 1 + 1.5: PDF -> ordered structural elements.

Public surface:
    load_pdf(path)            -> ParsedDocument   (strategy routing + fallback)
    ParsedDocument, Element   -> the intermediate representation
"""

from __future__ import annotations

from rag_pipeline.ingestion.elements import (
    BBox,
    Element,
    ElementType,
    FontInfo,
    ParsedDocument,
    ParseSource,
)
from rag_pipeline.ingestion.pdf_loader import load_pdf

__all__ = [
    "load_pdf",
    "ParsedDocument",
    "Element",
    "ElementType",
    "FontInfo",
    "BBox",
    "ParseSource",
]
