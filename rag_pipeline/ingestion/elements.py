"""Intermediate representation for parsed PDF content.

This is the contract between the parser (Phase 1) and the tree builder (Phase 2).
A parser produces a flat, reading-ordered list of ``Element``s carrying every
signal the tree builder needs: element type, page, geometry (bbox), font metrics,
and any hierarchy hints the source parser already provides (unstructured tags
Title/heading depth; pdfplumber gives raw font sizes to cluster).

Deliberately dependency-free (stdlib only) so it imports anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class ElementType(str, Enum):
    """Parser-level element categories (a superset of the DB ``node_type``).

    The tree builder maps these down to the storage vocabulary
    (title | heading | subheading | paragraph | list_item | article | table).
    """

    TITLE = "title"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    ARTICLE = "article"
    TABLE = "table"
    CAPTION = "caption"
    HEADER = "header"        # running page header (usually dropped later)
    FOOTER = "footer"        # running page footer
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    IMAGE = "image"
    UNCATEGORIZED = "uncategorized"


class ParseSource(str, Enum):
    """Provenance of an element / document — useful for debugging + confidence."""

    UNSTRUCTURED_HI_RES = "unstructured_hi_res"
    UNSTRUCTURED_FAST = "unstructured_fast"
    UNSTRUCTURED_OCR = "unstructured_ocr"
    PDFPLUMBER = "pdfplumber"


@dataclass
class BBox:
    """Bounding box in PDF points. Origin is top-left (y grows downward)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float | None = None
    page_height: float | None = None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class FontInfo:
    """Dominant font metrics for an element (drives Phase 2 depth clustering)."""

    size: float | None = None
    name: str | None = None
    bold: bool = False
    italic: bool = False

    @classmethod
    def from_fontname(cls, name: str | None, size: float | None) -> FontInfo:
        low = (name or "").lower()
        bold = any(t in low for t in ("bold", "black", "heavy", "semibold", "-bd"))
        italic = any(t in low for t in ("italic", "oblique", "-it"))
        return cls(size=size, name=name, bold=bold, italic=italic)


@dataclass
class Element:
    """One parsed text block (a line/span/paragraph/table cell-group)."""

    text: str
    element_type: ElementType
    page_number: int
    # Global reading-order position within the document. Assigned by reading_order;
    # -1 until then.
    order_index: int = -1

    bbox: BBox | None = None
    font: FontInfo | None = None
    source: ParseSource = ParseSource.PDFPLUMBER

    # Hierarchy hints the source parser already knows (may be None).
    element_id: str | None = None       # parser's stable id for this element
    parent_hint_id: str | None = None   # parser-provided parent id (unstructured)
    category_depth: int | None = None   # parser-provided nesting depth (unstructured)

    confidence: float | None = None     # 0..1 detection confidence, if known
    metadata: dict = field(default_factory=dict)

    def preview(self, width: int = 90) -> str:
        one_line = " ".join(self.text.split())
        return one_line if len(one_line) <= width else one_line[: width - 1] + "…"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["element_type"] = self.element_type.value
        d["source"] = self.source.value
        return d


@dataclass
class ParsedDocument:
    """A fully parsed document: metadata + a reading-ordered element list."""

    source_path: str
    title: str
    elements: list[Element] = field(default_factory=list)
    page_count: int = 0

    strategy_used: str = ""
    is_scanned: bool = False
    used_ocr: bool = False
    used_fallback: bool = False
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # --- convenience ---
    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for el in self.elements:
            counts[el.element_type.value] = counts.get(el.element_type.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "title": self.title,
            "page_count": self.page_count,
            "strategy_used": self.strategy_used,
            "is_scanned": self.is_scanned,
            "used_ocr": self.used_ocr,
            "used_fallback": self.used_fallback,
            "warnings": self.warnings,
            "type_counts": self.type_counts(),
            "metadata": self.metadata,
            "elements": [el.to_dict() for el in self.elements],
        }
