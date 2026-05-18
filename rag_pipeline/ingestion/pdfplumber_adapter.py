"""Fallback parser (Phase 1): raw layout recovery via ``pdfplumber``.

Used when the hi-res ``unstructured`` stack is unavailable or fails. Produces
faithful, *line-level* elements carrying geometry + font metrics — exactly the
raw signals Phase 2 needs to cluster fonts and assign depth. It deliberately
does NOT classify titles/headings (that's the tree builder's job); everything is
PARAGRAPH except obvious bullet/numbered LIST_ITEMs.

Column separation: words sharing a visual line are split at wide horizontal gaps,
so a left-column line and a right-column line never merge into one element. The
final left-to-right / top-to-bottom ordering is done in ``reading_order``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from statistics import median
from typing import Any

from rag_pipeline.ingestion.elements import (
    BBox,
    Element,
    ElementType,
    FontInfo,
    ParsedDocument,
    ParseSource,
)
from rag_pipeline.logging_setup import get_logger

logger = get_logger(__name__)

# Leading bullet or short enumerator, e.g. "•", "-", "(a)", "1.", "iii)"
_LIST_PREFIX = re.compile(r"^\s*([•▪◦‣·\-–*]|\(?[0-9a-zA-Z]{1,3}[.)])\s+")


def _dominant(pairs: Iterable[tuple[Any, float]]) -> Any:
    """Return the value carrying the most weight (e.g. most characters)."""
    weights: dict[Any, float] = {}
    for value, weight in pairs:
        if value is None:
            continue
        weights[value] = weights.get(value, 0.0) + weight
    if not weights:
        return None
    return max(weights.items(), key=lambda kv: kv[1])[0]


def _segment_font(words: list[dict]) -> FontInfo:
    sizes = [(round(float(w.get("size") or 0.0), 1), len(w.get("text", "")) or 1) for w in words]
    names = [(w.get("fontname"), len(w.get("text", "")) or 1) for w in words]
    size = _dominant(sizes)
    name = _dominant(names)
    return FontInfo.from_fontname(name, float(size) if size else None)


def _rows_from_words(words: list[dict]) -> list[list[dict]]:
    """Cluster words into visual rows by their top coordinate."""
    if not words:
        return []
    heights = [float(w["bottom"]) - float(w["top"]) for w in words]
    tol = max(2.0, 0.5 * median(heights))
    ordered = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))
    rows: list[list[dict]] = []
    current: list[dict] = [ordered[0]]
    row_top = float(ordered[0]["top"])
    for w in ordered[1:]:
        if abs(float(w["top"]) - row_top) <= tol:
            current.append(w)
        else:
            rows.append(current)
            current = [w]
            row_top = float(w["top"])
    rows.append(current)
    return rows


def _split_row_into_segments(row: list[dict], page_width: float) -> list[list[dict]]:
    """Split a row at wide horizontal gaps (column gutters / tab stops)."""
    row = sorted(row, key=lambda w: float(w["x0"]))
    gap_threshold = max(15.0, 0.06 * page_width)
    segments: list[list[dict]] = []
    current: list[dict] = [row[0]]
    for prev, w in zip(row, row[1:], strict=False):
        if float(w["x0"]) - float(prev["x1"]) > gap_threshold:
            segments.append(current)
            current = [w]
        else:
            current.append(w)
    segments.append(current)
    return segments


def _element_from_segment(
    words: list[dict], page_number: int, page_width: float, page_height: float
) -> Element | None:
    text = " ".join(w.get("text", "") for w in words).strip()
    if not text:
        return None
    x0 = min(float(w["x0"]) for w in words)
    x1 = max(float(w["x1"]) for w in words)
    top = min(float(w["top"]) for w in words)
    bottom = max(float(w["bottom"]) for w in words)
    etype = ElementType.LIST_ITEM if _LIST_PREFIX.match(text) else ElementType.PARAGRAPH
    return Element(
        text=text,
        element_type=etype,
        page_number=page_number,
        bbox=BBox(x0, top, x1, bottom, page_width=page_width, page_height=page_height),
        font=_segment_font(words),
        source=ParseSource.PDFPLUMBER,
    )


def parse_with_pdfplumber(path: str) -> ParsedDocument:
    """Parse ``path`` into line-level elements. Order is finalized by reading_order."""
    import pdfplumber  # lazy: keeps package import cheap

    elements: list[Element] = []
    page_count = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            pw, ph = float(page.width), float(page.height)
            words = page.extract_words(
                extra_attrs=["size", "fontname"],
                keep_blank_chars=False,
                use_text_flow=False,
            )
            for row in _rows_from_words(words):
                for segment in _split_row_into_segments(row, pw):
                    el = _element_from_segment(segment, page.page_number, pw, ph)
                    if el is not None:
                        elements.append(el)

    title = _guess_title(elements, path)
    logger.info("pdfplumber parsed %s: %d elements, %d pages", path, len(elements), page_count)
    return ParsedDocument(
        source_path=path,
        title=title,
        elements=elements,
        page_count=page_count,
        strategy_used="fallback",
    )


def _guess_title(elements: list[Element], path: str) -> str:
    """Cheap title guess: largest-font text on page 1 (tree builder refines later)."""
    import os

    sized = [
        (e, e.font.size)
        for e in elements
        if e.page_number == 1 and e.font is not None and e.font.size
    ]
    if sized:
        biggest_el, biggest_size = max(sized, key=lambda pair: pair[1])
        # Only trust it if it's clearly larger than the page-1 median body size.
        sizes = sorted(size for _, size in sized)
        body = sizes[len(sizes) // 2]
        if biggest_size >= body * 1.2 and len(biggest_el.text) <= 200:
            return biggest_el.preview(120)
    return os.path.splitext(os.path.basename(path))[0]
