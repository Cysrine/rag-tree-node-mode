"""Primary parser (Phase 1): structure-aware extraction via ``unstructured``.

Returns pre-tagged elements (Title / NarrativeText / ListItem / Table / ...) with
page numbers, geometry, and hierarchy hints (``parent_id``, ``category_depth``)
that the tree builder consumes directly. Tables are serialized to markdown.

Everything here is lazy-imported: the module loads without ``unstructured``
installed; only calling ``parse_with_unstructured`` requires it.
"""

from __future__ import annotations

from html.parser import HTMLParser

from rag_pipeline.ingestion.elements import (
    BBox,
    Element,
    ElementType,
    ParsedDocument,
    ParseSource,
)
from rag_pipeline.logging_setup import get_logger

logger = get_logger(__name__)

# unstructured category -> our ElementType. Note: unstructured tags both the
# document title and section headings as "Title"; depth is resolved in Phase 2
# using category_depth + order + geometry.
_CATEGORY_MAP: dict[str, ElementType] = {
    "Title": ElementType.TITLE,
    "Header": ElementType.HEADER,
    "Footer": ElementType.FOOTER,
    "NarrativeText": ElementType.PARAGRAPH,
    "Text": ElementType.PARAGRAPH,
    "UncategorizedText": ElementType.UNCATEGORIZED,
    "ListItem": ElementType.LIST_ITEM,
    "List": ElementType.LIST_ITEM,
    "Table": ElementType.TABLE,
    "FigureCaption": ElementType.CAPTION,
    "Caption": ElementType.CAPTION,
    "Formula": ElementType.FORMULA,
    "Image": ElementType.IMAGE,
    "PageNumber": ElementType.PAGE_NUMBER,
    "Address": ElementType.PARAGRAPH,
    "EmailAddress": ElementType.PARAGRAPH,
    "CompositeElement": ElementType.PARAGRAPH,
}

_STRATEGY_SOURCE = {
    "hi_res": ParseSource.UNSTRUCTURED_HI_RES,
    "fast": ParseSource.UNSTRUCTURED_FAST,
    "ocr_only": ParseSource.UNSTRUCTURED_OCR,
}


class _TableExtractor(HTMLParser):
    """Minimal <table> -> rows-of-cells extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _html_table_to_markdown(html: str) -> str:
    parser = _TableExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - never let table formatting break parsing
        return ""
    rows = [r for r in parser.rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    header, *body = rows
    lines = [fmt(header), "| " + " | ".join(["---"] * width) + " |"]
    lines += [fmt(r) for r in body]
    return "\n".join(lines)


def _bbox_from(coords: object) -> BBox | None:
    points = getattr(coords, "points", None)
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    system = getattr(coords, "system", None)
    return BBox(
        x0=min(xs),
        y0=min(ys),
        x1=max(xs),
        y1=max(ys),
        page_width=getattr(system, "width", None),
        page_height=getattr(system, "height", None),
    )


def parse_with_unstructured(
    path: str,
    *,
    strategy: str,
    hi_res_model: str,
    languages: list[str],
    infer_table_structure: bool,
) -> ParsedDocument:
    """Parse ``path`` with unstructured. ``strategy`` in {hi_res, fast, ocr_only}."""
    from unstructured.partition.pdf import partition_pdf  # lazy

    kwargs: dict = {
        "filename": str(path),
        "strategy": strategy,
        "languages": languages,
        "infer_table_structure": infer_table_structure,
        "extract_images_in_pdf": False,
    }
    if strategy == "hi_res":
        kwargs["hi_res_model_name"] = hi_res_model

    raw = partition_pdf(**kwargs)
    source = _STRATEGY_SOURCE.get(strategy, ParseSource.UNSTRUCTURED_HI_RES)

    elements: list[Element] = []
    max_page = 0
    title = None
    for el in raw:
        meta = el.metadata
        category = el.category if getattr(el, "category", None) else type(el).__name__
        etype = _CATEGORY_MAP.get(category, ElementType.UNCATEGORIZED)
        text = (el.text or "").strip()

        table_md = None
        if etype is ElementType.TABLE:
            html = getattr(meta, "text_as_html", None)
            if html:
                table_md = _html_table_to_markdown(html)
        display_text = table_md or text
        if not display_text:
            continue

        page = getattr(meta, "page_number", None) or 1
        max_page = max(max_page, page)

        element = Element(
            text=display_text,
            element_type=etype,
            page_number=page,
            bbox=_bbox_from(getattr(meta, "coordinates", None)),
            font=None,  # layout-model path has no reliable font metrics
            source=source,
            element_id=getattr(el, "id", None),
            parent_hint_id=getattr(meta, "parent_id", None),
            category_depth=getattr(meta, "category_depth", None),
            confidence=getattr(meta, "detection_class_prob", None),
            metadata={"category": category},
        )
        if table_md:
            element.metadata["text_as_html"] = getattr(meta, "text_as_html", None)
        elements.append(element)

        if title is None and etype is ElementType.TITLE and page == 1:
            title = display_text

    import os

    used_ocr = strategy == "ocr_only"
    logger.info(
        "unstructured(%s) parsed %s: %d elements, %d pages", strategy, path, len(elements), max_page
    )
    return ParsedDocument(
        source_path=str(path),
        title=title or os.path.splitext(os.path.basename(path))[0],
        elements=elements,
        page_count=max_page,
        strategy_used=f"unstructured:{strategy}",
        used_ocr=used_ocr,
    )
