"""Phase 1.5: reading-order normalization.

Sorts a flat element list into logical reading order and stamps ``order_index``.
For now this is page -> top-to-bottom; multi-column handling comes next.

Elements lacking geometry (e.g. from unstructured, which already emits reading
order) keep their input order for that page.
"""

from __future__ import annotations

from collections import OrderedDict

from rag_pipeline.ingestion.elements import Element


def normalize(elements: list[Element]) -> list[Element]:
    """Return ``elements`` reordered into reading order, with ``order_index`` set."""
    if not elements:
        return elements

    pages: OrderedDict[int, list[tuple[int, Element]]] = OrderedDict()
    for idx, el in enumerate(elements):
        pages.setdefault(el.page_number, []).append((idx, el))

    ordered: list[Element] = []
    for page_no in sorted(pages.keys()):
        items = pages[page_no]
        if all(el.bbox is not None for _, el in items):
            items = sorted(items, key=lambda pair: (round(pair[1].bbox.y0, 1), pair[0]))
        ordered.extend(el for _, el in items)

    for i, el in enumerate(ordered):
        el.order_index = i
    return ordered
