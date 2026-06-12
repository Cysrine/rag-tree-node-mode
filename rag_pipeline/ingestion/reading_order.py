"""Phase 1.5: reading-order normalization.

Sorts a flat element list into logical reading order and stamps ``order_index``.
Critical for multi-column PDFs, where naive top-to-bottom reading interleaves the
columns. We detect columns from left-edge (x0) clusters, then order:
page -> column (left-to-right) -> top-to-bottom.

Elements lacking geometry (e.g. from unstructured, which already emits reading
order) keep their input order for that page.
"""

from __future__ import annotations

from collections import OrderedDict

from rag_pipeline.ingestion.elements import Element


def _detect_column_edges(elements: list[Element], page_width: float) -> list[float]:
    """Cluster element left-edges into column start positions (ascending)."""
    xs = sorted(e.bbox.x0 for e in elements if e.bbox is not None)
    if not xs:
        return [0.0]
    threshold = max(30.0, 0.12 * page_width)
    edges: list[float] = [xs[0]]
    cluster_last = xs[0]
    for x in xs[1:]:
        if x - cluster_last > threshold:
            edges.append(x)
        cluster_last = x
    return edges


def _assign_column(element: Element, edges: list[float]) -> int:
    if element.bbox is None or not edges:
        return 0
    x0 = element.bbox.x0
    # Nearest column whose left edge best matches this element's left edge.
    return min(range(len(edges)), key=lambda i: abs(x0 - edges[i]))


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
        page_elements = [el for _, el in items]

        if all(el.bbox is not None for el in page_elements):
            page_width = max(el.bbox.x1 for el in page_elements)
            edges = _detect_column_edges(page_elements, page_width)

            def sort_key(pair: tuple[int, Element], edges: list[float] = edges) -> tuple:
                orig_idx, el = pair
                col = _assign_column(el, edges)
                top = round(el.bbox.y0, 1) if el.bbox else 0.0
                left = round(el.bbox.x0, 1) if el.bbox else 0.0
                return (col, top, left, orig_idx)

            page_sorted = sorted(items, key=sort_key)
        else:
            # No geometry: trust the source parser's order (stable).
            page_sorted = items

        ordered.extend(el for _, el in page_sorted)

    for i, el in enumerate(ordered):
        el.order_index = i
    return ordered
