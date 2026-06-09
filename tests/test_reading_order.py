"""Reading-order normalization (Phase 1.5) — pure unit tests, no PDF needed."""

from __future__ import annotations

from rag_pipeline.ingestion import reading_order
from rag_pipeline.ingestion.elements import BBox, Element, ElementType


def _el(text: str, x0: float, top: float, page: int = 1) -> Element:
    return Element(
        text=text,
        element_type=ElementType.PARAGRAPH,
        page_number=page,
        bbox=BBox(x0, top, x0 + 100, top + 12, page_width=600, page_height=800),
    )


def test_two_columns_ordered_left_then_right():
    els = [_el("R1", 350, 100), _el("L1", 60, 100), _el("L2", 60, 140), _el("R2", 350, 140)]
    out = reading_order.normalize(els)
    assert [e.text for e in out] == ["L1", "L2", "R1", "R2"]
    assert [e.order_index for e in out] == [0, 1, 2, 3]


def test_single_column_top_to_bottom():
    els = [_el("b", 72, 200), _el("a", 72, 100), _el("c", 72, 300)]
    out = reading_order.normalize(els)
    assert [e.text for e in out] == ["a", "b", "c"]


def test_pages_are_ordered():
    els = [_el("p2", 72, 50, page=2), _el("p1", 72, 50, page=1)]
    out = reading_order.normalize(els)
    assert [e.text for e in out] == ["p1", "p2"]


def test_missing_geometry_keeps_input_order():
    a = Element("first", ElementType.PARAGRAPH, page_number=1)   # no bbox
    b = Element("second", ElementType.PARAGRAPH, page_number=1)
    out = reading_order.normalize([a, b])
    assert [e.text for e in out] == ["first", "second"]
    assert [e.order_index for e in out] == [0, 1]
