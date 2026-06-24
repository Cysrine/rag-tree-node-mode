"""Phase 2: per-node structure-detection confidence (0..1).

Low document confidence is the signal to prefer the flat-but-ordered fallback in
``tree_builder`` over a speculative hierarchy.
"""

from __future__ import annotations

from rag_pipeline.ingestion.elements import Element
from rag_pipeline.structure.depth_heuristics import FontModel


def heading_confidence(
    element: Element, mode: str, font_model: FontModel | None, has_numbering: bool
) -> float:
    if mode == "outline":
        return 0.95
    score = 0.55
    if (
        font_model is not None
        and element.font
        and element.font.size
        and round(element.font.size) > font_model.body_size
    ):
        score = 0.8
    if element.font and element.font.bold:
        score += 0.1
    if has_numbering:
        score += 0.1
    return round(min(score, 0.98), 2)


def body_confidence(element: Element, font_model: FontModel | None, flat: bool) -> float:
    if flat:
        return 0.3
    if (
        font_model is not None
        and element.font
        and element.font.size
        and round(element.font.size) == font_model.body_size
    ):
        return 0.9
    return 0.75


def document_confidence(node_count: int, heading_count: int, mode: str) -> float:
    """Rough whole-document score: did we find a believable amount of structure?"""
    if mode == "flat" or node_count <= 1:
        return 0.3
    if mode == "outline":
        return 0.95
    ratio = heading_count / node_count
    # Healthy docs are mostly leaves with a sensible sprinkling of headings.
    structured = 0.6 if 0.005 <= ratio <= 0.5 else 0.4
    return round(structured, 2)
