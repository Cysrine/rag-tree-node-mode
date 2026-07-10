"""Phase 2 tree-assembly tests (font tiers, numbering, tags, flat fallback)."""

from __future__ import annotations

from rag_pipeline.ingestion.elements import Element, ElementType, FontInfo, ParsedDocument
from rag_pipeline.structure.tree_builder import build_tree, serialize_tree


def _el(text, size=None, bold=False, etype=ElementType.PARAGRAPH, page=1, cat=None):
    font = FontInfo(size=size, bold=bold) if size is not None else None
    return Element(text=text, element_type=etype, page_number=page, font=font, category_depth=cat)


def _doc(elements, title="doc", pages=1):
    # source_path points at a nonexistent file so no PDF outline is loaded.
    return ParsedDocument(
        source_path="nonexistent.pdf", title=title, elements=elements, page_count=pages
    )


def _by_text(nodes):
    return {n.text: n for n in nodes}


def test_font_tiers_produce_nested_depths():
    els = [
        _el("Computer Security Fundamentals", 30, bold=True),
        _el("Chapter One", 18, bold=True),
        _el("Intro paragraph body text here.", 12),
        _el("1.1 Scope", 12),                      # numbered -> promoted subheading
        _el("Details of the scope go here.", 12),
        _el("Chapter Two", 18, bold=True),
        _el("Second chapter body.", 12),
    ]
    nodes = build_tree(_doc(els))
    by = _by_text(nodes)
    root = nodes[0]

    assert root.node_type == "title" and root.depth == 0
    assert root.text == "Computer Security Fundamentals"

    c1 = by["Chapter One"]
    assert (c1.node_type, c1.depth) == ("heading", 1) and c1.parent is root

    body1 = by["Intro paragraph body text here."]
    assert (body1.node_type, body1.depth) == ("paragraph", 2) and body1.parent is c1

    sub = by["1.1 Scope"]
    assert (sub.node_type, sub.depth) == ("subheading", 2) and sub.parent is c1

    det = by["Details of the scope go here."]
    assert det.depth == 3 and det.parent is sub

    c2 = by["Chapter Two"]
    assert c2.depth == 1 and c2.parent is root


def test_mode_is_fonts_and_confidence_present():
    els = [_el("Title Here", 24, bold=True), _el("Heading", 16, bold=True), _el("body text", 11)]
    doc = _doc(els)
    nodes = build_tree(doc)
    assert doc.metadata["tree_mode"] == "fonts"
    assert doc.metadata["tree_headings"] == 1
    assert all(n.confidence is not None for n in nodes)


def test_flat_fallback_when_no_structure():
    els = [_el("alpha beta gamma", 12), _el("delta epsilon zeta", 12)]
    doc = _doc(els, title="MyDoc")
    nodes = build_tree(doc)
    assert nodes[0].node_type == "title" and nodes[0].text == "MyDoc"
    assert all(n.depth == 1 and n.node_type == "paragraph" for n in nodes[1:])
    assert doc.metadata["tree_mode"] == "flat"
    assert any("flat" in w.lower() for w in doc.warnings)


def test_tag_mode_for_unstructured_style_input():
    # No fonts; hierarchy comes from Title tags + category_depth (unstructured path).
    els = [
        _el("Introduction", etype=ElementType.TITLE, cat=0),
        _el("Some body text.", etype=ElementType.PARAGRAPH),
        _el("Methods", etype=ElementType.TITLE, cat=0),
    ]
    doc = _doc(els, title="Paper")
    nodes = build_tree(doc)
    by = _by_text(nodes)
    assert doc.metadata["tree_mode"] == "tags"
    intro = by["Introduction"]
    assert intro.node_type == "heading" and intro.depth == 1
    assert by["Some body text."].parent is intro and by["Some body text."].depth == 2
    assert by["Methods"].depth == 1


def test_running_headers_dropped():
    els = []
    for p in range(1, 11):
        els.append(_el("Running Header", 12, page=p))
        els.append(_el(f"Unique body content for page {p}", 12, page=p))
    nodes = build_tree(_doc(els, pages=10))
    assert "Running Header" not in [n.text for n in nodes]


def test_list_items_not_promoted_to_headings():
    els = [
        _el("Document Title", 22, bold=True),
        _el("Overview", 16, bold=True),
        _el("1. first list item here", 11, etype=ElementType.LIST_ITEM),
        _el("2. second list item here", 11, etype=ElementType.LIST_ITEM),
    ]
    nodes = build_tree(_doc(els))
    by = _by_text(nodes)
    assert by["1. first list item here"].node_type == "list_item"
    assert by["1. first list item here"].depth == 2  # leaf under "Overview"


def test_serialize_tree_parent_indices():
    els = [_el("T", 20, bold=True), _el("H", 14, bold=True), _el("body", 10)]
    nodes = build_tree(_doc(els))
    rows = serialize_tree(nodes)
    assert rows[0]["parent"] is None and rows[0]["depth"] == 0
    # every non-root row points at a valid earlier node
    for row in rows[1:]:
        assert row["parent"] is not None and 0 <= row["parent"] < len(rows)
