"""Phase 7: structural-fidelity — trees rebuild correctly across (messy) inputs."""

from __future__ import annotations

from rag_pipeline.eval.synth import make_document
from rag_pipeline.ingestion.elements import Element, ElementType, FontInfo, ParsedDocument
from rag_pipeline.structure.tree_builder import build_tree, serialize_tree


def _by_text(nodes):
    return {n.text: n for n in nodes}


def _plain(*texts):
    els = [
        Element(text=t, element_type=ElementType.PARAGRAPH, page_number=1, font=FontInfo(size=11))
        for t in texts
    ]
    return ParsedDocument(source_path="d.pdf", title=texts[0], elements=els, page_count=1)


def test_font_hierarchy_rebuilds_correctly():
    doc = make_document(
        "d.pdf",
        "Manual",
        [("Chapter 1", ["alpha body one", "alpha body two"]), ("Chapter 2", ["beta body"])],
    )
    nodes = build_tree(doc)
    by = _by_text(nodes)
    assert (nodes[0].node_type, nodes[0].depth, nodes[0].text) == ("title", 0, "Manual")
    assert (by["Chapter 1"].node_type, by["Chapter 1"].depth) == ("heading", 1)
    assert by["Chapter 1"].parent is nodes[0]
    assert by["alpha body one"].depth == 2 and by["alpha body one"].parent is by["Chapter 1"]
    assert by["Chapter 2"].parent is nodes[0]


def test_tree_invariants_hold():
    doc = make_document(
        "d.pdf", "Doc", [("Section A", ["a1", "a2", "a3"]), ("Section B", ["b1"])]
    )
    rows = serialize_tree(build_tree(doc))
    assert sum(1 for r in rows if r["parent"] is None) == 1  # exactly one root
    for r in rows:
        if r["parent"] is None:
            assert r["depth"] == 0
        else:
            parent = rows[r["parent"]]
            assert r["depth"] == parent["depth"] + 1   # depth is parent + 1
            assert r["parent"] < r["id"]               # parents precede children


def test_numbering_detected_without_font_cues():
    # Everything is body-sized; numbering alone must reveal the headings.
    doc = _plain(
        "Plain Cover",
        "1. Introduction",
        "the introduction body text",
        "2. Methods",
        "the methods body text",
    )
    nodes = build_tree(doc)
    by = _by_text(nodes)
    assert by["1. Introduction"].node_type in ("heading", "subheading", "article")
    assert by["the introduction body text"].parent is by["1. Introduction"]
    assert doc.metadata["tree_mode"] == "numbering"


def test_low_confidence_degrades_to_flat():
    doc = _plain("just one line of prose", "another line of prose", "a third line")
    nodes = build_tree(doc)
    assert doc.metadata["tree_mode"] == "flat"
    assert all(n.depth == 1 for n in nodes[1:])  # everything hangs off the title


def test_deep_nesting_preserved():
    # title(24) > section(18) > subsection(14) > body(11)
    els = [
        Element("Spec", ElementType.PARAGRAPH, 1, font=FontInfo(size=24, bold=True)),
        Element("Part I", ElementType.PARAGRAPH, 1, font=FontInfo(size=18, bold=True)),
        Element("Chapter A", ElementType.PARAGRAPH, 1, font=FontInfo(size=14, bold=True)),
        Element("the clause body", ElementType.PARAGRAPH, 1, font=FontInfo(size=11)),
    ]
    doc = ParsedDocument(source_path="d.pdf", title="Spec", elements=els, page_count=1)
    by = _by_text(build_tree(doc))
    assert by["Part I"].depth == 1
    assert by["Chapter A"].depth == 2 and by["Chapter A"].parent is by["Part I"]
    assert by["the clause body"].depth == 3 and by["the clause body"].parent is by["Chapter A"]
