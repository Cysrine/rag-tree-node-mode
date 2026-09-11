"""Phase 7: orphan detection — no clause loses its path; no heading stands alone."""

from __future__ import annotations

import pytest

from rag_pipeline.chunking.chunker import chunk_document
from rag_pipeline.config import ParseStrategy, Settings
from rag_pipeline.eval.synth import make_document, make_pdf
from rag_pipeline.structure.tree_builder import Node, build_tree

HEADING_TYPES = {"title", "heading", "subheading", "article"}


def _settings() -> Settings:
    # min=1 keeps each clause its own chunk (no grouping) for sharper assertions.
    return Settings(_env_file=None, chunk_min_tokens=1, chunk_max_tokens=1000)


def _chunk(doc):
    nodes = build_tree(doc)
    chunks = chunk_document(nodes, settings=_settings())
    return nodes, chunks


def test_no_chunk_is_anchored_on_a_heading():
    doc = make_document(
        "d.pdf",
        "Doc",
        [("Section A", ["clause one here", "clause two here"]), ("Section B", ["clause three"])],
    )
    nodes, chunks = _chunk(doc)
    node_type = {n.id: n.node_type for n in nodes}
    assert chunks
    for c in chunks:
        assert node_type.get(c.node_id) not in HEADING_TYPES


def test_every_chunk_keeps_its_ancestor_path():
    doc = make_document(
        "d.pdf", "Doc", [("Section A", ["alpha clause"]), ("Section B", ["beta clause"])]
    )
    _, chunks = _chunk(doc)
    assert chunks
    for c in chunks:
        assert c.ancestor_path and c.ancestor_path[0] == "Doc"
        assert c.text.strip()


def test_empty_heading_is_never_emitted_as_a_chunk():
    root = Node("title", 0, 0, "Doc")
    empty = Node("heading", 1, 0, "Empty Section", parent=root)
    filled = Node("heading", 1, 1, "Filled Section", parent=root)
    leaf = Node("paragraph", 2, 0, "the only clause", parent=filled)
    root.children.extend([empty, filled])
    filled.children.append(leaf)
    nodes = [root, empty, filled, leaf]

    chunks = chunk_document(nodes, settings=_settings())
    texts = [c.text for c in chunks]
    assert "Empty Section" not in texts and "Filled Section" not in texts
    survivor = next(c for c in chunks if "the only clause" in c.text)
    assert "Filled Section" in survivor.ancestor_path  # heading rides along as context


def test_orphans_on_a_real_generated_pdf(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    from rag_pipeline.ingestion.pdf_loader import load_pdf

    path = make_pdf(
        tmp_path / "m.pdf",
        "Report",
        [
            ("Overview", ["The system preserves document hierarchy end to end."]),
            ("Details", ["Chunks always carry their ancestor path."]),
        ],
    )
    doc = load_pdf(path, settings=Settings(_env_file=None, parse_strategy=ParseStrategy.fallback))
    nodes = build_tree(doc)
    chunks = chunk_document(nodes, settings=_settings())
    node_type = {n.id: n.node_type for n in nodes}

    assert chunks
    for c in chunks:
        assert node_type.get(c.node_id) not in HEADING_TYPES
        assert c.ancestor_path  # no clause is orphaned from its headings
