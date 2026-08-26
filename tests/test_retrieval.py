"""Phase 5 retrieval logic: chunkset assembly, ancestor-merge, rendering.

Pure unit tests with a fake repository — no DB or embedder needed.
"""

from __future__ import annotations

from rag_pipeline.retrieval.chunkset import Hit, assemble, merge, render


class _FakeRepo:
    def __init__(self, paths: dict[str, list[dict]]):
        self._paths = paths

    def get_paths(self, node_ids):
        return {nid: self._paths[nid] for nid in node_ids if nid in self._paths}


def _row(id, parent, ntype, depth, order, text):
    return {
        "id": id,
        "parent_id": parent,
        "node_type": ntype,
        "depth": depth,
        "order_index": order,
        "text": text,
    }


# A tiny doc: root -> {Section 1 -> leaf1, Section 2 -> leaf2, leaf3}
_PATHS = {
    "l1": [_row("r", None, "title", 0, 0, "Loan Agreement"),
           _row("a", "r", "heading", 1, 0, "Section 1"),
           _row("l1", "a", "paragraph", 2, 0, "definition clause")],
    "l2": [_row("r", None, "title", 0, 0, "Loan Agreement"),
           _row("b", "r", "heading", 1, 1, "Section 2"),
           _row("l2", "b", "paragraph", 2, 0, "rate clause")],
    "l3": [_row("r", None, "title", 0, 0, "Loan Agreement"),
           _row("b", "r", "heading", 1, 1, "Section 2"),
           _row("l3", "b", "paragraph", 2, 1, "penalty clause")],
}


def _hits():
    return [
        Hit(chunk_id="c2", node_id="l2", score=0.9, clause="the rate is 5%"),
        Hit(chunk_id="c3", node_id="l3", score=0.8, clause="late penalty is 2%"),
        Hit(chunk_id="c1", node_id="l1", score=0.5, clause="terms are defined below"),
    ]


def test_assemble_builds_root_to_leaf_paths():
    chunksets = assemble(_hits(), _FakeRepo(_PATHS))
    assert len(chunksets) == 3
    cs = next(c for c in chunksets if c.node_id == "l2")
    assert [p.text for p in cs.path] == ["Loan Agreement", "Section 2", "rate clause"]
    assert cs.ancestor_texts == ["Loan Agreement", "Section 2"]


def test_merge_collapses_shared_ancestors():
    roots = merge(assemble(_hits(), _FakeRepo(_PATHS)))
    assert len(roots) == 1
    root = roots[0]
    assert root.text == "Loan Agreement" and root.depth == 0
    # Section 1 and Section 2 both hang off the single root.
    assert len(root.children) == 2
    sec2 = next(c for c in root.children.values() if c.text == "Section 2")
    # Both Section 2 clauses merged under one section node (two leaf anchors).
    assert len(sec2.children) == 2
    assert sec2.best_score() == 0.9


def test_render_shows_titles_once_and_orders_by_score():
    roots = merge(assemble(_hits(), _FakeRepo(_PATHS)))
    out = render(roots)
    assert out.count("Loan Agreement") == 1          # title not repeated
    assert out.count("Section 2") == 1
    # Highest-scoring section (Section 2) rendered before Section 1.
    assert out.index("Section 2") < out.index("Section 1")
    assert "the rate is 5%" in out and "late penalty is 2%" in out


def test_empty_hits_yield_nothing():
    assert assemble([], _FakeRepo(_PATHS)) == []
    assert merge([]) == []
    assert render([]) == ""
