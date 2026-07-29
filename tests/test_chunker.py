"""Phase 3 chunker tests: orphan guard, tiny-leaf grouping, large-leaf splitting."""

from __future__ import annotations

from rag_pipeline.chunking.chunker import chunk_document
from rag_pipeline.config import Settings
from rag_pipeline.structure.tree_builder import Node

# Deterministic tokenizer for tests: 1 token per whitespace word.
WORDS = lambda s: len(s.split())  # noqa: E731


def _settings(max_t=6, min_t=3):
    return Settings(_env_file=None, chunk_max_tokens=max_t, chunk_min_tokens=min_t)


def _tree(*leaves: Node, title="Doc Title", heading="Section A"):
    root = Node("title", 0, 0, title)
    head = Node("heading", 1, 0, heading, parent=root)
    root.children.append(head)
    for i, leaf in enumerate(leaves):
        leaf.parent = head
        leaf.order_index = i
        head.children.append(leaf)
    return [root, head, *leaves]


def _leaf(text, node_type="paragraph"):
    return Node(node_type, 2, 0, text)


def test_tiny_siblings_are_grouped():
    nodes = _tree(*[_leaf(f"word{i}") for i in range(5)])  # 5 leaves, 1 token each
    chunks = chunk_document(nodes, settings=_settings(max_t=6, min_t=3), count_tokens=WORDS)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.ancestor_path == ["Doc Title", "Section A"]
    assert all(f"word{i}" in c.text for i in range(5))
    assert len(c.source_node_ids) == 5


def test_group_flushes_at_budget():
    # 8 tiny leaves, budget 6 -> two chunks (6 + 2).
    nodes = _tree(*[_leaf("x") for _ in range(8)])
    chunks = chunk_document(nodes, settings=_settings(max_t=6, min_t=3), count_tokens=WORDS)
    assert len(chunks) == 2
    assert chunks[0].token_count == 6
    assert chunks[1].token_count == 2


def test_normal_leaf_stands_alone():
    nodes = _tree(_leaf("one two three four"))  # 4 tokens >= min 3
    chunks = chunk_document(nodes, settings=_settings(max_t=6, min_t=3), count_tokens=WORDS)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three four"


def test_large_leaf_split_on_sentences():
    text = "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lam mu."
    nodes = _tree(_leaf(text))  # 12 tokens > max 6
    chunks = chunk_document(nodes, settings=_settings(max_t=6, min_t=3), count_tokens=WORDS)
    assert len(chunks) == 3
    assert all(c.token_count <= 6 for c in chunks)
    # All sub-chunks share one source node and identical ancestor path.
    assert len({c.node_id for c in chunks}) == 1
    assert all(c.ancestor_path == ["Doc Title", "Section A"] for c in chunks)
    # No sentence was split mid-clause.
    assert chunks[0].text == "Alpha beta gamma delta."


def test_tables_are_kept_whole():
    table = _leaf("| a | b | c | d | e | f | g |\n| 1 | 2 | 3 | 4 | 5 | 6 | 7 |", node_type="table")
    nodes = _tree(table)  # far over the 6-token budget, but must not split
    chunks = chunk_document(nodes, settings=_settings(max_t=6, min_t=3), count_tokens=WORDS)
    assert len(chunks) == 1
    assert chunks[0].node_id == table.id
    assert "| a | b" in chunks[0].text


def test_embed_input_is_path_prefixed():
    nodes = _tree(_leaf("some clause text here"))
    chunks = chunk_document(nodes, settings=_settings(max_t=20, min_t=1), count_tokens=WORDS)
    assert chunks[0].embed_input == "Doc Title > Section A: some clause text here"


def test_orphan_guard_holds():
    # Every chunk carries its heading path; no chunk IS a heading.
    root = Node("title", 0, 0, "T")
    h1 = Node("heading", 1, 0, "H1", parent=root)
    h2 = Node("subheading", 2, 0, "H1.1", parent=h1)
    body = Node("paragraph", 3, 0, "the clause body text", parent=h2)
    root.children.append(h1)
    h1.children.append(h2)
    h2.children.append(body)
    nodes = [root, h1, h2, body]

    chunks = chunk_document(nodes, settings=_settings(max_t=20, min_t=1), count_tokens=WORDS)
    heading_texts = {"T", "H1", "H1.1"}
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.ancestor_path, "no chunk may be orphaned from its headings"
        assert c.text not in heading_texts, "headings must never be emitted as chunks"
    assert chunks[0].ancestor_path == ["T", "H1", "H1.1"]


