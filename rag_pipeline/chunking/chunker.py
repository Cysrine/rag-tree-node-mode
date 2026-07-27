"""Phase 3: hierarchy-preserving, leaf-anchored chunker with orphan guard.

Turns a Phase-2 tree into embeddable chunks:
  - Chunks anchor at leaf (body) nodes; headings are NEVER emitted as their own
    chunk — they ride along as ``ancestor_path`` context on every descendant
    chunk. That is the orphan guard, satisfied structurally.
  - Tiny leaves are grouped with their siblings (same parent) up to the token
    budget. This is what reconstructs paragraphs/sections from the line-level
    output of the pdfplumber fallback.
  - Large leaves are split on sentence boundaries (never mid-clause); every
    sub-chunk replicates the ancestor path and source node.
  - Tables are kept whole (never split, never grouped).

``embed_input`` = ``"Title > Section > Subsection: <clause>"`` — the path-prefixed
text that gets embedded, so the vector carries hierarchical context.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.structure.tree_builder import Node

TokenCounter = Callable[[str], int]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Rough, dependency-free token estimate (~4 chars/token). Swappable per call."""
    text = text.strip()
    if not text:
        return 0
    return max(len(text) // 4, 1)


@dataclass
class Chunk:
    """A leaf-anchored, embeddable chunk (maps to the ``chunks`` table)."""

    node_id: str                 # anchor node (the leaf, or first of a sibling group)
    text: str                    # clause text shown to the LLM
    embed_input: str             # path-prefixed text that gets embedded
    ancestor_path: list[str] = field(default_factory=list)
    token_count: int = 0
    source_node_ids: list[str] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "token_count": self.token_count,
            "ancestor_path": self.ancestor_path,
            "source_node_ids": self.source_node_ids,
            "embed_input": self.embed_input,
            "text": self.text,
        }


# --- helpers --------------------------------------------------------------


def _ensure_node_ids(nodes: list[Node]) -> None:
    """Give every node a stable id if it lacks one (post-persist it's already a UUID)."""
    for i, n in enumerate(nodes):
        if n.id is None:
            n.id = f"n{i}"


def _ancestor_path(node: Node) -> list[str]:
    path: list[str] = []
    parent = node.parent
    while parent is not None:
        path.append(parent.text)
        parent = parent.parent
    path.reverse()
    return path


def _embed_input(path: list[str], text: str) -> str:
    prefix = " > ".join(p for p in path if p)
    return f"{prefix}: {text}" if prefix else text


def _split_large_text(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split an over-budget leaf on sentence boundaries, packing greedily."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sentence in (s for s in _SENTENCE.split(text.strip()) if s):
        st = count(sentence)
        if cur and cur_tokens + st > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sentence)
        cur_tokens += st
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _make_chunk(anchor: Node, group: list[Node], text: str, count: TokenCounter) -> Chunk:
    path = _ancestor_path(anchor)
    return Chunk(
        node_id=anchor.id or "",
        text=text,
        embed_input=_embed_input(path, text),
        ancestor_path=path,
        token_count=count(text),
        source_node_ids=[n.id or "" for n in group],
        depth=anchor.depth,
    )


# --- entrypoint -----------------------------------------------------------


def chunk_document(
    nodes: list[Node],
    settings: Settings | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Produce leaf-anchored chunks from a Phase-2 tree (root first)."""
    settings = settings or get_settings()
    count = count_tokens or estimate_tokens
    max_tokens = settings.chunk_max_tokens
    min_tokens = settings.chunk_min_tokens

    _ensure_node_ids(nodes)
    leaves = [n for n in nodes if not n.is_heading and n.text.strip()]

    chunks: list[Chunk] = []
    buffer: list[Node] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            text = "\n".join(n.text for n in buffer)
            chunks.append(_make_chunk(buffer[0], buffer, text, count))
            buffer, buffer_tokens = [], 0

    for leaf in leaves:
        tokens = count(leaf.text)

        # Tables and over-budget leaves are handled on their own.
        if leaf.node_type == "table" or tokens > max_tokens:
            flush()
            if leaf.node_type == "table":
                chunks.append(_make_chunk(leaf, [leaf], leaf.text, count))
            else:
                for piece in _split_large_text(leaf.text, max_tokens, count):
                    chunks.append(_make_chunk(leaf, [leaf], piece, count))
            continue

        # Normal-sized leaf: stands alone (leaf/clause granularity).
        if tokens >= min_tokens:
            flush()
            chunks.append(_make_chunk(leaf, [leaf], leaf.text, count))
            continue

        # Tiny leaf: group with same-parent siblings up to the budget.
        if buffer and buffer_tokens + tokens > max_tokens:
            flush()
        buffer.append(leaf)
        buffer_tokens += tokens

    flush()
    return chunks
