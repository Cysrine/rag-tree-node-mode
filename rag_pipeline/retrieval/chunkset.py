"""Phase 5: assemble root-to-leaf chunksets from matched leaves, and merge them.

A vector hit is a leaf clause. Its *chunkset* is that clause plus every heading
above it (root title → section → subsection). When several hits share ancestors,
we merge them into one hierarchy so a title/section is shown once, with its
matched clauses nested underneath — exactly what the LLM should receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    """One vector-search result (a leaf clause)."""

    chunk_id: str
    node_id: str
    score: float
    clause: str
    ancestor_path: list[str] = field(default_factory=list)


@dataclass
class PathNode:
    id: str
    node_type: str
    depth: int
    order_index: int
    text: str


@dataclass
class Chunkset:
    """A matched clause plus its full root-to-anchor node path."""

    chunk_id: str
    node_id: str
    score: float
    clause: str
    path: list[PathNode]  # root first, anchor (leaf) last

    @property
    def ancestor_texts(self) -> list[str]:
        return [p.text for p in self.path[:-1]]


@dataclass
class ClauseHit:
    clause: str
    score: float
    chunk_id: str
    node_id: str


@dataclass
class MergedNode:
    """A node in the merged view; shared ancestors are collapsed to one node."""

    id: str
    node_type: str
    depth: int
    text: str
    order_index: int = 0
    children: dict[str, MergedNode] = field(default_factory=dict)
    hits: list[ClauseHit] = field(default_factory=list)

    def best_score(self) -> float:
        scores = [h.score for h in self.hits]
        scores += [c.best_score() for c in self.children.values()]
        return max(scores) if scores else 0.0


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    chunksets: list[Chunkset]
    roots: list[MergedNode]

    def render(self, max_clause: int = 400) -> str:
        return render(self.roots, max_clause=max_clause)


def assemble(hits: list[Hit], repo) -> list[Chunkset]:
    """Expand each hit's node_id into its root-to-leaf path (one batched query)."""
    if not hits:
        return []
    paths = repo.get_paths([h.node_id for h in hits])
    chunksets: list[Chunkset] = []
    for h in hits:
        rows = paths.get(h.node_id, [])
        path = [
            PathNode(
                id=str(r["id"]),
                node_type=r["node_type"],
                depth=r["depth"],
                order_index=r["order_index"],
                text=r["text"],
            )
            for r in rows
        ]
        chunksets.append(
            Chunkset(
                chunk_id=h.chunk_id,
                node_id=h.node_id,
                score=h.score,
                clause=h.clause,
                path=path,
            )
        )
    return chunksets


def merge(chunksets: list[Chunkset]) -> list[MergedNode]:
    """Merge chunksets into a forest, collapsing shared ancestors. Roots ranked by score."""
    roots: dict[str, MergedNode] = {}
    index: dict[str, MergedNode] = {}
    for cs in chunksets:
        if not cs.path:
            continue
        parent: MergedNode | None = None
        for pn in cs.path:
            node = index.get(pn.id)
            if node is None:
                node = MergedNode(
                    id=pn.id,
                    node_type=pn.node_type,
                    depth=pn.depth,
                    text=pn.text,
                    order_index=pn.order_index,
                )
                index[pn.id] = node
                if parent is None:
                    roots[pn.id] = node
                else:
                    parent.children[pn.id] = node
            parent = node
        # Attach the matched clause to the anchor (deepest node in the path).
        anchor = index[cs.path[-1].id]
        anchor.hits.append(
            ClauseHit(clause=cs.clause, score=cs.score, chunk_id=cs.chunk_id, node_id=cs.node_id)
        )
    return sorted(roots.values(), key=lambda m: m.best_score(), reverse=True)


def _one_line(text: str, width: int) -> str:
    joined = " ".join((text or "").split())
    return joined if len(joined) <= width else joined[: width - 1] + "…"


def render(roots: list[MergedNode], max_clause: int = 400) -> str:
    """A labeled hierarchy: headings once, matched clauses nested with scores."""
    lines: list[str] = []

    def walk(node: MergedNode) -> None:
        indent = "  " * node.depth
        is_leaf = bool(node.hits) and not node.children
        if not is_leaf:
            lines.append(f"{indent}[{node.node_type}] {_one_line(node.text, 100)}")
            for child in sorted(node.children.values(), key=lambda m: m.best_score(), reverse=True):
                walk(child)
        clause_indent = indent if is_leaf else "  " * (node.depth + 1)
        for h in sorted(node.hits, key=lambda x: x.score, reverse=True):
            lines.append(f"{clause_indent}- [{h.score:.2f}] {_one_line(h.clause, max_clause)}")

    for root in roots:
        walk(root)
    return "\n".join(lines)
