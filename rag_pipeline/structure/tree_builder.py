"""Phase 2: assemble a structural tree from parsed elements.

Consumes ``doc.elements`` (already reading-ordered by Phase 1.5) and emits a
adjacency-list tree of ``Node``s (root title first). Depth/parent come from a
stack walk over per-element heading *levels*; those levels come from the signal
hierarchy in ``depth_heuristics`` (bookmarks > numbering > fonts). When no
structure is found, it degrades to a flat, reading-ordered tree instead of
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_pipeline.ingestion.elements import Element, ElementType, ParsedDocument
from rag_pipeline.structure.confidence import (
    body_confidence,
    document_confidence,
    heading_confidence,
)
from rag_pipeline.structure.depth_heuristics import (
    FontModel,
    assign_levels,
    detect_running_headers,
    norm,
    numbering,
)
from rag_pipeline.structure.outline import extract_outline

_HEADING_TYPES = {"title", "heading", "subheading", "article"}


@dataclass
class Node:
    """A structural tree node (maps to the ``nodes`` table)."""

    node_type: str          # title | heading | subheading | article | paragraph | list_item | table
    depth: int              # title = 0
    order_index: int        # sibling order under the parent, in reading order
    text: str
    confidence: float | None = None
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    id: str | None = None                       # assigned on persist
    source_element_ids: list[str] = field(default_factory=list)

    @property
    def is_heading(self) -> bool:
        return self.node_type in _HEADING_TYPES


def _eid(element: Element, index: int) -> str:
    return element.element_id or f"el-{index}"


def _choose_title_idx(elements: list[Element], drop: set[int], fm: FontModel | None) -> int | None:
    """The biggest-font, reasonably short line — our best guess at the doc title."""
    if fm is None:
        return None
    best: int | None = None
    best_size = -1.0
    for i, e in enumerate(elements):
        if i in drop or not (e.font and e.font.size) or len(e.text) > 140:
            continue
        if e.font.size > best_size:
            best_size, best = e.font.size, i
    return best


def _heading_type(element: Element, depth: int) -> str:
    num = numbering(element.text)
    if num and num.kind == "article":
        return "article"
    return "heading" if depth <= 1 else "subheading"


def _body_type(element: Element) -> str:
    if element.element_type is ElementType.TABLE:
        return "table"
    if element.element_type is ElementType.LIST_ITEM:
        return "list_item"
    return "paragraph"


def build_tree(doc: ParsedDocument) -> list[Node]:
    """Return the document's nodes, root (title) first."""
    elements = doc.elements
    if not elements:
        return [Node("title", 0, 0, doc.title or "Document", confidence=0.3)]

    outline = extract_outline(doc.source_path)
    levels, mode, fm = assign_levels(elements, outline)
    drop = detect_running_headers(elements, doc.page_count)

    # Pick the title and lift it to the root; drop any exact repeats (page furniture).
    title_idx = _choose_title_idx(elements, drop, fm)
    consume_title = title_idx is not None and levels[title_idx] == 0
    root_text = elements[title_idx].text if consume_title else (doc.title or "Document")
    if consume_title:
        target = norm(root_text)
        drop |= {i for i, e in enumerate(elements) if i != title_idx and norm(e.text) == target}

    root = Node("title", 0, 0, root_text, confidence=0.9 if consume_title else 0.5)
    if consume_title:
        root.source_element_ids = [_eid(elements[title_idx], title_idx)]

    nodes: list[Node] = [root]
    counter = 0
    stack: list[tuple[int, Node]] = [(-1, root)]

    def next_order(parent: Node) -> int:
        nonlocal counter
        counter += 1
        return counter - 1

    heading_count = 0
    for i, e in enumerate(elements):
        if i in drop or (consume_title and i == title_idx):
            continue
        lvl = levels[i]
        if lvl is not None:
            while len(stack) > 1 and stack[-1][0] > lvl:
                stack.pop()
            parent = stack[-1][1]
            depth = parent.depth + 1
            node = Node(
                node_type=_heading_type(e, depth),
                depth=depth,
                order_index=next_order(parent),
                text=e.text,
                confidence=heading_confidence(e, mode, fm, numbering(e.text) is not None),
                parent=parent,
                source_element_ids=[_eid(e, i)],
            )
            stack.append((lvl, node))
            heading_count += 1
        else:
            parent = stack[-1][1]
            depth = parent.depth + 1
            node = Node(
                node_type=_body_type(e),
                depth=depth,
                order_index=next_order(parent),
                text=e.text,
                confidence=body_confidence(e, fm, flat=(mode == "flat")),
                parent=parent,
                source_element_ids=[_eid(e, i)],
            )
        parent.children.append(node)
        nodes.append(node)

    doc.metadata["tree_mode"] = mode
    doc.metadata["tree_headings"] = heading_count
    doc.metadata["tree_confidence"] = document_confidence(len(nodes), heading_count, mode)
    if mode == "flat":
        doc.warnings.append(
            "Low structural confidence: no headings detected — emitted a flat, "
            "reading-ordered tree (all leaves under the title)."
        )
    return nodes


def serialize_tree(nodes: list[Node]) -> list[dict]:
    """Flatten to rows with parent indices (for JSON / DB-style inspection)."""
    index = {id(n): i for i, n in enumerate(nodes)}
    return [
        {
            "id": i,
            "parent": index.get(id(n.parent)) if n.parent is not None else None,
            "depth": n.depth,
            "node_type": n.node_type,
            "order_index": n.order_index,
            "confidence": n.confidence,
            "text": n.text,
        }
        for i, n in enumerate(nodes)
    ]
