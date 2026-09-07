"""Export a structural tree to formats you can paste into Obsidian / Notion.

- ``to_markdown_list``      nested bullets  -> Obsidian & Notion (import as bullets)
- ``to_markdown_headings``  '#'-headings    -> Obsidian outline panel + folding
- ``to_mermaid_mindmap``    a ```mermaid``` mindmap -> Obsidian renders it as a diagram

Nodes come from ``build_tree`` (root first, reading/pre-order), so indentation by
``depth`` reproduces the hierarchy.
"""

from __future__ import annotations

import os
import re

from rag_pipeline.structure.tree_builder import Node


def _visible(nodes: list[Node], headings_only: bool, max_depth: int | None) -> list[Node]:
    out = []
    for n in nodes:
        if headings_only and not n.is_heading:
            continue
        if max_depth is not None and n.depth > max_depth:
            continue
        out.append(n)
    return out


def _clean(text: str) -> str:
    return " ".join(text.split())


def to_markdown_list(
    nodes: list[Node], headings_only: bool = True, max_depth: int | None = None
) -> str:
    visible = _visible(nodes, headings_only, max_depth)
    return "\n".join(f"{'  ' * n.depth}- {_clean(n.text)}" for n in visible)


def to_markdown_headings(
    nodes: list[Node], headings_only: bool = True, max_depth: int | None = None
) -> str:
    lines: list[str] = []
    for n in nodes:
        if max_depth is not None and n.depth > max_depth:
            continue
        if n.is_heading:
            lines += ["", f"{'#' * min(n.depth + 1, 6)} {_clean(n.text)}"]
        elif not headings_only:
            lines.append(_clean(n.text))
    return "\n".join(lines).strip()


_MERMAID_BAD = re.compile(r'[()\[\]{}"#;|]')


def _mermaid_label(text: str) -> str:
    label = _MERMAID_BAD.sub("", _clean(text)) or "node"
    return f"{label[:57]}…" if len(label) > 58 else label


def to_mermaid_mindmap(
    nodes: list[Node], headings_only: bool = True, max_depth: int | None = None
) -> str:
    out = ["```mermaid", "mindmap"]
    for n in _visible(nodes, headings_only, max_depth):
        indent = "  " * (n.depth + 1)
        label = _mermaid_label(n.text)
        out.append(f"{indent}root(({label}))" if n.depth == 0 else f"{indent}{label}")
    out.append("```")
    return "\n".join(out)


def _flow_label(text: str) -> str:
    label = _clean(text).replace('"', "'").replace("#", "")
    return f"{label[:57]}…" if len(label) > 58 else label


def to_mermaid_flowchart(
    nodes: list[Node], headings_only: bool = True, max_depth: int | None = None
) -> str:
    """A ```mermaid graph TD``` — robust across Mermaid versions (unlike mindmap)."""
    visible = _visible(nodes, headings_only, max_depth)
    id_of = {id(n): f"n{i}" for i, n in enumerate(visible)}
    out = ["```mermaid", "graph TD"]
    for n in visible:
        out.append(f'  {id_of[id(n)]}["{_flow_label(n.text)}"]')
    for n in visible:
        if n.parent is not None and id(n.parent) in id_of:
            out.append(f"  {id_of[id(n.parent)]} --> {id_of[id(n)]}")
    out.append("```")
    return "\n".join(out)


FORMATS = {
    "list": to_markdown_list,
    "headings": to_markdown_headings,
    "mermaid": to_mermaid_mindmap,
    "graph": to_mermaid_flowchart,
}


# --- Obsidian vault (one note per node, [[wikilinks]] -> Graph View) ---------

# Characters not allowed in Obsidian note filenames / that break wikilinks.
_ILLEGAL_FILENAME = re.compile(r'[\\/:*?"<>|#^\[\]]+')


def _safe_filename(text: str) -> str:
    name = " ".join(_ILLEGAL_FILENAME.sub(" ", text).split()).rstrip(". ")
    return name[:80].rstrip(". ") or "node"


def _unique_names(nodes: list[Node]) -> dict[int, str]:
    """Assign each node a unique, filename-safe note name (dedupes with ' (2)')."""
    names: dict[int, str] = {}
    used: set[str] = set()
    for n in nodes:
        base = _safe_filename(n.text)
        name = base
        i = 2
        while name.lower() in used:
            name = f"{base} ({i})"
            i += 1
        used.add(name.lower())
        names[id(n)] = name
    return names


def _vault_note(node: Node, parent_name: str | None, children_names: list[str]) -> str:
    front = ["---", f"node_type: {node.node_type}", f"depth: {node.depth}",
             f"order_index: {node.order_index}"]
    if node.confidence is not None:
        front.append(f"confidence: {node.confidence}")
    front += ["tags: [rag-tree]", "---", "", f"# {_clean(node.text)}", ""]
    if parent_name:
        front += [f"**Parent:** [[{parent_name}]]", ""]
    if children_names:
        front.append(f"**Sections ({len(children_names)}):**")
        front += [f"- [[{c}]]" for c in children_names]
        front.append("")
    return "\n".join(front)


def write_obsidian_vault(
    nodes: list[Node],
    out_dir: str,
    *,
    headings_only: bool = True,
    max_depth: int | None = None,
) -> int:
    """Write one .md note per node into ``out_dir``; each links to its parent.

    Open the folder as an Obsidian vault (or move it into one) and the Graph View
    renders the tree. Returns the number of notes written.
    """
    visible = _visible(nodes, headings_only, max_depth)
    names = _unique_names(visible)
    os.makedirs(out_dir, exist_ok=True)
    for n in visible:
        parent_name = names.get(id(n.parent)) if n.parent is not None else None
        children = [names[id(c)] for c in n.children if id(c) in names]
        with open(os.path.join(out_dir, names[id(n)] + ".md"), "w", encoding="utf-8") as fh:
            fh.write(_vault_note(n, parent_name, children))
    return len(visible)


def export_tree(
    nodes: list[Node],
    fmt: str,
    *,
    headings_only: bool = True,
    max_depth: int | None = None,
) -> str:
    return FORMATS[fmt](nodes, headings_only=headings_only, max_depth=max_depth)
