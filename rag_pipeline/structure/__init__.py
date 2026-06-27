"""Phase 2: structural tree assembly (depth + parent + order + confidence).

    from rag_pipeline.structure import build_tree
    nodes = build_tree(parsed_doc)   # root (title) first

Signal priority: PDF bookmarks > numbering patterns > font clustering > position,
with a flat-but-ordered fallback when structural confidence is low.
"""

from __future__ import annotations

from rag_pipeline.structure.tree_builder import Node, build_tree, serialize_tree

__all__ = ["Node", "build_tree", "serialize_tree"]
