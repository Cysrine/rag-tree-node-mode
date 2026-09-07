"""Tree export formats for Obsidian / Notion."""

from __future__ import annotations

import os

from rag_pipeline.structure.export import (
    to_markdown_headings,
    to_markdown_list,
    to_mermaid_flowchart,
    to_mermaid_mindmap,
    write_obsidian_vault,
)
from rag_pipeline.structure.tree_builder import Node


def _tree():
    root = Node("title", 0, 0, "Doc Title")
    h1 = Node("heading", 1, 0, "Section One", parent=root)
    sub = Node("subheading", 2, 0, "Sub A", parent=h1)
    body = Node("paragraph", 3, 0, "the clause body text", parent=sub)
    h2 = Node("heading", 1, 1, "Section Two (special) chars", parent=root)
    root.children = [h1, h2]
    h1.children = [sub]
    sub.children = [body]
    return [root, h1, sub, body, h2]


def test_markdown_list_headings_only_nests_by_depth():
    out = to_markdown_list(_tree(), headings_only=True)
    lines = out.splitlines()
    assert lines[0] == "- Doc Title"
    assert "  - Section One" in lines
    assert "    - Sub A" in lines
    assert "  - Section Two (special) chars" in lines
    assert "the clause body text" not in out  # body excluded when headings_only


def test_markdown_list_all_includes_body():
    out = to_markdown_list(_tree(), headings_only=False)
    assert "      - the clause body text" in out  # depth 3 -> 6 spaces


def test_markdown_headings_map_depth_to_hashes():
    out = to_markdown_headings(_tree())
    assert "# Doc Title" in out
    assert "## Section One" in out
    assert "### Sub A" in out


def test_mermaid_mindmap_is_wellformed_and_sanitized():
    out = to_mermaid_mindmap(_tree())
    assert out.startswith("```mermaid\nmindmap")
    assert "root((Doc Title))" in out
    # parentheses stripped from labels so mindmap doesn't choke
    assert "Section Two special chars" in out
    assert "(special)" not in out


def test_mermaid_flowchart_has_nodes_and_edges():
    out = to_mermaid_flowchart(_tree())
    assert out.startswith("```mermaid\ngraph TD")
    assert '["Doc Title"]' in out
    assert "-->" in out                      # parent -> child edges exist
    assert out.count("-->") == 3             # h1, sub, h2 each edge to a parent (body excluded)


def test_max_depth_limits_export():
    out = to_markdown_list(_tree(), headings_only=True, max_depth=1)
    assert "Section One" in out
    assert "Sub A" not in out  # depth 2 excluded


def test_obsidian_vault_writes_linked_notes(tmp_path):
    root = Node("title", 0, 0, "My Doc")
    a = Node("heading", 1, 0, "Introduction", parent=root)
    b = Node("heading", 1, 1, "Section: A/B?", parent=root)   # illegal filename chars
    c = Node("subheading", 2, 0, "Introduction", parent=b)    # duplicate name
    root.children = [a, b]
    b.children = [c]
    nodes = [root, a, b, c]

    count = write_obsidian_vault(nodes, str(tmp_path), headings_only=True)
    files = set(os.listdir(tmp_path))
    assert count == 4 and len(files) == 4

    # Illegal characters are stripped from the filename.
    assert "Section A B.md" in files
    # Duplicate "Introduction" is disambiguated.
    assert "Introduction.md" in files and "Introduction (2).md" in files

    root_note = (tmp_path / "My Doc.md").read_text(encoding="utf-8")
    assert "[[Introduction]]" in root_note and "[[Section A B]]" in root_note  # child links

    intro = (tmp_path / "Introduction.md").read_text(encoding="utf-8")
    assert "**Parent:** [[My Doc]]" in intro   # links back up -> tree edge


