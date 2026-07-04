"""Tree-builder validation CLI.

    rag-tree file.pdf                 # summary + heading tree (table of contents)
    rag-tree file.pdf --all           # include body leaves
    rag-tree file.pdf --json tree.json

Lets you eyeball whether depths/parents/node-types came out right on real docs.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_pipeline.config import ParseStrategy, get_settings
from rag_pipeline.ingestion.pdf_loader import load_pdf
from rag_pipeline.logging_setup import enable_utf8_stdout
from rag_pipeline.structure.tree_builder import Node, build_tree, serialize_tree


def _preview(text: str, width: int) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


def _print_summary(doc, nodes: list[Node]) -> None:
    types: dict[str, int] = {}
    depths: dict[int, int] = {}
    for n in nodes:
        types[n.node_type] = types.get(n.node_type, 0) + 1
        depths[n.depth] = depths.get(n.depth, 0) + 1

    print(f"\nTree for: {doc.source_path}")
    print(f"  parse strategy:  {doc.strategy_used}")
    print(f"  tree mode:       {doc.metadata.get('tree_mode')}  (signal that won)")
    print(f"  tree confidence: {doc.metadata.get('tree_confidence')}")
    print(f"  nodes:           {len(nodes)}  (headings: {doc.metadata.get('tree_headings')})")
    print("  node types:      " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    print("  depth histogram: " + ", ".join(f"d{d}={depths[d]}" for d in sorted(depths)))
    for w in doc.warnings:
        print(f"  ! {w}")


def _print_tree(nodes: list[Node], show_all: bool, limit: int) -> None:
    shown = [n for n in nodes if show_all or n.is_heading]
    total = len(shown)
    if limit >= 0:
        shown = shown[:limit]

    header = "headings" if not show_all else "all nodes"
    print(f"\n  structural tree ({header}):")
    print("  " + "-" * 78)
    for n in shown:
        indent = "  " * n.depth
        conf = f"{n.confidence:.2f}" if n.confidence is not None else "  - "
        label = f"[{n.node_type}]"
        print(f"  {conf}  {indent}{label} {_preview(n.text, 74 - len(indent))}")
    if limit >= 0 and total > limit:
        print(f"  … {total - limit} more (use --max -1 to show all)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-tree", description="Parse a PDF and build its tree.")
    p.add_argument("pdf", help="path to a PDF file")
    p.add_argument("--strategy", choices=[s.value for s in ParseStrategy])
    p.add_argument("--all", action="store_true", help="include body leaves (default: headings)")
    p.add_argument("--max", type=int, default=120, help="max nodes to print (-1 = all)")
    p.add_argument("--json", dest="json_out", metavar="PATH", help="dump the full tree as JSON")
    p.add_argument(
        "--format",
        choices=["text", "list", "headings", "mermaid", "graph", "obsidian-vault"],
        default="text",
        help="text | list/headings (Obsidian & Notion) | mermaid (mindmap, needs Mermaid>=9.3) "
        "| graph (flowchart) | obsidian-vault (one note/heading -> Graph View; needs --out DIR)",
    )
    p.add_argument("--max-depth", type=int, default=None, help="only export nodes up to this depth")
    p.add_argument(
        "--out", metavar="PATH", help="output file (or DIR for obsidian-vault) instead of stdout"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    settings = get_settings()
    if args.strategy:
        settings = settings.model_copy(update={"parse_strategy": ParseStrategy(args.strategy)})

    try:
        doc = load_pdf(args.pdf, settings=settings)
    except FileNotFoundError:
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return 2

    nodes = build_tree(doc)

    if args.format == "text":
        _print_summary(doc, nodes)
        _print_tree(nodes, show_all=args.all, limit=args.max)
    elif args.format == "obsidian-vault":
        if not args.out:
            print("error: --format obsidian-vault requires --out DIR (a folder)", file=sys.stderr)
            return 2
        from rag_pipeline.structure.export import write_obsidian_vault

        try:
            count = write_obsidian_vault(
                nodes, args.out, headings_only=not args.all, max_depth=args.max_depth
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {count} notes -> {args.out}")
        print("Open that folder as an Obsidian vault (or move it into one), then open Graph View.")
    else:
        from rag_pipeline.structure.export import export_tree

        content = export_tree(
            nodes, args.format, headings_only=not args.all, max_depth=args.max_depth
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(content + "\n")
            print(f"Wrote {args.format} export -> {args.out}  ({len(content.splitlines())} lines)")
        else:
            print(content)

    if args.json_out:
        payload = {
            "title": nodes[0].text if nodes else None,
            "metadata": doc.metadata,
            "nodes": serialize_tree(nodes),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote tree JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
