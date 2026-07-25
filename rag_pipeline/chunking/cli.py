"""Chunker validation CLI.

    rag-chunk file.pdf                 # stats + sample chunks
    rag-chunk file.pdf --max 20 --json chunks.json

Shows each chunk's ancestor path + text so you can confirm no clause is orphaned
from its headings and chunk sizes look sane.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from rag_pipeline.chunking.chunker import Chunk, chunk_document
from rag_pipeline.config import ParseStrategy, get_settings
from rag_pipeline.ingestion.pdf_loader import load_pdf
from rag_pipeline.logging_setup import enable_utf8_stdout
from rag_pipeline.structure.tree_builder import build_tree


def _preview(text: str, width: int) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


def _print_stats(chunks: list[Chunk]) -> None:
    print(f"\n  chunks:        {len(chunks)}")
    if not chunks:
        return
    toks = [c.token_count for c in chunks]
    grouped = sum(1 for c in chunks if len(c.source_node_ids) > 1)
    split = len(chunks) - len({c.node_id for c in chunks})
    print(f"  tokens/chunk:  min={min(toks)} median={int(statistics.median(toks))} max={max(toks)}")
    print(f"  grouped (tiny siblings merged): {grouped}")
    print(f"  from large-leaf splits:         {split}")
    orphaned = [c for c in chunks if not c.ancestor_path]
    print(f"  orphaned (no ancestor path):    {len(orphaned)}  <- must be 0")


def _print_samples(chunks: list[Chunk], limit: int) -> None:
    print("\n  sample chunks:")
    print("  " + "-" * 78)
    for c in chunks[:limit] if limit >= 0 else chunks:
        path = " > ".join(c.ancestor_path)
        print(f"  [{c.token_count:>3}t] {_preview(path, 74)}")
        print(f"         {_preview(c.text, 72)}")
    if limit >= 0 and len(chunks) > limit:
        print(f"  … {len(chunks) - limit} more (use --max -1 to show all)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-chunk", description="Parse, build tree, and chunk a PDF.")
    p.add_argument("pdf", help="path to a PDF file")
    p.add_argument("--strategy", choices=[s.value for s in ParseStrategy])
    p.add_argument("--max", type=int, default=25, help="max sample chunks to print (-1 = all)")
    p.add_argument("--json", dest="json_out", metavar="PATH", help="dump all chunks as JSON")
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
    chunks = chunk_document(nodes, settings=settings)

    print(f"\nChunks for: {doc.source_path}")
    print(f"  tree mode:     {doc.metadata.get('tree_mode')}  nodes={len(nodes)}")
    _print_stats(chunks)
    _print_samples(chunks, args.max)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in chunks], fh, indent=2, ensure_ascii=False)
        print(f"\nWrote chunks JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
