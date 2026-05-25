"""Parser validation CLI.

    rag-parse path/to/file.pdf
    python -m rag_pipeline.ingestion.cli file.pdf --strategy hi_res --json out.json

Prints the parse summary + a structural listing so you can eyeball whether
titles/headings/tables came out right — the spec's "validate on real documents
before building downstream" step.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_pipeline.config import ParseStrategy, get_settings
from rag_pipeline.ingestion.elements import ParsedDocument
from rag_pipeline.ingestion.pdf_loader import load_pdf
from rag_pipeline.logging_setup import enable_utf8_stdout


def _print_summary(doc: ParsedDocument) -> None:
    print(f"\nParsed: {doc.source_path}")
    print(f"  title:         {doc.title!r}")
    print(f"  pages:         {doc.page_count}")
    print(f"  strategy:      {doc.strategy_used}")
    avg = doc.metadata.get("avg_chars_per_page")
    scanned_note = f" (avg {avg} chars/page)" if avg is not None else ""
    print(f"  scanned:       {doc.is_scanned}{scanned_note}")
    print(f"  used OCR:      {doc.used_ocr}")
    print(f"  used fallback: {doc.used_fallback}")
    print(f"  elements:      {len(doc.elements)}")

    counts = doc.type_counts()
    if counts:
        joined = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  type counts:   {joined}")

    if doc.warnings:
        print("  warnings:")
        for w in doc.warnings:
            print(f"    - {w}")


def _print_elements(doc: ParsedDocument, limit: int, full: bool) -> None:
    print("\n  #   pg  type          font    text")
    print("  " + "-" * 78)
    shown = doc.elements[:limit] if limit >= 0 else doc.elements
    for el in shown:
        size = f"{el.font.size:>4.0f}pt" if el.font and el.font.size else "  -  "
        depth = el.category_depth or 0
        indent = "  " * depth
        text = " ".join(el.text.split()) if full else el.preview(80 - len(indent))
        prefix = f"  {el.order_index:>3} p{el.page_number:<2} {el.element_type.value:<12} {size}"
        print(f"{prefix}  {indent}{text}")
    if limit >= 0 and len(doc.elements) > limit:
        print(f"  … {len(doc.elements) - limit} more (use --max -1 to show all)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-parse", description="Parse a PDF and show its structure.")
    p.add_argument("pdf", help="path to a PDF file")
    p.add_argument(
        "--strategy",
        choices=[s.value for s in ParseStrategy],
        help="override RAG_PARSE_STRATEGY for this run",
    )
    p.add_argument("--json", dest="json_out", metavar="PATH", help="also dump full result as JSON")
    p.add_argument("--max", type=int, default=60, help="max elements to print (-1 = all)")
    p.add_argument("--full", action="store_true", help="print full element text (no truncation)")
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

    _print_summary(doc)
    _print_elements(doc, limit=args.max, full=args.full)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(doc.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
