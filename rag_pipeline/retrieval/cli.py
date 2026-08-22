"""Retrieval CLI (Phase 5).

    rag-search "how do I defend against DoS attacks?" -k 8   # merged chunksets
    rag-search "..." --flat                                   # raw ranked chunks

Merged view shows each title/section once with its matched clauses nested — the
root-to-leaf chunksets the LLM will receive in Phase 6.
"""

from __future__ import annotations

import argparse

from rag_pipeline.config import get_settings
from rag_pipeline.logging_setup import enable_utf8_stdout
from rag_pipeline.retrieval.search import retrieve, search


def _one_line(text: str, width: int) -> str:
    joined = " ".join((text or "").split())
    return joined if len(joined) <= width else joined[: width - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    p = argparse.ArgumentParser(prog="rag-search", description="Structure-aware retrieval.")
    p.add_argument("query", help="natural-language query")
    p.add_argument("-k", "--top-k", type=int, default=8)
    p.add_argument("--flat", action="store_true", help="raw ranked chunks, not merged")
    args = p.parse_args(argv)

    settings = get_settings()

    if args.flat:
        hits = search(args.query, top_k=args.top_k, settings=settings)
        if not hits:
            print("No results. Have you ingested anything (rag-ingest)?")
            return 0
        print(f'\nTop {len(hits)} for: "{args.query}"')
        for h in hits:
            path = " > ".join(h.ancestor_path)
            print(f"\n[{h.score:.3f}] {_one_line(path, 74)}")
            print(f"        {_one_line(h.clause, 74)}")
        return 0

    result = retrieve(args.query, top_k=args.top_k, settings=settings)
    if not result.roots:
        print("No results. Have you ingested anything (rag-ingest)?")
        return 0
    print(
        f'\nChunksets for: "{args.query}"  '
        f"({len(result.hits)} hits, {len(result.roots)} document(s))"
    )
    print()
    print(result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
