"""Ask a grounded question over the ingested corpus (Phase 6).

    rag-ask "what is the penalty for late payment?"
    rag-ask "..." -k 10 --show-context

Retrieves root-to-leaf chunksets and asks the configured LLM (default: Groq) to
answer using only that context, with [n] citations to structural locations.
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline.config import get_settings
from rag_pipeline.generation.generator import answer
from rag_pipeline.logging_setup import enable_utf8_stdout


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    p = argparse.ArgumentParser(prog="rag-ask", description="Ask a grounded question.")
    p.add_argument("question", help="natural-language question")
    p.add_argument("-k", "--top-k", type=int, default=None, help="chunks to retrieve")
    p.add_argument("--show-context", action="store_true", help="print retrieved chunksets too")
    args = p.parse_args(argv)

    settings = get_settings()
    try:
        result = answer(args.question, top_k=args.top_k, settings=settings)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.show_context and result.result is not None:
        print("=== CONTEXT ===")
        print(result.result.render())
        print()

    print("=== ANSWER ===")
    print(result.text)
    if result.citations:
        print("\n=== SOURCES ===")
        for c in result.citations:
            path = " > ".join(c.path) if c.path else "(document)"
            print(f"[{c.n}] {path}  (node {c.node_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
