"""Run a retrieval eval set against the configured store (Phase 7).

    rag-eval evalset.json -k 8

evalset.json is a list of objects:
    [{"query": "...", "expect_path_contains": ["Title", "Section"],
      "expect_clause_contains": "..."}]
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_pipeline.config import get_settings
from rag_pipeline.eval.harness import EvalCase, run_eval
from rag_pipeline.logging_setup import enable_utf8_stdout
from rag_pipeline.retrieval.search import retrieve


def _load_cases(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        EvalCase(
            query=c["query"],
            expect_path_contains=list(c.get("expect_path_contains", [])),
            expect_clause_contains=c.get("expect_clause_contains", ""),
        )
        for c in raw
    ]


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    p = argparse.ArgumentParser(prog="rag-eval", description="Evaluate retrieval quality.")
    p.add_argument("evalset", help="JSON eval set (see module docstring)")
    p.add_argument("-k", "--top-k", type=int, default=8)
    args = p.parse_args(argv)

    try:
        cases = _load_cases(args.evalset)
    except FileNotFoundError:
        print(f"error: file not found: {args.evalset}", file=sys.stderr)
        return 2

    settings = get_settings()

    def retrieve_fn(query: str, k: int):
        return retrieve(query, top_k=k, settings=settings)

    report = run_eval(cases, retrieve_fn, top_k=args.top_k)

    print(f"\n{'query':<45} hit  ctx  rank")
    print("-" * 62)
    for r in report.results:
        q = (r.case.query[:42] + "…") if len(r.case.query) > 43 else r.case.query
        print(
            f"{q:<45} {'Y' if r.hit else '.'}    "
            f"{'Y' if r.context_complete else '.'}    {r.rank or '-'}"
        )
    print("-" * 62)
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
