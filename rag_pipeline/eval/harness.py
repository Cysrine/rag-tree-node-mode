"""Retrieval evaluation: measure context-completeness alongside recall/MRR.

Context-completeness (the spec's key metric): for each query, did we return the
expected clause AND deliver its full structural context (right title + heading)?
The harness is embedder/DB-agnostic — pass any ``retrieve_fn(query, k)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from rag_pipeline.retrieval.chunkset import RetrievalResult

RetrieveFn = Callable[[str, int], RetrievalResult]


@dataclass
class EvalCase:
    query: str
    expect_path_contains: list[str]  # substrings that must appear across the chunkset path
    expect_clause_contains: str      # substring that must appear in a retrieved clause


@dataclass
class CaseResult:
    case: EvalCase
    hit: bool                 # expected clause found within top-k
    context_complete: bool    # found AND its path contains every expected part
    rank: int | None          # 1-based rank of the matching chunkset, if any
    top_score: float


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def recall(self) -> float:
        return _mean(1.0 if r.hit else 0.0 for r in self.results)

    @property
    def context_completeness(self) -> float:
        return _mean(1.0 if r.context_complete else 0.0 for r in self.results)

    @property
    def mrr(self) -> float:
        return _mean((1.0 / r.rank) if (r.hit and r.rank) else 0.0 for r in self.results)

    def summary(self) -> str:
        return (
            f"cases={self.n}  recall={self.recall:.2f}  "
            f"context_completeness={self.context_completeness:.2f}  mrr={self.mrr:.2f}"
        )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def evaluate_case(case: EvalCase, result: RetrievalResult) -> CaseResult:
    want = case.expect_clause_contains.lower()
    match = None
    rank = None
    for i, cs in enumerate(result.chunksets):
        if want in (cs.clause or "").lower():
            match, rank = cs, i + 1
            break

    hit = match is not None
    context_complete = False
    if hit:
        path_texts = [p.text.lower() for p in match.path]
        context_complete = all(
            any(part.lower() in text for text in path_texts) for part in case.expect_path_contains
        )
    top_score = result.hits[0].score if result.hits else 0.0
    return CaseResult(
        case=case,
        hit=hit,
        context_complete=context_complete,
        rank=rank,
        top_score=top_score,
    )


def run_eval(cases: list[EvalCase], retrieve_fn: RetrieveFn, top_k: int = 5) -> EvalReport:
    return EvalReport([evaluate_case(c, retrieve_fn(c.query, top_k)) for c in cases])
