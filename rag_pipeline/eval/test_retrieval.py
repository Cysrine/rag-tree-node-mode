"""Phase 7: retrieval eval — context-completeness + recall on an in-memory corpus.

Uses the deterministic dev embedder (lexical overlap) so the harness exercises the
full retrieve -> chunkset -> merge path without a DB, model download, or network.
Semantic quality is a separate concern (point the harness at BGE + Postgres for that).
"""

from __future__ import annotations

import pytest

from rag_pipeline.config import EmbeddingProvider, Settings
from rag_pipeline.embedding.providers.dev import DevEmbedder
from rag_pipeline.eval.harness import EvalCase, run_eval
from rag_pipeline.eval.synth import make_document
from rag_pipeline.pipeline import ingest_parsed
from rag_pipeline.retrieval.search import retrieve
from rag_pipeline.storage.memory import InMemoryRepository


@pytest.fixture
def corpus():
    settings = Settings(
        _env_file=None,
        embedding_provider=EmbeddingProvider.dev,
        embedding_dim=256,
        chunk_min_tokens=1,
        chunk_max_tokens=1000,
    )
    embedder = DevEmbedder(dim=256)
    repo = InMemoryRepository()
    docs = [
        make_document(
            "loan.pdf",
            "Loan Agreement",
            [
                ("Section 2. Interest Rates", [
                    "The base rate is five percent per annum.",
                    "Late payments accrue an additional two percent penalty.",
                ]),
                ("Section 1. Definitions", [
                    "Capitalized terms have the meanings assigned below.",
                ]),
            ],
        ),
        make_document(
            "hr.pdf",
            "Employee Handbook",
            [
                ("Section 3. Leave Policy", [
                    "Employees accrue fifteen vacation days each year.",
                ]),
                ("Section 4. Conduct", [
                    "Harassment of any kind is strictly prohibited.",
                ]),
            ],
        ),
    ]
    for doc in docs:
        ingest_parsed(doc, settings=settings, repository=repo, embedder=embedder)

    def retrieve_fn(query: str, k: int):
        return retrieve(query, top_k=k, settings=settings, repo=repo, embedder=embedder)

    return retrieve_fn


_LOAN_RATES = ["Loan Agreement", "Interest Rates"]
_CASES = [
    EvalCase("base rate five percent per annum", _LOAN_RATES, "five percent"),
    EvalCase("late payment penalty two percent", _LOAN_RATES, "two percent"),
    EvalCase(
        "vacation days accrue each year", ["Employee Handbook", "Leave Policy"], "vacation days"
    ),
    EvalCase(
        "harassment strictly prohibited conduct", ["Employee Handbook", "Conduct"], "Harassment"
    ),
]


def test_context_completeness_and_recall(corpus):
    report = run_eval(_CASES, corpus, top_k=5)
    assert report.recall == 1.0, report.summary()
    assert report.context_completeness == 1.0, report.summary()


def test_winning_chunkset_carries_full_path(corpus):
    result = corpus("base rate five percent per annum", 5)
    top = result.chunksets[0]
    assert "five percent" in top.clause
    assert [p.text for p in top.path] == [
        "Loan Agreement",
        "Section 2. Interest Rates",
        "The base rate is five percent per annum.",
    ]


def test_cross_document_separation(corpus):
    # An HR query must not surface loan clauses at the top.
    result = corpus("vacation days accrue each year", 3)
    assert "vacation days" in result.chunksets[0].clause
    assert result.chunksets[0].path[0].text == "Employee Handbook"
