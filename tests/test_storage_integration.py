"""Phase 4 storage integration test — hits a real Postgres/pgvector.

Skips cleanly when '.[storage]' isn't installed or no DB is reachable at
RAG_DATABASE_URL, so the default test run stays green everywhere.
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("pgvector")
pytest.importorskip("numpy")

from rag_pipeline.config import EmbeddingProvider, ParseStrategy, Settings  # noqa: E402
from rag_pipeline.embedding.providers.dev import DevEmbedder  # noqa: E402
from rag_pipeline.pipeline import ingest_pdf  # noqa: E402
from rag_pipeline.retrieval.search import retrieve  # noqa: E402
from rag_pipeline.storage.repository import Repository  # noqa: E402


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        parse_strategy=ParseStrategy.fallback,
        embedding_provider=EmbeddingProvider.dev,
        embedding_dim=1024,  # matches schema vector(1024)
    )


def _reachable(url: str) -> bool:
    try:
        with psycopg.connect(url, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture
def repo() -> Repository:
    settings = _settings()
    if not _reachable(settings.database_url):
        pytest.skip("Postgres not reachable at RAG_DATABASE_URL")
    r = Repository(settings.database_url)
    r.apply_schema()
    return r


def test_ingest_search_and_recursive_path(sample_pdf, repo):
    settings = _settings()
    doc_id = None
    try:
        result = ingest_pdf(
            sample_pdf, settings=settings, repository=repo, embedder=DevEmbedder(dim=1024)
        )
        doc_id = result.document_id
        assert result.chunk_count > 0

        # Vector search returns scored hits.
        query_vec = DevEmbedder(dim=1024).embed_query("hierarchy retrieval clause")
        hits = repo.vector_search(query_vec, top_k=3)
        assert hits and "score" in hits[0]

        # Recursive path expansion: root (depth 0) first, leaf last.
        path = repo.get_path(hits[0]["node_id"])
        assert path and path[0]["depth"] == 0
        assert path[-1]["depth"] >= path[0]["depth"]
    finally:
        if doc_id:
            repo.delete_document(doc_id)


def test_retrieve_returns_merged_chunksets(sample_pdf, repo):
    settings = _settings()
    embedder = DevEmbedder(dim=1024)
    doc_id = None
    try:
        doc_id = ingest_pdf(
            sample_pdf, settings=settings, repository=repo, embedder=embedder
        ).document_id

        result = retrieve(
            "structure aware retrieval hierarchy",
            top_k=5,
            settings=settings,
            repo=repo,
            embedder=embedder,
        )
        assert result.roots, "expected at least one merged chunkset root"
        assert result.roots[0].depth == 0          # a title node anchors the tree
        assert result.render().strip()             # renders a non-empty hierarchy
    finally:
        if doc_id:
            repo.delete_document(doc_id)
