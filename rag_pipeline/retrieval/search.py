"""Phase 5: vector search + full retrieval (search -> chunksets -> merged view)."""

from __future__ import annotations

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embedding import get_embedder
from rag_pipeline.retrieval.chunkset import Hit, RetrievalResult, assemble, merge
from rag_pipeline.storage.base import RepositoryLike
from rag_pipeline.storage.repository import Repository


def search(
    query: str,
    *,
    top_k: int = 10,
    settings: Settings | None = None,
    repo: RepositoryLike | None = None,
    embedder=None,
) -> list[Hit]:
    """Embed the query and return the top-k nearest leaf clauses (ranked)."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    repo = repo or Repository(settings.database_url)

    query_vec = embedder.embed_query(query)
    rows = repo.vector_search(query_vec, top_k=top_k)
    return [
        Hit(
            chunk_id=str(r["id"]),
            node_id=str(r["node_id"]),
            score=float(r["score"]),
            clause=r["text"],
            ancestor_path=list(r.get("ancestor_path") or []),
        )
        for r in rows
    ]


def retrieve(
    query: str,
    *,
    top_k: int = 10,
    settings: Settings | None = None,
    repo: RepositoryLike | None = None,
    embedder=None,
) -> RetrievalResult:
    """Full retrieval: top-k hits -> root-to-leaf chunksets -> merged hierarchy."""
    settings = settings or get_settings()
    repo = repo or Repository(settings.database_url)
    embedder = embedder or get_embedder(settings)

    hits = search(query, top_k=top_k, settings=settings, repo=repo, embedder=embedder)
    chunksets = assemble(hits, repo)
    roots = merge(chunksets)
    return RetrievalResult(query=query, hits=hits, chunksets=chunksets, roots=roots)
