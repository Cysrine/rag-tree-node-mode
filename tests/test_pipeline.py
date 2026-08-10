"""Phase 4 pipeline wiring (no DB, no heavy deps): dev embedder + fake repository."""

from __future__ import annotations

from rag_pipeline.config import EmbeddingProvider, ParseStrategy, Settings
from rag_pipeline.pipeline import ingest_pdf


class _FakeRepo:
    def __init__(self):
        self.stored = {}

    def ping(self, timeout: float = 5.0) -> None:
        return None

    def store_document(self, *, title, source_path, nodes, chunks, embeddings):
        self.stored = dict(
            title=title, source_path=source_path, nodes=nodes, chunks=chunks, embeddings=embeddings
        )
        return "fake-doc-id"


def test_ingest_wires_tree_chunks_and_embeddings(sample_pdf):
    settings = Settings(
        _env_file=None,
        parse_strategy=ParseStrategy.fallback,
        embedding_provider=EmbeddingProvider.dev,
        embedding_dim=64,
    )
    repo = _FakeRepo()
    result = ingest_pdf(sample_pdf, settings=settings, repository=repo)

    assert result.document_id == "fake-doc-id"
    assert result.node_count > 0 and result.chunk_count > 0
    assert result.title  # resolved from the tree root, not the filename fallback

    nodes = repo.stored["nodes"]
    chunks = repo.stored["chunks"]
    embeddings = repo.stored["embeddings"]

    # Every node got a stable id; chunk node_ids reference real nodes.
    assert all(n.id for n in nodes)
    node_ids = {n.id for n in nodes}
    assert all(c.node_id in node_ids for c in chunks)

    # One embedding per chunk, each of the configured dimension.
    assert len(embeddings) == result.chunk_count == len(chunks)
    assert all(len(v) == 64 for v in embeddings)

    # Orphan guard end-to-end: every chunk carries a heading path.
    assert all(c.ancestor_path for c in chunks)
