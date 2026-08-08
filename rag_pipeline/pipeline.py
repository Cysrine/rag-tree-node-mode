"""Phase 4 orchestrator: PDF -> parse -> tree -> chunk -> embed -> store.

    from rag_pipeline.pipeline import ingest_pdf
    result = ingest_pdf("doc.pdf")   # returns document_id + counts

Also the ``rag-ingest`` CLI entrypoint.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass

from rag_pipeline.chunking.chunker import chunk_document
from rag_pipeline.config import ParseStrategy, Settings, get_settings
from rag_pipeline.embedding import get_embedder
from rag_pipeline.ingestion.elements import ParsedDocument
from rag_pipeline.ingestion.pdf_loader import load_pdf
from rag_pipeline.logging_setup import enable_utf8_stdout, get_logger
from rag_pipeline.storage.base import RepositoryLike
from rag_pipeline.storage.repository import Repository
from rag_pipeline.structure.tree_builder import build_tree

logger = get_logger(__name__)


@dataclass
class IngestResult:
    document_id: str
    title: str
    node_count: int
    chunk_count: int
    tree_mode: str


def ingest_parsed(
    doc: ParsedDocument,
    *,
    settings: Settings | None = None,
    repository: RepositoryLike | None = None,
    embedder=None,
) -> IngestResult:
    """Index an already-parsed document: tree -> chunk -> embed -> store.

    Split out from ``ingest_pdf`` so callers (e.g. the eval harness) can index a
    document built in memory, without a PDF file on disk.
    """
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    repository = repository or Repository(settings.database_url)
    repository.ping()  # fail fast if storage is unreachable — before the slow embed

    nodes = build_tree(doc)
    # Assign stable ids up front so chunk.node_id references them.
    for n in nodes:
        n.id = str(uuid.uuid4())

    # Use the embedder's real tokenizer for budgeting when it exposes one.
    counter = getattr(embedder, "count_tokens", None)
    chunks = chunk_document(nodes, settings=settings, count_tokens=counter)

    logger.info("Embedding %d chunks ...", len(chunks))
    embeddings = embedder.embed_documents([c.embed_input for c in chunks]) if chunks else []

    title = nodes[0].text if nodes else (doc.title or "Document")
    doc_id = repository.store_document(
        title=title,
        source_path=doc.source_path,
        nodes=nodes,
        chunks=chunks,
        embeddings=embeddings,
    )
    return IngestResult(
        document_id=doc_id,
        title=title,
        node_count=len(nodes),
        chunk_count=len(chunks),
        tree_mode=str(doc.metadata.get("tree_mode", "")),
    )


def ingest_pdf(
    path: str,
    settings: Settings | None = None,
    repository: RepositoryLike | None = None,
    embedder=None,
) -> IngestResult:
    """Run the full ingest pipeline for one PDF and persist it."""
    settings = settings or get_settings()
    repository = repository or Repository(settings.database_url)
    repository.ping()  # fail fast before the slow parse + embed, not after
    doc = load_pdf(path, settings=settings)
    return ingest_parsed(doc, settings=settings, repository=repository, embedder=embedder)


def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    p = argparse.ArgumentParser(
        prog="rag-ingest", description="Parse -> tree -> chunk -> embed -> store a PDF."
    )
    p.add_argument("pdf", help="path to a PDF file")
    p.add_argument("--strategy", choices=[s.value for s in ParseStrategy])
    p.add_argument("--init-db", action="store_true", help="apply schema.sql before ingesting")
    args = p.parse_args(argv)

    settings = get_settings()
    if args.strategy:
        settings = settings.model_copy(update={"parse_strategy": ParseStrategy(args.strategy)})

    if args.init_db:
        Repository(settings.database_url).apply_schema()
        print("Applied schema.")

    try:
        result = ingest_pdf(args.pdf, settings=settings)
    except FileNotFoundError:
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nIngested: {result.title}")
    print(f"  document_id: {result.document_id}")
    print(f"  tree mode:   {result.tree_mode}")
    print(f"  nodes:       {result.node_count}")
    print(f"  chunks:      {result.chunk_count}")
    print(f"  provider:    {settings.embedding_provider.value} (dim {settings.embedding_dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
