"""Phase 4: pluggable embeddings.

    from rag_pipeline.embedding import get_embedder
    emb = get_embedder()            # provider chosen by RAG_EMBEDDING_PROVIDER
    vecs = emb.embed_documents([...])

The interface + the dependency-free ``dev`` provider are implemented now (tests
use them). The default ``bge`` provider works once '.[embed-local]' is installed;
``voyage``/``openai`` are stubs.
"""

from __future__ import annotations

from rag_pipeline.embedding.provider import Embedder, get_embedder

__all__ = ["Embedder", "get_embedder"]
