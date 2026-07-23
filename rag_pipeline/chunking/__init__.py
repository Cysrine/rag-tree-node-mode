"""Phase 3: hierarchy-preserving, leaf-anchored chunker with orphan guard.

    from rag_pipeline.chunking import chunk_document
    chunks = chunk_document(nodes)   # nodes from rag_pipeline.structure.build_tree
"""

from __future__ import annotations

from rag_pipeline.chunking.chunker import Chunk, chunk_document, estimate_tokens

__all__ = ["Chunk", "chunk_document", "estimate_tokens"]
