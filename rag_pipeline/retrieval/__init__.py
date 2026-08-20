"""Phase 5: retrieval engine — vector search + root-to-leaf chunkset assembly.

    from rag_pipeline.retrieval import retrieve
    result = retrieve("how do I defend against DoS attacks?", top_k=8)
    print(result.render())   # merged title -> section -> clause hierarchy
"""

from __future__ import annotations

from rag_pipeline.retrieval.chunkset import (
    Chunkset,
    Hit,
    MergedNode,
    RetrievalResult,
    assemble,
    merge,
    render,
)
from rag_pipeline.retrieval.search import retrieve, search

__all__ = [
    "search",
    "retrieve",
    "assemble",
    "merge",
    "render",
    "Hit",
    "Chunkset",
    "MergedNode",
    "RetrievalResult",
]
