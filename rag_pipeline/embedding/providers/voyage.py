"""Voyage AI embedder. STUB — wire up in Phase 4 if you switch providers.

Needs '.[embed-voyage]' and VOYAGE_API_KEY. Use input_type="document"/"query"
so document and query embeddings are asymmetrically optimized.
"""

from __future__ import annotations

from collections.abc import Sequence


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-3", dim: int = 1024) -> None:
        raise NotImplementedError(
            "Voyage provider is a stub. Default is 'bge'. Implement in Phase 4 "
            "(pip install '.[embed-voyage]', set VOYAGE_API_KEY)."
        )

    @property
    def dim(self) -> int:  # pragma: no cover
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        ...

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        ...
