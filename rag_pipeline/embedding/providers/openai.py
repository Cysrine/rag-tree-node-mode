"""OpenAI embedder. STUB — wire up in Phase 4 if you switch providers.

Needs '.[embed-openai]' and OPENAI_API_KEY. text-embedding-3-small is 1536-dim,
text-embedding-3-large is 3072 — update RAG_EMBEDDING_DIM and the vector(N) column
to match, or pass `dimensions=` to shorten.
"""

from __future__ import annotations

from collections.abc import Sequence


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        raise NotImplementedError(
            "OpenAI provider is a stub. Default is 'bge'. Implement in Phase 4 "
            "(pip install '.[embed-openai]', set OPENAI_API_KEY, match the vector dim)."
        )

    @property
    def dim(self) -> int:  # pragma: no cover
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        ...

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        ...
