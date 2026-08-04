"""Embedding provider interface + factory.

Every provider implements the same tiny surface so the rest of the pipeline is
provider-agnostic (spec: "pluggable behind an interface; swappable").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_pipeline.config import EmbeddingProvider, Settings, get_settings


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int:
        """Embedding dimension (must match the ``vector(N)`` column)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage. Returns L2-normalized vectors (cosine-ready)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a query. May apply a provider-specific query instruction/prefix."""


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the configured embedder. Heavy providers import lazily."""
    settings = settings or get_settings()
    provider = settings.embedding_provider

    if provider is EmbeddingProvider.dev:
        from rag_pipeline.embedding.providers.dev import DevEmbedder

        return DevEmbedder(dim=settings.embedding_dim)

    if provider is EmbeddingProvider.bge:
        from rag_pipeline.embedding.providers.bge import BGEEmbedder

        return BGEEmbedder(model=settings.embedding_model, dim=settings.embedding_dim)

    if provider is EmbeddingProvider.voyage:
        from rag_pipeline.embedding.providers.voyage import VoyageEmbedder

        return VoyageEmbedder(model=settings.embedding_model, dim=settings.embedding_dim)

    if provider is EmbeddingProvider.openai:
        from rag_pipeline.embedding.providers.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=settings.embedding_model, dim=settings.embedding_dim)

    raise ValueError(f"Unknown embedding provider: {provider!r}")
