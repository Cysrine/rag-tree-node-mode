"""Structural interface shared by the Postgres and in-memory repositories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RepositoryLike(Protocol):
    """The repository surface the pipeline + retrieval depend on."""

    def ping(self, timeout: float = 5.0) -> None: ...

    def store_document(
        self,
        *,
        title: str,
        source_path: str,
        nodes: list,
        chunks: list,
        embeddings: Sequence[Sequence[float]],
    ) -> str: ...

    def get_paths(self, node_ids: Sequence[Any]) -> dict[str, list[dict]]: ...

    def vector_search(self, embedding: Sequence[float], top_k: int = 10) -> list[dict]: ...
