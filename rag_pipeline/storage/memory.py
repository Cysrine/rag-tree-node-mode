"""In-memory repository: same interface as ``Repository``, no Postgres.

Lets the full ingest -> retrieve -> answer pipeline (and the eval harness) run
with zero infrastructure. Vector search is brute-force cosine (dot product on
normalized embeddings), which is fine for tests, demos, and small corpora.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any


class InMemoryRepository:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.nodes: dict[str, dict] = {}
        self.chunks: list[dict] = []

    def ping(self, timeout: float = 5.0) -> None:
        """Always reachable — in-memory has no connection to check."""
        return None

    def store_document(
        self,
        *,
        title: str,
        source_path: str,
        nodes: list,
        chunks: list,
        embeddings: Sequence[Sequence[float]],
    ) -> str:
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {"title": title, "source_path": source_path}
        for n in nodes:
            self.nodes[str(n.id)] = {
                "id": str(n.id),
                "parent_id": str(n.parent.id) if n.parent is not None else None,
                "node_type": n.node_type,
                "depth": n.depth,
                "order_index": n.order_index,
                "text": n.text,
                "document_id": doc_id,
            }
        for c, emb in zip(chunks, embeddings, strict=True):
            self.chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "document_id": doc_id,
                    "node_id": str(c.node_id),
                    "text": c.text,
                    "embed_input": c.embed_input,
                    "ancestor_path": list(c.ancestor_path),
                    "embedding": [float(x) for x in emb],
                }
            )
        return doc_id

    def _path(self, node_id: Any) -> list[dict]:
        rows: list[dict] = []
        current = self.nodes.get(str(node_id))
        while current is not None:
            rows.append(dict(current))
            parent_id = current["parent_id"]
            current = self.nodes.get(parent_id) if parent_id else None
        rows.reverse()  # root first
        return rows

    def get_path(self, node_id: Any) -> list[dict]:
        return self._path(node_id)

    def get_paths(self, node_ids: Sequence[Any]) -> dict[str, list[dict]]:
        return {str(nid): self._path(nid) for nid in node_ids if str(nid) in self.nodes}

    def vector_search(self, embedding: Sequence[float], top_k: int = 10) -> list[dict]:
        query = [float(x) for x in embedding]
        scored = [
            (sum(a * b for a, b in zip(query, ch["embedding"], strict=False)), ch)
            for ch in self.chunks
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            {
                "id": ch["id"],
                "node_id": ch["node_id"],
                "text": ch["text"],
                "ancestor_path": list(ch["ancestor_path"]),
                "score": float(score),
            }
            for score, ch in scored[:top_k]
        ]

    def table_counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "nodes": len(self.nodes),
            "chunks": len(self.chunks),
        }

    def delete_document(self, document_id: Any) -> None:
        did = str(document_id)
        self.documents.pop(did, None)
        self.nodes = {k: v for k, v in self.nodes.items() if v["document_id"] != did}
        self.chunks = [c for c in self.chunks if c["document_id"] != did]
