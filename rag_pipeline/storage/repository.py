"""Phase 4/5: DB access — inserts, recursive path expansion, vector search.

All heavy imports (psycopg, pgvector, numpy) are lazy so importing this module
(and the pipeline) never requires the '.[storage]' extra to be installed.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Given a matched chunk's node_id, return the full root-to-leaf path.
RECURSIVE_PATH_SQL = """
WITH RECURSIVE path AS (
    SELECT id, parent_id, node_type, depth, text
    FROM nodes WHERE id = %(node_id)s
    UNION ALL
    SELECT n.id, n.parent_id, n.node_type, n.depth, n.text
    FROM nodes n
    JOIN path p ON n.id = p.parent_id
)
SELECT * FROM path ORDER BY depth ASC;   -- root (title) first, leaf last
"""


def _as_uuid(value: Any) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class Repository:
    """Thin data-access layer over Postgres/pgvector."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self):
        import psycopg  # lazy: needs '.[storage]'

        return psycopg.connect(self._database_url)

    def _endpoint(self) -> str:
        from urllib.parse import urlsplit

        try:
            parts = urlsplit(self._database_url)
            return f"{parts.hostname or '?'}:{parts.port or '?'}"
        except Exception:  # noqa: BLE001
            return "the database"

    def ping(self, timeout: float = 5.0) -> None:
        """Raise a clear error if the database is unreachable (call before slow work)."""
        import psycopg

        try:
            with psycopg.connect(self._database_url, connect_timeout=int(timeout)):
                pass
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cannot reach Postgres at {self._endpoint()}: {exc}. "
                f"Is it running? Start it with:  docker compose up -d db"
            ) from exc

    # --- schema ---
    def apply_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        # Strip line comments first (they can contain ';'), then split on ';'.
        # schema.sql has no dollar-quoted functions or string literals with ';'.
        no_comments = "\n".join(
            line[: line.index("--")] if "--" in line else line for line in sql.splitlines()
        )
        statements = [s.strip() for s in no_comments.split(";") if s.strip()]
        with self._connect() as conn, conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)

    # --- writes ---
    def store_document(
        self,
        *,
        title: str,
        source_path: str,
        nodes: list,
        chunks: list,
        embeddings: Sequence[Sequence[float]],
    ) -> str:
        """Insert a document with its node tree and embedded chunks. Returns document id."""
        import numpy as np
        from pgvector.psycopg import register_vector

        with self._connect() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (title, source_path) VALUES (%s, %s) RETURNING id",
                    (title, source_path),
                )
                doc_id = cur.fetchone()[0]

                # Nodes are already topologically ordered (a parent precedes its
                # children), so per-row FK checks in executemany are satisfied.
                node_rows = [
                    (
                        _as_uuid(n.id),
                        doc_id,
                        _as_uuid(n.parent.id) if n.parent is not None else None,
                        n.node_type,
                        n.depth,
                        n.order_index,
                        n.text,
                        n.confidence,
                    )
                    for n in nodes
                ]
                cur.executemany(
                    "INSERT INTO nodes"
                    " (id, document_id, parent_id, node_type, depth, order_index, text, confidence)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    node_rows,
                )

                chunk_rows = [
                    (
                        doc_id,
                        _as_uuid(c.node_id),
                        c.text,
                        c.embed_input,
                        list(c.ancestor_path),
                        np.asarray(emb, dtype=np.float32),
                    )
                    for c, emb in zip(chunks, embeddings, strict=True)
                ]
                cur.executemany(
                    "INSERT INTO chunks"
                    " (document_id, node_id, text, embed_input, ancestor_path, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    chunk_rows,
                )
        return str(doc_id)

    # --- reads ---
    def get_path(self, node_id: Any) -> list[dict]:
        """Root-to-leaf ancestor path for a node (the retrieval core)."""
        from psycopg.rows import dict_row

        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(RECURSIVE_PATH_SQL, {"node_id": _as_uuid(node_id)})
            return cur.fetchall()

    def get_paths(self, node_ids: Sequence[Any]) -> dict[str, list[dict]]:
        """Batched root-to-leaf paths for many leaves in one recursive query.

        Returns {leaf_node_id: [root_row, ..., leaf_row]} (each list ordered root
        first). Each row carries id/parent_id/node_type/depth/order_index/text.
        """
        from psycopg.rows import dict_row

        ids = [_as_uuid(x) for x in node_ids]
        if not ids:
            return {}
        sql = """
        WITH RECURSIVE path AS (
            SELECT id, parent_id, node_type, depth, order_index, text, id AS origin
            FROM nodes WHERE id = ANY(%(ids)s)
            UNION ALL
            SELECT n.id, n.parent_id, n.node_type, n.depth, n.order_index, n.text, p.origin
            FROM nodes n JOIN path p ON n.id = p.parent_id
        )
        SELECT id, parent_id, node_type, depth, order_index, text, origin
        FROM path ORDER BY origin, depth ASC
        """
        out: dict[str, list[dict]] = {}
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, {"ids": ids})
            for row in cur.fetchall():
                out.setdefault(str(row["origin"]), []).append(row)
        return out

    def vector_search(self, embedding: Sequence[float], top_k: int = 10) -> list[dict]:
        """Top-k nearest chunks by cosine similarity (HNSW). Returns rows with score."""
        import numpy as np
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        vec = np.asarray(embedding, dtype=np.float32)
        with self._connect() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT c.id, c.node_id, c.text, c.ancestor_path,"
                    " 1 - (c.embedding <=> %s) AS score"
                    " FROM chunks c ORDER BY c.embedding <=> %s LIMIT %s",
                    (vec, vec, top_k),
                )
                return cur.fetchall()

    def table_counts(self) -> dict[str, int]:
        with self._connect() as conn, conn.cursor() as cur:
            counts = {}
            for table in ("documents", "nodes", "chunks"):
                cur.execute(f"SELECT count(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
            return counts

    def delete_document(self, document_id: Any) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (_as_uuid(document_id),))
