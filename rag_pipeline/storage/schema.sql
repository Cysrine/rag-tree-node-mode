-- Structure-aware RAG store: documents -> nodes (adjacency-list tree) -> chunks.
-- Applied automatically by the docker-compose 'db' service on first boot.
-- Vector dimension is 768 to match the default embedding model
-- (BAAI/bge-base-en-v1.5). If you change RAG_EMBEDDING_DIM, change vector(N) too.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Structural tree: adjacency list. Every heading, clause, list, table is a node.
CREATE TABLE IF NOT EXISTS nodes (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id    UUID REFERENCES nodes(id) ON DELETE CASCADE,
    node_type    TEXT NOT NULL,          -- title | heading | subheading | paragraph | list_item | article | table
    depth        INT  NOT NULL,          -- title = 0, its headings = 1, ...
    order_index  INT  NOT NULL,          -- sibling order (logical reading order)
    text         TEXT,                   -- raw text; for tables, serialized markdown
    confidence   REAL,                   -- structure-detection confidence 0..1
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_nodes_document ON nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent   ON nodes(parent_id);

-- Chunks: leaf-anchored, embedded. Each references its source node.
CREATE TABLE IF NOT EXISTS chunks (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    node_id        UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    text           TEXT NOT NULL,        -- clause text (what's shown to LLM)
    embed_input    TEXT NOT NULL,        -- path-prefixed text (what's embedded)
    ancestor_path  TEXT[],               -- denormalized ["Title","Section 3","3.2 Rates"] for fast display
    embedding      vector(768),          -- match dim to chosen model
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);
