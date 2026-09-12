# RAG-Tree

**Structure-aware Retrieval-Augmented Generation for PDFs — retrieved text is
never orphaned from its headings.**

Most RAG pipelines split documents into fixed-size windows, which quietly
destroys the most useful property a document has: its hierarchy. RAG-Tree parses
a PDF into a structural tree, chunks at the leaves, and makes every chunk carry
its full ancestor path. Retrieval returns a complete root-to-leaf chunkset:
document title -> section -> subsection -> the exact clause.

## Pipeline

```
PDF
 |- 1   parse      ->  elements: text + bounding box + font metrics
    |- 1.5 order   ->  column-aware reading order
       |- 2 tree   ->  adjacency-list hierarchy + confidence
          |- 3 chunk        ->  leaf-anchored chunks + ancestor path
             |- 4 embed/store  ->  pgvector, tree and vectors in one DB
                |- 5 retrieve     ->  recursive ancestry expansion + merge
                   |- 6 answer       ->  grounded, cited generation  [wip]
```

## Quick start

Structure extraction needs no database, no model download and no network:

```bash
python -m venv .venv
pip install -e ".[dev]"

rag-parse path/to/file.pdf --max 40   # elements + font tiers
rag-tree  path/to/file.pdf            # the reconstructed table of contents
rag-chunk path/to/file.pdf            # chunks + orphan count (must be 0)
pytest -q
```

## With a database

```bash
docker compose up -d db                       # Postgres 16 + pgvector, host port 5433
pip install -e ".[storage,embed-local]"

rag-ingest data/mydoc.pdf                     # parse -> tree -> chunk -> embed -> store
rag-search "how do I defend against DoS attacks?" -k 8
```

## Data model

The structural tree and the embedding vectors live in one Postgres database, so
ancestry expansion is a single recursive CTE rather than a cross-system join:

```sql
documents -+- nodes   (id, parent_id, node_type, depth, order_index, text, confidence)
           +- chunks  (node_id, text, embed_input, ancestor_path, embedding vector(1024))
```

## Visualise the tree

```bash
rag-tree doc.pdf --format list     --out tree.md     # nested bullets (Obsidian/Notion)
rag-tree doc.pdf --format graph    --max-depth 1     # Mermaid flowchart
rag-tree doc.pdf --format obsidian-vault --out doc-vault
```

## Configuration

All settings are environment variables prefixed `RAG_` (see `.env.example`).

## Testing

```bash
pytest -q
ruff check rag_pipeline tests
```
