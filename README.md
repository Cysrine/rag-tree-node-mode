# RAG-Tree

Structure-aware RAG for PDFs: parse a document into a hierarchy, chunk at the
leaves, and keep every chunk attached to the headings above it.

**Status:** parsing and reading-order normalisation are done; the tree builder
is in progress. See `ProjectDetails.md` for the build plan.

## Pipeline

```
PDF
 |- 1   parse      ->  elements: text + bounding box + font metrics   [done]
    |- 1.5 order   ->  column-aware reading order                     [done]
       |- 2 tree   ->  adjacency-list hierarchy + confidence          [wip]
          |- 3 chunk        ->  leaf-anchored chunks + ancestor path
             |- 4 embed/store  ->  pgvector, tree and vectors in one DB
                |- 5 retrieve     ->  recursive ancestry expansion + merge
                   |- 6 answer       ->  grounded, cited generation
```

## Try it

```bash
python -m venv .venv
pip install -e ".[dev]"

rag-parse path/to/file.pdf --max 40   # elements + font tiers
pytest -q
```

Parsing needs no database and no model download: the `pdfplumber` fallback
recovers text, coordinates and font metrics with the light dependency set.
`unstructured` (hi-res layout model + OCR) is optional and lives behind the
`[parse]` extra, or the Linux parser image in `docker-compose.yml`.

## Storage

```bash
docker compose up -d db     # Postgres 16 + pgvector on host port 5433
```

Port 5433 is used deliberately so a Postgres already running on 5432 is left
alone. The schema is applied on first boot.

## Configuration

All settings are environment variables prefixed `RAG_` (see `.env.example`).
