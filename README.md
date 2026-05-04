# RAG-Tree

Structure-aware RAG for PDFs: parse a document into a hierarchy, chunk at the
leaves, and keep every chunk attached to the headings above it.

**Status:** early scaffolding. See `ProjectDetails.md` for the build plan.

## Planned pipeline

```
PDF -> parse -> reading order -> tree -> chunk -> embed -> retrieve -> answer
```

The tree and the vectors are meant to live in one Postgres database, so
expanding a matched clause back up to its title is a single recursive query
rather than a join across two systems.

## Development

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
```
