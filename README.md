# RAG-Tree

**Structure-aware Retrieval-Augmented Generation for PDFs — retrieved text is never orphaned from its headings.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Postgres](https://img.shields.io/badge/postgres-16%20%2B%20pgvector-336791)](https://github.com/pgvector/pgvector)
[![Tests](https://img.shields.io/badge/tests-65%20passing-brightgreen)](#testing)
[![Lint](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The problem

Almost every RAG pipeline splits documents into fixed-size windows. That's convenient, and it
silently destroys the most useful property a document has: its hierarchy.

Retrieve this from a contract:

> "The base rate is five percent per annum."

Five percent of *what*, under which clause, in which agreement? The context that makes the sentence
interpretable — *Loan Agreement → Section 2. Interest Rates* — lived in headings that the chunker cut
away several windows earlier and never associated with the text beneath them.

It gets worse at scale. A 919-page textbook in the test corpus contains the heading **"Summary" 15
times** and **"Introduction" 15 times** — once per chapter. A hit on "Summary" tells you nothing
unless the system can say *which chapter's*.

## The idea

RAG-Tree parses a PDF into a **structural tree**, chunks at the leaves, and makes every chunk carry
its full ancestor path. Retrieval then returns a complete **root-to-leaf chunkset**:

```mermaid
graph TD
    A["Loan Agreement<br/><i>title · depth 0</i>"] --> B["Section 1. Definitions<br/><i>heading · depth 1</i>"]
    A --> C["Section 2. Interest Rates<br/><i>heading · depth 1</i>"]
    C --> D["The base rate is five percent per annum.<br/><i>clause · depth 2 · retrieved</i>"]
    style D fill:#2f81f7,color:#fff
```

The clause is what you searched for; the path above it is delivered with it, every time. Two
consequences:

- **Structure is recovered *before* chunking**, not bolted on afterwards.
- **Citations are verifiable** — every answer resolves to a node UUID in the database whose ancestry
  can be walked with one SQL query.

## Demo

```console
$ rag-ask "What is the base interest rate and the penalty for late payment?"

=== ANSWER ===
The base interest rate is five percent per annum [1]. Late payments accrue an
additional two percent penalty [1].

=== SOURCES ===
[1] Loan Agreement > Section 2. Interest Rates  (node 1eacd015-8c93-470f-b4fe-ff320e24e1bf)
```

Retrieval on its own (`rag-search`) returns the merged hierarchy — each title and section shown
**once**, matched clauses nested beneath, ranked by relevance:

```console
$ rag-search "interest rate and late payment penalty"

[title] Loan Agreement
  [heading] Section 2. Interest Rates
    - [0.79] (a) The base rate is five percent per annum. (b) Late payments accrue...
  [heading] Section 1. Definitions
    - [0.52] This Agreement sets out the terms between the parties...
```

---

## Quick start

Structure extraction needs **no database, no model download, and no network**. Clone, install the
light dependency set, and inspect a PDF:

```bash
git clone https://github.com/Cysrine/rag-tree-node-mode.git
cd rag-tree-node-mode

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

rag-parse  path/to/file.pdf --max 40   # elements + font tiers
rag-tree   path/to/file.pdf            # the reconstructed table of contents
rag-chunk  path/to/file.pdf            # chunks + orphan count (must be 0)
pytest -q                              # 63 tests, no infrastructure required
```

### Zero-infrastructure end-to-end (Python API)

The full pipeline — including retrieval — runs in-process using the in-memory repository and the
dependency-free dev embedder. Useful for tests and for trying the API without provisioning anything:

```python
from rag_pipeline.config import EmbeddingProvider, ParseStrategy, Settings
from rag_pipeline.embedding.providers.dev import DevEmbedder
from rag_pipeline.ingestion.pdf_loader import load_pdf
from rag_pipeline.pipeline import ingest_parsed
from rag_pipeline.retrieval.search import retrieve
from rag_pipeline.storage.memory import InMemoryRepository

settings = Settings(
    _env_file=None,
    parse_strategy=ParseStrategy.fallback,
    embedding_provider=EmbeddingProvider.dev,
    embedding_dim=1024,
)
repo, embedder = InMemoryRepository(), DevEmbedder(dim=1024)

doc = load_pdf("path/to/file.pdf", settings=settings)
ingest_parsed(doc, settings=settings, repository=repo, embedder=embedder)

result = retrieve("interest rate", top_k=5, settings=settings, repo=repo, embedder=embedder)
print(result.render())
```

> **Note:** the `dev` embedder is a deterministic token-hash, not a semantic model. It exercises the
> plumbing exactly; for meaningful retrieval quality use BGE (below).

---

## Full setup

### 1. Database

```bash
docker compose up -d db     # Postgres 16 + pgvector on host port 5433
```

Port **5433** is used deliberately, so a Postgres you already run on 5432 is left alone. The schema
is applied automatically on first boot; override the connection with `RAG_DATABASE_URL`.

### 2. Embeddings and the LLM

```bash
pip install -e ".[storage,embed-local,llm-groq]"
```

First run downloads **BAAI/bge-large-en-v1.5** (~1.3 GB). For generation, put your API key in `.env`
under the name given by `RAG_LLM_API_KEY_ENV` (default `GROQ-KEY`):

```dotenv
GROQ-KEY=your-groq-api-key
```

### 3. Ingest, search, ask

```bash
rag-ingest data/mydoc.pdf                       # parse → tree → chunk → embed → store
rag-search "how do I defend against DoS attacks?" -k 8
rag-ask    "how do I defend against DoS attacks?" --show-context
```

### Optional: high-resolution parsing and OCR

The `unstructured` layout model (plus Poppler and Tesseract) gives native table extraction and OCR
for scanned PDFs. Those native dependencies are awkward on Windows, so a Linux parser image is
included:

```bash
docker compose --profile parse run --rm parser /data/yourfile.pdf   # put PDFs in ./data
```

Or install natively on Linux/macOS: `pip install -e ".[parse]"` plus `tesseract-ocr` and
`poppler-utils`. **Without it the pipeline still works** — it falls back to `pdfplumber` and recovers
structure from font metrics instead.

---

## CLI reference

| Command | Purpose | Needs DB? |
|---|---|:--:|
| `rag-parse` | Raw elements: page, type, **font size**, text | — |
| `rag-tree` | Reconstructed hierarchy; exports to Markdown/Mermaid/Obsidian | — |
| `rag-chunk` | Chunk sizes, token stats, **orphan count** | — |
| `rag-ingest` | Full ingestion into Postgres | ✔ |
| `rag-search` | Merged root-to-leaf chunksets with scores | ✔ |
| `rag-ask` | Grounded answer with `[n]` citations | ✔ |
| `rag-eval` | recall · **context-completeness** · MRR | ✔ |

Every command accepts `--help`. `rag-parse`, `rag-tree` and `rag-chunk` re-parse the PDF each run, so
you can inspect any stage in isolation without touching the database.

---

## How it works

```
PDF
 └─ 1  parse       →  elements: text + bounding box + font metrics
     └─ 1.5 order  →  column-aware reading order
         └─ 2 tree →  adjacency-list hierarchy + confidence
             └─ 3 chunk    →  leaf-anchored chunks + ancestor path
                 └─ 4 embed/store →  pgvector, tree and vectors in one DB
                     └─ 5 retrieve  →  recursive ancestry expansion + merge
                         └─ 6 answer  →  grounded, cited generation
```

| Stage | What it does |
|---|---|
| **1 · Parse** | Two parsers behind one interface: `unstructured` (layout model + OCR) and a `pdfplumber` fallback recovering coordinates and font metrics. Every heavy dependency is lazy-imported, so missing components degrade with a warning instead of crashing. The fallback deliberately does **not** classify headings — that's stage 2's job. |
| **1.5 · Reading order** | Clusters element left-edges into columns, then orders page → column → top-to-bottom. Without this, multi-column PDFs interleave. |
| **2 · Tree** | Heading detection by prioritised signal: **PDF bookmarks** → **numbering patterns** (`Chapter 1`, `1.2.3`, `Article 4`) → **font clustering** → position. A stack walk turns levels into `depth`/`parent`. Running headers/footers are dropped. When no signal is reliable it emits a flat-but-ordered tree with low confidence rather than inventing structure. |
| **3 · Chunk** | **Orphan guard:** chunks anchor only on leaf nodes, so a heading can never be emitted alone — it rides along as `ancestor_path`. Tiny sibling leaves are grouped; oversized leaves split on sentence boundaries; tables are never split. |
| **4 · Embed & store** | Pluggable embedders (BGE local · dev · Voyage/OpenAI stubs). Chunks embed a **path-prefixed** string so the vector itself carries hierarchy. |
| **5 · Retrieve** | Vector search → one **batched** recursive query expands every hit to its root-to-leaf path → chunksets sharing ancestors are merged so titles aren't repeated. |
| **6 · Generate** | Prompt renders the hierarchy with numbered clauses plus a sources legend mapping each `[n]` to a structural location and node UUID. |
| **7 · Evaluate** | Measures **context-completeness** — did we return the right clause *and* its correct context? — alongside recall and MRR. |

### Data model

The structural tree and the embedding vectors live in **one** Postgres database — no graph store
plus vector store, no cross-system joins:

```sql
documents ─┬─ nodes   (id, parent_id, node_type, depth, order_index, text, confidence)
           └─ chunks  (node_id, text, embed_input, ancestor_path, embedding vector(1024))
```

Which makes ancestry expansion a single recursive CTE — the retrieval core of the project:

```sql
WITH RECURSIVE path AS (
    SELECT id, parent_id, node_type, depth, text FROM nodes WHERE id = %(node_id)s
    UNION ALL
    SELECT n.id, n.parent_id, n.node_type, n.depth, n.text
    FROM nodes n JOIN path p ON n.id = p.parent_id
)
SELECT * FROM path ORDER BY depth ASC;   -- root first, leaf last
```

---

## Visualise the document tree

`rag-tree` exports the recovered hierarchy so you can actually look at it:

```bash
rag-tree doc.pdf --format list     --out tree.md        # nested bullets (Obsidian + Notion)
rag-tree doc.pdf --format headings --out outline.md     # Markdown heading outline
rag-tree doc.pdf --format graph    --max-depth 1        # Mermaid flowchart (any version)
rag-tree doc.pdf --format mermaid  --max-depth 1        # Mermaid mindmap (needs Mermaid ≥ 9.3)
rag-tree doc.pdf --format obsidian-vault --out doc-vault
```

The last one writes **one note per heading**, each linking to its parent, so Obsidian's graph view
renders the whole document as a navigable node web. Use `--max-depth` on large documents.

---

## Configuration

All settings are environment variables prefixed `RAG_` (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` | Postgres + pgvector |
| `RAG_EMBEDDING_PROVIDER` | `bge` | `bge` · `dev` · `voyage` · `openai` |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `RAG_EMBEDDING_DIM` | `1024` | Must match the `vector(N)` column |
| `RAG_PARSE_STRATEGY` | `auto` | `auto` · `hi_res` · `fast` · `ocr_only` · `fallback` |
| `RAG_CHUNK_MAX_TOKENS` | `384` | Upper bound per chunk |
| `RAG_CHUNK_MIN_TOKENS` | `64` | Below this, leaves are grouped with siblings |
| `RAG_LLM_MODEL` | `qwen/qwen3.8-27b` | Groq model id |
| `RAG_LLM_API_KEY_ENV` | `GROQ-KEY` | *Name* of the env var holding the API key |

---

## Testing

```bash
pytest -q          # 65 tests
ruff check rag_pipeline tests
```

**63 of 65 tests run with no infrastructure at all** — no database, no model download, no network —
because the repository, embedder and LLM client are all injectable and an in-memory repository ships
with the project. The two database integration tests skip cleanly when Postgres isn't reachable.

Tests cover the structural invariants directly: exactly one root, `depth == parent.depth + 1`,
parents precede children, **no chunk anchored on a heading**, and every chunk carrying a non-empty
ancestor path.

---

## Results

Measured on a 919-page technical book:

| Stage | Result |
|---|---|
| Parse | 21,711 elements across 919 pages |
| Tree | bookmark outline detected · confidence **0.95** · **184 headings** · title recovered |
| Chunk | **730 chunks** · median 390 tokens · **0 orphaned** |
| Retrieval | recall **1.00** · context-completeness **1.00** · MRR **1.00** |

---

## Project structure

```
rag_pipeline/
├── config.py        environment-driven settings
├── pipeline.py      ingestion orchestrator
├── ingestion/       parsers, reading order, element IR
├── structure/       tree builder, heuristics, outline, confidence, export
├── chunking/        leaf-anchored chunker + orphan guard
├── embedding/       provider interface + bge / dev / voyage / openai
├── storage/         schema.sql, Postgres repo, in-memory repo, Protocol
├── retrieval/       vector search, chunkset assembly + merge
├── generation/      prompt assembly, LLM client, generator
└── eval/            harness, synthetic corpora, invariant tests
```

---

## Limitations

- **Table extraction requires the `[parse]` extra.** The `pdfplumber` fallback recovers text and
  fonts but does not detect tables, so table-heavy documents lose that structure.
- **Shallow bookmarks are reproduced faithfully.** If a PDF's outline is only two levels deep, the
  tree is too — even when typography implies finer structure. A hybrid mode is on the roadmap.
- **CPU embedding is slow.** ~10 minutes for a 900-page document with BGE-large on CPU.
- **No incremental re-ingest.** An updated PDF is re-embedded in full.
- **English-centric heuristics.** Numbering patterns and OCR default to English.

## Roadmap

- Sibling / parent-section expansion for cross-referencing clauses
- Reranking stage before chunkset assembly
- Hybrid structure mode: bookmarks for the skeleton, typography for the leaves
- Incremental re-ingest
- HTTP API around `answer()`

## License

[MIT](LICENSE)
