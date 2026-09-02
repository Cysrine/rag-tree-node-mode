"""Central configuration (env-driven, prefix ``RAG_``).

Kept dependency-light: importing this module must not import torch, unstructured,
psycopg, etc. Those are pulled in lazily by the components that need them.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    bge = "bge"
    voyage = "voyage"
    openai = "openai"
    dev = "dev"


class LLMProvider(str, Enum):
    groq = "groq"
    anthropic = "anthropic"
    openai = "openai"


class ParseStrategy(str, Enum):
    """How to parse a PDF.

    auto      pick per-document: scanned -> ocr_only, digital -> hi_res.
    hi_res    unstructured layout model (best structure).
    fast      unstructured fast text extraction.
    ocr_only  unstructured + Tesseract (scanned docs).
    fallback  pdfplumber only (no heavy deps; text + coords + fonts).
    """

    auto = "auto"
    hi_res = "hi_res"
    fast = "fast"
    ocr_only = "ocr_only"
    fallback = "fallback"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    # Host port 5433 matches docker-compose (kept off 5432 to avoid clashing with a
    # native Postgres). Override via RAG_DATABASE_URL to point elsewhere.
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.bge
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # --- Parsing ---
    parse_strategy: ParseStrategy = ParseStrategy.auto
    # Comma-separated Tesseract language codes, e.g. "eng" or "eng,deu".
    ocr_languages: str = "eng"
    hi_res_model: str = "yolox"
    infer_table_structure: bool = True
    # Average extractable chars/page below which a PDF is treated as scanned.
    min_chars_per_page_digital: int = 100

    # --- Chunking (Phase 3) ---
    # Max is kept under a typical 512-token embedder limit to leave headroom for the
    # ancestor-path prefix in embed_input. Leaves below the min are grouped with
    # siblings; leaves above the max are split on sentence boundaries.
    chunk_max_tokens: int = 384
    chunk_min_tokens: int = 64
    chunk_overlap_tokens: int = 0

    # --- Retrieval / Generation (Phase 5/6) ---
    retrieval_top_k: int = 8
    llm_provider: LLMProvider = LLMProvider.groq
    llm_model: str = "qwen/qwen3.8-27b"
    # Name of the env var holding the API key (read from .env at call time).
    llm_api_key_env: str = "GROQ-KEY"
    llm_base_url: str = ""  # blank = provider default
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    @property
    def ocr_language_list(self) -> list[str]:
        return [code.strip() for code in self.ocr_languages.split(",") if code.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (reads env / .env once)."""
    return Settings()
