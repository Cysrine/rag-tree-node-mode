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
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Parsing ---
    parse_strategy: ParseStrategy = ParseStrategy.auto
    # Comma-separated Tesseract language codes, e.g. "eng" or "eng,deu".
    ocr_languages: str = "eng"
    hi_res_model: str = "yolox"
    infer_table_structure: bool = True
    # Average extractable chars/page below which a PDF is treated as scanned.
    min_chars_per_page_digital: int = 100

    @property
    def ocr_language_list(self) -> list[str]:
        return [code.strip() for code in self.ocr_languages.split(",") if code.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (reads env / .env once)."""
    return Settings()
