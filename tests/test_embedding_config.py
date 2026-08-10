"""Embedding interface (dev provider), config defaults, and table serialization."""

from __future__ import annotations

import math

from rag_pipeline.config import EmbeddingProvider, ParseStrategy, Settings
from rag_pipeline.embedding import get_embedder
from rag_pipeline.embedding.providers.dev import DevEmbedder
from rag_pipeline.ingestion.unstructured_adapter import _html_table_to_markdown


def test_dev_embedder_dim_and_normalization():
    emb = DevEmbedder(dim=64)
    v = emb.embed_query("hello structured world")
    assert len(v) == 64
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6


def test_dev_embedder_is_deterministic_and_order_preserving():
    emb = DevEmbedder(dim=64)
    assert emb.embed_query("alpha beta") == emb.embed_query("alpha beta")
    docs = emb.embed_documents(["alpha", "beta"])
    assert docs[0] != docs[1]
    assert emb.embed_documents(["alpha", "beta"])[0] == docs[0]


def test_get_embedder_dev_respects_dim():
    settings = Settings(_env_file=None, embedding_provider=EmbeddingProvider.dev, embedding_dim=32)
    emb = get_embedder(settings)
    assert emb.dim == 32
    assert len(emb.embed_query("x")) == 32


def test_config_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_provider is EmbeddingProvider.bge
    assert s.embedding_dim == 1024
    assert s.parse_strategy is ParseStrategy.auto
    assert s.ocr_language_list == ["eng"]


def test_config_multi_ocr_languages():
    s = Settings(_env_file=None, ocr_languages="eng, deu ,fra")
    assert s.ocr_language_list == ["eng", "deu", "fra"]


def test_html_table_to_markdown():
    html = "<table><tr><th>Rate</th><th>Term</th></tr><tr><td>5%</td><td>30y</td></tr></table>"
    md = _html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| Rate | Term |"
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert lines[2] == "| 5% | 30y |"


def test_html_table_escapes_pipes():
    md = _html_table_to_markdown("<table><tr><td>a|b</td></tr></table>")
    assert "a\\|b" in md
