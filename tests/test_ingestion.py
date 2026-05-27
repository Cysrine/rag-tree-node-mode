"""Phase 1 parser tests (pdfplumber fallback path + orchestrator routing)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ParseStrategy, Settings, get_settings
from rag_pipeline.ingestion.elements import ElementType
from rag_pipeline.ingestion.pdf_loader import detect_scanned, load_pdf
from rag_pipeline.ingestion.pdfplumber_adapter import parse_with_pdfplumber

pytest.importorskip("pdfplumber")


def test_fallback_extracts_text_and_fonts(sample_pdf):
    doc = parse_with_pdfplumber(sample_pdf)
    assert doc.page_count == 1
    assert doc.elements, "expected some elements"

    # Every element carries geometry + font metrics (the signals Phase 2 needs).
    assert all(el.bbox is not None for el in doc.elements)
    assert any(el.font and el.font.size for el in doc.elements)

    # The largest-font element is the title, at ~24pt.
    biggest = max((e for e in doc.elements if e.font and e.font.size), key=lambda e: e.font.size)
    assert "Structure Aware Retrieval" in biggest.text
    assert round(biggest.font.size) == 24
    assert biggest.font.bold is True

    all_text = " ".join(e.text for e in doc.elements)
    assert "hierarchy end to end" in all_text


def test_fallback_detects_list_items(sample_pdf):
    doc = parse_with_pdfplumber(sample_pdf)
    list_items = [e for e in doc.elements if e.element_type is ElementType.LIST_ITEM]
    assert len(list_items) >= 2
    assert all(e.text[0].isdigit() for e in list_items)


def test_title_guess(sample_pdf):
    doc = parse_with_pdfplumber(sample_pdf)
    assert doc.title == "Structure Aware Retrieval"


def test_load_pdf_forced_fallback_orders_elements(sample_pdf):
    settings = Settings(_env_file=None, parse_strategy=ParseStrategy.fallback)
    doc = load_pdf(sample_pdf, settings=settings)
    assert doc.used_fallback is True
    assert doc.strategy_used == "fallback"
    # order_index is a contiguous 0..n-1 reading-order sequence.
    assert [e.order_index for e in doc.elements] == list(range(len(doc.elements)))


def test_detect_scanned_false_on_digital(sample_pdf):
    scanned, avg = detect_scanned(sample_pdf, get_settings())
    assert scanned is False
    assert avg > 100


def test_detect_scanned_true_on_image_only(image_only_pdf):
    scanned, avg = detect_scanned(image_only_pdf, get_settings())
    assert scanned is True
    assert 0 <= avg < 100


def test_scanned_pdf_without_ocr_warns(image_only_pdf):
    settings = Settings(_env_file=None, parse_strategy=ParseStrategy.fallback)
    doc = load_pdf(image_only_pdf, settings=settings)
    assert doc.is_scanned is True
    assert any("scanned" in w.lower() for w in doc.warnings)
