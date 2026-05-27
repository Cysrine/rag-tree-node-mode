"""Shared fixtures. PDFs are synthesized with reportlab so tests need no user files."""

from __future__ import annotations

import pytest


def _canvas(path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    return canvas.Canvas(str(path), pagesize=letter), letter


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    """One page: big title, a heading, body paragraphs, and a numbered list."""
    pytest.importorskip("reportlab")
    path = tmp_path / "sample.pdf"
    c, (width, height) = _canvas(path)

    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, height - 72, "Structure Aware Retrieval")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 120, "Introduction")
    c.setFont("Helvetica", 11)
    c.drawString(72, height - 150, "This document explains the pipeline in detail.")
    c.drawString(72, height - 165, "It preserves document hierarchy end to end.")
    c.drawString(72, height - 180, "The retrieval query walks from leaf to root.")
    c.drawString(72, height - 210, "1. First item in the list")
    c.drawString(72, height - 225, "2. Second item in the list")
    c.showPage()
    c.save()
    return str(path)


@pytest.fixture
def two_column_pdf(tmp_path) -> str:
    """One page, two columns — for reading-order (column) tests."""
    pytest.importorskip("reportlab")
    path = tmp_path / "two_column.pdf"
    c, (width, height) = _canvas(path)
    c.setFont("Helvetica", 11)
    # Left column (x=72), right column (x=350), same vertical positions.
    c.drawString(72, height - 100, "Left column line one")
    c.drawString(72, height - 120, "Left column line two")
    c.drawString(350, height - 100, "Right column line one")
    c.drawString(350, height - 120, "Right column line two")
    c.showPage()
    c.save()
    return str(path)


@pytest.fixture
def image_only_pdf(tmp_path) -> str:
    """One page with an image and no extractable text — a stand-in for a scan."""
    pytest.importorskip("reportlab")
    pytest.importorskip("PIL")
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    path = tmp_path / "scanned.pdf"
    c, (width, height) = _canvas(path)
    img = Image.new("RGB", (400, 300), (200, 30, 30))
    c.drawImage(ImageReader(img), 100, 400, width=400, height=300)
    c.showPage()
    c.save()
    return str(path)
