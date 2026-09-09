"""Synthetic documents for evaluation (no PDF needed), plus a reportlab PDF maker."""

from __future__ import annotations

from rag_pipeline.ingestion.elements import Element, ElementType, FontInfo, ParsedDocument

Section = tuple[str, list[str]]  # (heading text, [clause texts])


def _el(text: str, size: float, bold: bool = False) -> Element:
    return Element(
        text=text,
        element_type=ElementType.PARAGRAPH,
        page_number=1,
        font=FontInfo(size=size, bold=bold),
    )


def make_document(
    source: str,
    title: str,
    sections: list[Section],
    *,
    title_size: float = 24,
    heading_size: float = 16,
    body_size: float = 11,
) -> ParsedDocument:
    """Build a ParsedDocument with a clear title/heading/body font hierarchy."""
    elements = [_el(title, title_size, bold=True)]
    for heading, clauses in sections:
        elements.append(_el(heading, heading_size, bold=True))
        elements.extend(_el(c, body_size) for c in clauses)
    return ParsedDocument(source_path=source, title=title, elements=elements, page_count=1)


def make_pdf(
    path,
    title: str,
    sections: list[Section],
    *,
    title_size: int = 22,
    heading_size: int = 15,
    body_size: int = 11,
) -> str:
    """Render the same structure to a real PDF (requires reportlab)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", title_size)
    c.drawString(72, y, title)
    y -= title_size + 18
    for heading, clauses in sections:
        c.setFont("Helvetica-Bold", heading_size)
        c.drawString(72, y, heading)
        y -= heading_size + 10
        c.setFont("Helvetica", body_size)
        for clause in clauses:
            c.drawString(72, y, clause)
            y -= body_size + 6
        y -= 8
    c.showPage()
    c.save()
    return str(path)
