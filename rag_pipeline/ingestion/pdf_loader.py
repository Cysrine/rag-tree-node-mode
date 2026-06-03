"""Phase 1 orchestrator: PDF path -> ordered ``ParsedDocument``.

Routing:
  1. Probe the PDF for extractable text -> is it scanned?
  2. Resolve a strategy from config (auto picks hi_res vs ocr_only).
  3. Downgrade gracefully when heavy deps are missing (no unstructured -> pdfplumber;
     OCR needed but no Tesseract -> pdfplumber + warning).
  4. Parse, retry via fallback if empty, then normalize reading order.
"""

from __future__ import annotations

import os
import shutil

from rag_pipeline.config import ParseStrategy, Settings, get_settings
from rag_pipeline.ingestion import reading_order
from rag_pipeline.ingestion.elements import ParsedDocument
from rag_pipeline.ingestion.pdfplumber_adapter import parse_with_pdfplumber
from rag_pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def _unstructured_available() -> bool:
    try:
        import unstructured.partition.pdf  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def detect_scanned(
    path: str, settings: Settings, max_pages: int = 5
) -> tuple[bool, float]:
    """Return (is_scanned, avg_extractable_chars_per_page) using a cheap text probe."""
    try:
        import pdfplumber
    except Exception:  # noqa: BLE001
        return (False, -1.0)

    total = 0
    pages = 0
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:max_pages]:
                total += len((page.extract_text() or "").strip())
                pages += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanned-detection failed for %s: %s", path, exc)
        return (False, -1.0)

    if pages == 0:
        return (False, 0.0)
    avg = total / pages
    return (avg < settings.min_chars_per_page_digital, avg)


def _resolve_strategy(configured: ParseStrategy, scanned: bool) -> str:
    if configured is ParseStrategy.fallback:
        return "fallback"
    if configured is ParseStrategy.auto:
        return "ocr_only" if scanned else "hi_res"
    return configured.value  # hi_res | fast | ocr_only


def _run(path: str, chosen: str, settings: Settings, warnings: list[str]) -> ParsedDocument:
    if chosen == "fallback":
        doc = parse_with_pdfplumber(path)
        doc.used_fallback = True
        return doc
    try:
        from rag_pipeline.ingestion.unstructured_adapter import parse_with_unstructured

        return parse_with_unstructured(
            path,
            strategy=chosen,
            hi_res_model=settings.hi_res_model,
            languages=settings.ocr_language_list,
            infer_table_structure=settings.infer_table_structure,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("unstructured(%s) failed for %s: %s; falling back", chosen, path, exc)
        warnings.append(f"unstructured({chosen}) failed: {exc}. Fell back to pdfplumber.")
        doc = parse_with_pdfplumber(path)
        doc.used_fallback = True
        return doc


def load_pdf(path: str, settings: Settings | None = None) -> ParsedDocument:
    """Parse ``path`` into a reading-ordered ``ParsedDocument`` (Phase 1 entrypoint)."""
    settings = settings or get_settings()
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    warnings: list[str] = []
    scanned, avg_chars = detect_scanned(path, settings)
    chosen = _resolve_strategy(settings.parse_strategy, scanned)

    # Capability-based downgrades.
    if chosen != "fallback" and not _unstructured_available():
        warnings.append(
            "unstructured not installed; using pdfplumber fallback "
            "(install '.[parse]' or use the Docker parser for hi-res/OCR)."
        )
        chosen = "fallback"
    if chosen == "ocr_only" and not _tesseract_available():
        warnings.append(
            "Tesseract not found on PATH; cannot OCR. Using pdfplumber fallback."
        )
        chosen = "fallback"

    logger.info(
        "Parsing %s (scanned=%s, avg_chars/page=%.0f) via %s", path, scanned, avg_chars, chosen
    )
    doc = _run(path, chosen, settings, warnings)

    doc.is_scanned = scanned
    doc.metadata.setdefault("avg_chars_per_page", round(avg_chars, 1))
    doc.warnings = warnings + doc.warnings
    if scanned and not doc.used_ocr:
        doc.warnings.append(
            "Document looks scanned but was parsed without OCR; text may be incomplete. "
            "Install Tesseract + '.[parse]' (or use the Docker parser)."
        )

    doc.elements = reading_order.normalize(doc.elements)
    return doc
