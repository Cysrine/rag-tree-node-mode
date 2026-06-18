"""Signal #1: the PDF's own bookmarks/outline.

If a PDF ships an outline, it is the most reliable structure source we have — it
was authored by hand. We flatten it to (title, nesting level, page) entries and
let the tree builder match them back to parsed elements.

Best-effort: returns [] if pypdf is missing, the file has no outline, or parsing
fails. Never raises.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutlineEntry:
    title: str
    level: int          # 0 = top level
    page: int | None    # 1-based page number, if resolvable


def extract_outline(path: str) -> list[OutlineEntry]:
    try:
        from pypdf import PdfReader
    except Exception:  # noqa: BLE001
        return []

    try:
        reader = PdfReader(path)
        raw = reader.outline
    except Exception:  # noqa: BLE001
        return []

    entries: list[OutlineEntry] = []

    def walk(node: object, level: int) -> None:
        if not isinstance(node, list):
            return
        for item in node:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            page: int | None = None
            try:
                num = reader.get_destination_page_number(item)  # 0-based
                page = num + 1 if num is not None else None
            except Exception:  # noqa: BLE001
                page = None
            entries.append(OutlineEntry(str(title).strip(), level, page))

    try:
        walk(raw, 0)
    except Exception:  # noqa: BLE001
        return []
    return entries
