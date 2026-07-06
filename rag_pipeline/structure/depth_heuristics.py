"""Phase 2 heuristics: map structure signals to a heading *level* per element.

Level semantics: 0 = highest heading (biggest), 1 = next, ...; ``None`` = body/leaf.
The tree builder turns these relative levels into depth/parent via a stack, so
absolute values matter less than ordering.

Signal precedence (per the spec):
  1. bookmarks/outline  -> authoritative when present & matchable (handled here via
                           ``apply_outline``; chosen by the tree builder).
  2. numbering patterns -> promote body-sized headings the fonts missed; also used
                           as the primary signal when fonts are uninformative.
  3. font size/weight    -> cluster styles; the most common style is body, larger/
                           bold styles are heading tiers.
  4. indentation         -> (reserved) tiebreaker.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from rag_pipeline.ingestion.elements import Element, ElementType
from rag_pipeline.structure.outline import OutlineEntry

# --- text helpers ---------------------------------------------------------

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


# --- numbering ------------------------------------------------------------

_WORD_HEADING = re.compile(
    r"^\s*(chapter|section|article|part|appendix|annex|schedule|clause)\s+"
    r"(\d+|[ivxlcdm]+|[a-z])\b",
    re.IGNORECASE,
)
_DOTTED = re.compile(r"^\s*(\d+(?:\.\d+){0,6})[.)]?\s+\S")
_PAREN = re.compile(r"^\s*\(?([a-z]|[ivxlcdm]+)[.)]\s+\S", re.IGNORECASE)


@dataclass(frozen=True)
class Numbering:
    kind: str      # word | article | dotted | paren
    depth: int     # 0-based relative depth within the numbering scheme
    label: str


def numbering(text: str) -> Numbering | None:
    m = _WORD_HEADING.match(text)
    if m:
        word = m.group(1).lower()
        return Numbering("article" if word == "article" else "word", 0, m.group(0).strip())
    m = _DOTTED.match(text)
    if m:
        label = m.group(1)
        return Numbering("dotted", label.count(".") , label)
    m = _PAREN.match(text)
    if m:
        return Numbering("paren", 3, m.group(1))
    return None


def _is_promotable(element: Element, num: Numbering) -> bool:
    """Would this (non-heading, body-sized) numbered line be a heading, not prose?"""
    words = element.text.split()
    if not (1 <= len(words) <= 10):
        return False
    if num.kind in ("word", "article"):
        return True
    if num.kind == "dotted":
        return True
    if num.kind == "paren":
        return len(words) <= 8
    return False


# --- font clustering ------------------------------------------------------

Style = tuple[int, bool]  # (rounded size, bold)


@dataclass
class FontModel:
    body_size: int
    body_bold: bool
    level_by_style: dict[Style, int]

    @property
    def num_tiers(self) -> int:
        return len(self.level_by_style)


def font_model(elements: list[Element]) -> FontModel | None:
    """Cluster font styles; most common style = body, larger/bold styles = headings."""
    styles: Counter[Style] = Counter()
    for e in elements:
        if e.font and e.font.size:
            styles[(round(e.font.size), bool(e.font.bold))] += 1
    if not styles:
        return None
    body_size, body_bold = styles.most_common(1)[0][0]
    heading_styles = [
        st
        for st in styles
        if st[0] > body_size or (st[0] == body_size and st[1] and not body_bold)
    ]
    # Bigger first; at equal size, bold outranks non-bold.
    heading_styles.sort(key=lambda st: (-st[0], not st[1]))
    level_by_style = {st: i for i, st in enumerate(heading_styles)}
    return FontModel(body_size, body_bold, level_by_style)


def _style_of(element: Element) -> Style | None:
    if element.font and element.font.size:
        return (round(element.font.size), bool(element.font.bold))
    return None


# --- running headers / footers -------------------------------------------


def detect_running_headers(elements: list[Element], page_count: int) -> set[int]:
    """Indices of short lines repeated across a large fraction of pages (page furniture)."""
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for e in elements:
        if 1 <= len(e.text.split()) <= 8:
            pages_by_text[norm(e.text)].add(e.page_number)
    threshold = max(4, int(0.3 * max(1, page_count)))
    repeated = {t for t, pages in pages_by_text.items() if len(pages) >= threshold}
    if not repeated:
        return set()
    return {
        i
        for i, e in enumerate(elements)
        if 1 <= len(e.text.split()) <= 8 and norm(e.text) in repeated
    }


# --- outline matching -----------------------------------------------------


def _outline_usable(outline: list[OutlineEntry], elements: list[Element]) -> bool:
    if len(outline) < 3:
        return False
    index = _element_index(elements)
    matched = sum(1 for entry in outline if _match_outline(entry, index) is not None)
    return matched >= 0.5 * len(outline)


def _element_index(elements: list[Element]) -> dict[tuple[int | None, str], int]:
    index: dict[tuple[int | None, str], int] = {}
    for i, e in enumerate(elements):
        index.setdefault((e.page_number, norm(e.text)), i)
    return index


def _match_outline(
    entry: OutlineEntry, index: dict[tuple[int | None, str], int]
) -> int | None:
    key = norm(entry.title)
    if not key:
        return None
    return index.get((entry.page, key))


def apply_outline(
    levels: list[int | None], elements: list[Element], outline: list[OutlineEntry]
) -> bool:
    """If the outline is usable, stamp its levels onto matched elements. Returns applied?"""
    if not _outline_usable(outline, elements):
        return False
    index = _element_index(elements)
    applied = False
    for entry in outline:
        idx = _match_outline(entry, index)
        if idx is not None:
            levels[idx] = entry.level
            applied = True
    return applied


# --- orchestrator ---------------------------------------------------------


def assign_levels(
    elements: list[Element], outline: list[OutlineEntry] | None
) -> tuple[list[int | None], str, FontModel | None]:
    """Return (level-per-element, mode, font_model). ``mode`` names the winning signal."""
    n = len(elements)
    levels: list[int | None] = [None] * n
    fm = font_model(elements)

    # 1. Bookmarks win outright when present and matchable.
    if outline and apply_outline(levels, elements, outline):
        return levels, "outline", fm

    # 3. Font tiers (primary for the pdfplumber path).
    if fm and fm.num_tiers > 0:
        for i, e in enumerate(elements):
            style = _style_of(e)
            if style is not None:
                lvl = fm.level_by_style.get(style)
                if lvl is not None:
                    levels[i] = lvl
        base = fm.num_tiers
    elif fm is None:
        # No font metrics (unstructured path): trust its Title tags + category_depth.
        for i, e in enumerate(elements):
            if e.element_type is ElementType.TITLE:
                levels[i] = e.category_depth if e.category_depth is not None else 0
        base = max((lvl for lvl in levels if lvl is not None), default=-1) + 1
    else:
        base = 0

    # 2. Numbering: promote body-sized numbered headings the fonts missed.
    promoted = False
    for i, e in enumerate(elements):
        if levels[i] is not None or e.element_type is ElementType.LIST_ITEM:
            continue
        num = numbering(e.text)
        if num and _is_promotable(e, num):
            levels[i] = base + num.depth
            promoted = True

    if fm and fm.num_tiers > 0:
        mode = "fonts"
    elif any(lvl is not None for lvl in levels):
        mode = "numbering" if promoted else "tags"
    else:
        mode = "flat"
    return levels, mode, fm
