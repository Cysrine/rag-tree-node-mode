"""Phase 2: structural tree assembly (depth + parent + order + confidence).

Signal priority: PDF bookmarks > numbering patterns > font clustering > position,
with a flat-but-ordered fallback when structural confidence is low.
"""

from __future__ import annotations
