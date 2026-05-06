"""Structure-aware RAG pipeline.

A PDF is parsed into a structural tree (adjacency list); chunks are leaf-anchored
but always carry their full ancestor path, so retrieval returns complete
root-to-leaf chunksets. See ProjectDetails.md for the full spec.

Note: importing this package is intentionally cheap. Heavy subpackages
(``ingestion``, ``embedding``, ``storage``, ...) are imported explicitly by
callers so optional/heavy dependencies stay lazy.
"""

from __future__ import annotations

__version__ = "0.1.0"

from rag_pipeline.config import Settings, get_settings  # noqa: E402  (light, safe)

__all__ = ["Settings", "get_settings", "__version__"]
