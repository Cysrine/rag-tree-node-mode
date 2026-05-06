"""Tiny logging helper. Use ``get_logger(__name__)`` throughout the package."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def enable_utf8_stdout() -> None:
    """Make stdout/stderr tolerate any Unicode (Windows consoles default to cp1252).

    Call at the start of a CLI so printing real-document text never crashes.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("RAG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
