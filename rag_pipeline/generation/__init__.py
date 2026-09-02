"""Phase 6: prompt assembly + generation.

    from rag_pipeline.generation import answer
    a = answer("what is the penalty for late payment?")
    print(a.text)          # grounded, with [n] citations
    print(a.citations)     # [n] -> structural location + node_id

Default LLM: Groq (RAG_LLM_MODEL, key from the env var named by RAG_LLM_API_KEY_ENV).
"""

from __future__ import annotations

from rag_pipeline.generation.generator import Answer, answer, generate_answer
from rag_pipeline.generation.prompt import Citation, build_prompt, format_context

__all__ = [
    "answer",
    "generate_answer",
    "Answer",
    "Citation",
    "build_prompt",
    "format_context",
]
