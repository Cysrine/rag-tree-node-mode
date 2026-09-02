"""Phase 6 orchestrator: question -> retrieve -> prompt -> LLM -> grounded answer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.generation.llm import LLMClient, get_llm
from rag_pipeline.generation.prompt import Citation, build_prompt
from rag_pipeline.retrieval.chunkset import RetrievalResult
from rag_pipeline.retrieval.search import retrieve

# Reasoning models (e.g. Qwen3) may wrap chain-of-thought in <think>...</think>.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_NO_CONTEXT = "I couldn't find anything relevant in the indexed documents to answer that."


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    result: RetrievalResult | None = None


def _strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def generate_answer(
    query: str,
    result: RetrievalResult,
    llm: LLMClient,
    *,
    settings: Settings | None = None,
) -> Answer:
    """Build the prompt from a retrieval result and get a grounded answer."""
    system, user, citations = build_prompt(query, result.roots)
    if not citations:
        return Answer(question=query, text=_NO_CONTEXT, citations=[], result=result)
    raw = llm.complete(system, user)
    return Answer(question=query, text=_strip_think(raw), citations=citations, result=result)


def answer(
    query: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
    repo=None,
    embedder=None,
    llm: LLMClient | None = None,
) -> Answer:
    """Full RAG: retrieve chunksets, prompt the LLM, return a cited answer."""
    settings = settings or get_settings()
    top_k = top_k or settings.retrieval_top_k
    result = retrieve(query, top_k=top_k, settings=settings, repo=repo, embedder=embedder)
    llm = llm or get_llm(settings)
    return generate_answer(query, result, llm, settings=settings)
