"""Phase 6: prompt formatting + answer generation (fake LLM, no network)."""

from __future__ import annotations

from rag_pipeline.generation.generator import generate_answer
from rag_pipeline.generation.prompt import SYSTEM_PROMPT, build_prompt, format_context
from rag_pipeline.retrieval.chunkset import Hit, RetrievalResult, assemble, merge


class _FakeRepo:
    def __init__(self, paths):
        self._paths = paths

    def get_paths(self, node_ids):
        return {nid: self._paths[nid] for nid in node_ids if nid in self._paths}


class _FakeLLM:
    def __init__(self, out):
        self.out = out
        self.last_system = None
        self.last_user = None

    def complete(self, system, user):
        self.last_system, self.last_user = system, user
        return self.out


def _row(id, parent, ntype, depth, order, text):
    return {"id": id, "parent_id": parent, "node_type": ntype, "depth": depth,
            "order_index": order, "text": text}


_PATHS = {
    "l1": [_row("r", None, "title", 0, 0, "Loan Agreement"),
           _row("b", "r", "heading", 1, 1, "Section 2. Interest Rates"),
           _row("l1", "b", "paragraph", 2, 0, "base rate clause")],
    "l2": [_row("r", None, "title", 0, 0, "Loan Agreement"),
           _row("a", "r", "heading", 1, 0, "Section 1. Definitions"),
           _row("l2", "a", "paragraph", 2, 0, "definitions clause")],
}


def _result():
    hits = [
        Hit(chunk_id="c1", node_id="l1", score=0.9, clause="the base rate is 5% per annum"),
        Hit(chunk_id="c2", node_id="l2", score=0.4, clause="terms are defined here"),
    ]
    chunksets = assemble(hits, _FakeRepo(_PATHS))
    roots = merge(chunksets)
    return RetrievalResult(query="q", hits=hits, chunksets=chunksets, roots=roots)


def test_format_context_numbers_clauses_with_paths():
    result = _result()
    context, citations = format_context(result.roots)
    assert len(citations) == 2
    # Clause under Interest Rates is highest-scoring -> cited [1] with its heading path.
    c1 = next(c for c in citations if c.node_id == "l1")
    assert c1.path == ["Loan Agreement", "Section 2. Interest Rates"]
    assert "the base rate is 5% per annum" in context
    assert "[title] Loan Agreement" in context
    assert context.count("Loan Agreement") == 1  # title not repeated


def test_build_prompt_has_sources_and_instructions():
    system, user, citations = build_prompt("what is the rate?", _result().roots)
    assert system == SYSTEM_PROMPT
    assert "CONTEXT" in user and "SOURCES" in user and "QUESTION: what is the rate?" in user
    assert "node_id=l1" in user
    assert len(citations) == 2


def test_generate_answer_strips_think_and_returns_citations():
    result = _result()
    llm = _FakeLLM("<think>internal reasoning</think>The base rate is 5% [1].")
    ans = generate_answer("what is the rate?", result, llm)
    assert ans.text == "The base rate is 5% [1]."   # <think> removed, trimmed
    assert len(ans.citations) == 2
    # The LLM actually received our grounding prompt.
    assert "CONTEXT" in llm.last_user
    assert "only" in llm.last_system.lower()


def test_generate_answer_handles_no_context():
    empty = RetrievalResult(query="q", hits=[], chunksets=[], roots=[])
    llm = _FakeLLM("should not be called")
    ans = generate_answer("anything?", empty, llm)
    assert not ans.citations
    assert "couldn't find" in ans.text.lower()
    assert llm.last_user is None  # LLM not invoked when there is no context
