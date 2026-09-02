"""Phase 6: format merged chunksets into an LLM prompt.

Each retrieved clause is rendered inside its document hierarchy (title > section
> subsection) and given a citation number [n]; a SOURCES legend maps [n] to the
structural location + node_id so the model can cite precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_pipeline.retrieval.chunkset import MergedNode

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the provided CONTEXT, "
    "which contains document excerpts organized by their position in the document "
    "hierarchy (title > section > subsection > clause).\n"
    "Rules:\n"
    "- Ground every statement in a specific clause; cite it inline as [n] using the "
    "numbered clauses in the context.\n"
    "- Where helpful, name the structural location (e.g., \"Section 2. Interest Rates\").\n"
    "- If the answer is not in the context, say you don't know — do not guess or use outside "
    "knowledge.\n"
    "- Be concise. Never invent clauses or citation numbers."
)


@dataclass
class Citation:
    n: int
    node_id: str
    path: list[str] = field(default_factory=list)  # ancestor heading texts, root first


def format_context(roots: list[MergedNode]) -> tuple[str, list[Citation]]:
    """Render the merged hierarchy with numbered clauses; return (text, citations)."""
    lines: list[str] = []
    citations: list[Citation] = []
    counter = {"n": 0}

    def walk(node: MergedNode, ancestors: list[str]) -> None:
        is_leaf = bool(node.hits) and not node.children
        if not is_leaf:
            lines.append(f"{'  ' * node.depth}[{node.node_type}] {' '.join(node.text.split())}")
            child_ancestors = ancestors + [node.text]
            for child in sorted(node.children.values(), key=lambda m: m.best_score(), reverse=True):
                walk(child, child_ancestors)
        clause_indent = "  " * (node.depth if is_leaf else node.depth + 1)
        for hit in sorted(node.hits, key=lambda x: x.score, reverse=True):
            counter["n"] += 1
            citations.append(Citation(n=counter["n"], node_id=hit.node_id, path=list(ancestors)))
            lines.append(f"{clause_indent}[{counter['n']}] {' '.join(hit.clause.split())}")

    for root in roots:
        walk(root, [])
    return "\n".join(lines), citations


def build_prompt(query: str, roots: list[MergedNode]) -> tuple[str, str, list[Citation]]:
    """Return (system_prompt, user_prompt, citations)."""
    context, citations = format_context(roots)
    sources = "\n".join(
        f"[{c.n}] {' > '.join(c.path) if c.path else '(document)'}  (node_id={c.node_id})"
        for c in citations
    )
    user = (
        "CONTEXT — document excerpts, indented by hierarchy, with numbered clauses:\n"
        f"{context}\n\n"
        "SOURCES — citation number -> structural location:\n"
        f"{sources}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the context above, citing each claim with [n]."
    )
    return SYSTEM_PROMPT, user, citations
