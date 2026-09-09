"""Phase 7: evaluation & hardening.

Real tests live in test_orphans.py / test_structure.py / test_retrieval.py.
The reusable pieces:
    harness.py  -> EvalCase, run_eval, EvalReport (context-completeness, recall, MRR)
    synth.py    -> make_document / make_pdf (synthetic corpora)
"""

from __future__ import annotations

from rag_pipeline.eval.harness import (
    CaseResult,
    EvalCase,
    EvalReport,
    evaluate_case,
    run_eval,
)

__all__ = ["EvalCase", "CaseResult", "EvalReport", "evaluate_case", "run_eval"]
