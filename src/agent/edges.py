"""
src/agent/edges.py
───────────────────
All conditional routing logic for the LangGraph agent.

Keeping edge functions in a single file means:
  - graph.py contains zero business logic (only wiring)
  - Edge conditions are easily unit-testable with mock states
  - Interviewers can see ALL routing decisions at once

Edge functions follow the LangGraph convention:
  - Take AgentState as input
  - Return a string key that maps to a destination node
"""

from __future__ import annotations

from src.config import settings
from .state import AgentState


def route_after_decompose(state: AgentState) -> str:
    """
    After decompose_query: validate the decomposed sub-questions.

    Routes:
      "valid"   → retrieve_context   (decomposition succeeded)
      "retry"   → decompose_query    (malformed output, try again)
      "failed"  → END                (gave up after MAX_ITERATIONS)

    Interview talking point:
      "This guard prevents the agent from proceeding with malformed sub-questions.
       If gpt-4o returns non-JSON or empty sub-questions, we retry. After 3 failures
       we surface the error rather than silently producing a bad report."
    """
    # Hard ceiling guard
    if state.get("iteration_count", 0) >= state.get("MAX_ITERATIONS", 3):
        return "failed"

    sub_questions = state.get("sub_questions", [])

    # Validate minimum count and content
    if not sub_questions or len(sub_questions) < 2:
        return "retry"

    for sq in sub_questions:
        if not sq.get("question") or len(sq["question"].split()) < 5:
            return "retry"

    return "valid"


def route_after_retrieve(state: AgentState) -> str:
    """
    After retrieve_context: check if more sub-questions need retrieval.

    Routes:
      "more_questions" → retrieve_context  (process next sub-question)
      "all_done"       → evaluate_sufficiency

    This implements sequential fan-out: current_sq_index advances from 0
    to len(sub_questions)-1, then we proceed to evaluation.
    """
    idx = state.get("current_sq_index", 0)
    total = len(state.get("sub_questions", []))

    if idx < total:
        return "more_questions"
    return "all_done"


def route_after_evaluate(state: AgentState) -> str:
    """
    After evaluate_sufficiency: decide whether to synthesize or retry.

    Routes:
      "sufficient"   → synthesize_report  (all scores >= threshold)
      "needs_refine" → refine_retrieval   (some scores too low, retry budget remains)
      "max_retries"  → synthesize_report  (retry budget exhausted, proceed best-effort)

    The threshold is configurable in settings (default: 0.60).
    The retry ceiling is max_retries_per_subquestion (default: 3).

    Interview talking point:
      "The fallback to synthesize_report on max_retries is intentional. A partial
       answer with honest uncertainty statements ('specific data on X was not in
       retrieved context') is more useful than an error or infinite loop. This is
       the production principle: degrade gracefully, never silently."
    """
    overall_sufficiency = state.get("overall_sufficiency", 0.0)

    # Check how many retrieval retries have been exhausted
    sub_questions = state.get("sub_questions", [])
    max_retries_done = (
        max((sq.get("retry_count", 0) for sq in sub_questions), default=0)
        if sub_questions else 0
    )

    if max_retries_done >= settings.max_retries_per_subquestion:
        return "max_retries"

    if overall_sufficiency >= settings.sufficiency_threshold:
        return "sufficient"

    return "needs_refine"
