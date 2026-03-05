"""
tests/test_agent_graph.py
──────────────────────────
Unit tests for the agent graph routing logic.

Tests the edge functions independently — no LLM calls, no Azure calls.
This is possible because edge functions take AgentState dicts as input,
which we can mock easily.

Key tests:
  - test_retry_on_insufficient_context: proves refine loop triggers correctly
  - test_max_iterations_respected: proves agent doesn't loop forever
  - test_sufficient_routes_to_synthesis: happy path routing
"""

import pytest

from src.agent.state import AgentState, SubQuestion, initial_state
from src.agent.edges import (
    route_after_decompose,
    route_after_retrieve,
    route_after_evaluate,
)


def make_sub_question(
    sq_id: str = "sq_1",
    question: str = "What are the credit risk factors?",
    document_scope: str | None = "10k",
    retrieval_score: float = 0.8,
    retry_count: int = 0,
    retrieved_chunks: list | None = None,
) -> SubQuestion:
    return SubQuestion(
        id=sq_id,
        question=question,
        document_scope=document_scope,
        retrieved_chunks=retrieved_chunks or [{"chunk_id": "c1", "text": "sample"}],
        retrieval_score=retrieval_score,
        sufficiency_reason="",
        missing_context="",
        retry_count=retry_count,
    )


# ── route_after_decompose tests ────────────────────────────────────────────────

class TestRouteAfterDecompose:
    def test_valid_decomposition_routes_to_retrieve(self):
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1", "What credit risk factors does JPMorgan identify?"),
            make_sub_question("sq_2", "How does Basel III define credit risk?"),
        ]
        state["iteration_count"] = 0
        assert route_after_decompose(state) == "valid"

    def test_empty_sub_questions_routes_to_retry(self):
        state = initial_state("test query")
        state["sub_questions"] = []
        state["iteration_count"] = 0
        assert route_after_decompose(state) == "retry"

    def test_single_sub_question_routes_to_retry(self):
        """Require at least 2 sub-questions for a meaningful decomposition."""
        state = initial_state("test query")
        state["sub_questions"] = [make_sub_question()]
        state["iteration_count"] = 0
        assert route_after_decompose(state) == "retry"

    def test_max_iterations_routes_to_failed(self):
        state = initial_state("test query")
        state["sub_questions"] = []
        state["iteration_count"] = 3  # == MAX_ITERATIONS
        assert route_after_decompose(state) == "failed"

    def test_short_question_routes_to_retry(self):
        """Sub-questions must have at least 5 words."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1", "Credit risk?"),  # too short
            make_sub_question("sq_2", "What factors matter?"),  # borderline
        ]
        state["iteration_count"] = 0
        assert route_after_decompose(state) == "retry"


# ── route_after_retrieve tests ─────────────────────────────────────────────────

class TestRouteAfterRetrieve:
    def test_more_questions_pending(self):
        """When current_sq_index < total sub-questions, loop back."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1"),
            make_sub_question("sq_2"),
            make_sub_question("sq_3"),
        ]
        state["current_sq_index"] = 1  # Processed sq_1, now on sq_2
        assert route_after_retrieve(state) == "more_questions"

    def test_all_done_when_index_equals_total(self):
        """When current_sq_index == len(sub_questions), all retrieved."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1"),
            make_sub_question("sq_2"),
        ]
        state["current_sq_index"] = 2  # Past both sub-questions
        assert route_after_retrieve(state) == "all_done"

    def test_no_sub_questions_routes_all_done(self):
        state = initial_state("test query")
        state["sub_questions"] = []
        state["current_sq_index"] = 0
        assert route_after_retrieve(state) == "all_done"


# ── route_after_evaluate tests ─────────────────────────────────────────────────

class TestRouteAfterEvaluate:
    def test_sufficient_routes_to_synthesis(self):
        """High scores should go straight to synthesis."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1", retrieval_score=0.85),
            make_sub_question("sq_2", retrieval_score=0.90),
        ]
        state["overall_sufficiency"] = 0.85  # Above default threshold of 0.60
        assert route_after_evaluate(state) == "sufficient"

    def test_low_score_routes_to_refine(self):
        """Scores below threshold should trigger refinement."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1", retrieval_score=0.3, retry_count=0),
            make_sub_question("sq_2", retrieval_score=0.8, retry_count=0),
        ]
        state["overall_sufficiency"] = 0.30  # Below 0.60 threshold
        assert route_after_evaluate(state) == "needs_refine"

    def test_max_retries_routes_to_synthesis(self):
        """After max retries, proceed with best-effort synthesis."""
        state = initial_state("test query")
        state["sub_questions"] = [
            make_sub_question("sq_1", retrieval_score=0.3, retry_count=3),  # max retries
        ]
        state["overall_sufficiency"] = 0.30
        assert route_after_evaluate(state) == "max_retries"

    def test_boundary_at_threshold(self):
        """Score exactly at threshold should be sufficient."""
        state = initial_state("test query")
        state["sub_questions"] = [make_sub_question(retrieval_score=0.60)]
        state["overall_sufficiency"] = 0.60
        assert route_after_evaluate(state) == "sufficient"

    def test_just_below_threshold(self):
        state = initial_state("test query")
        state["sub_questions"] = [make_sub_question(retrieval_score=0.59, retry_count=0)]
        state["overall_sufficiency"] = 0.59
        assert route_after_evaluate(state) == "needs_refine"
