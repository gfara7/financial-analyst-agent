"""
src/agent/state.py
──────────────────
LangGraph agent state schema.

This is the most important file in the agent package.
Every node reads from and writes to this TypedDict.
Defining it separately from node logic means:
  - Each node can be unit-tested by passing a mock AgentState dict
  - The graph wiring in graph.py imports only state, not node implementation
  - Reviewers can understand the full agent lifecycle from this one file

State lifecycle:
  1. User query → original_query
  2. decompose_query → sub_questions (list of SubQuestion dicts)
  3. retrieve_context → populated retrieved_chunks on each sub_question
     (current_sq_index advances from 0 to len(sub_questions)-1)
  4. evaluate_sufficiency → overall_sufficiency + insufficiency_reasons
  5. (optional) refine_retrieval → reformulated sub_questions, retry
  6. synthesize_report → draft_report
  7. format_output → final_report
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SubQuestion(TypedDict):
    """A single decomposed sub-question with its retrieval results."""

    id: str                             # "sq_1", "sq_2", etc.
    question: str                       # The sub-question text
    document_scope: Optional[str]       # "10k" | "fsr" | "basel3" | "ecb_bulletin" | None
    retrieved_chunks: list[dict]        # Raw results from FinancialSearchClient
    retrieval_score: float              # Sufficiency score from evaluate node (0.0-1.0)
    sufficiency_reason: str             # LLM's explanation of the score
    missing_context: str                # What is missing (used by refine node)
    retry_count: int                    # How many retrieval retries for this sub-question


class AgentState(TypedDict):
    """
    Full state for the Financial Analyst Agent.

    Design notes:
      - `messages` uses `add_messages` reducer so nodes can append without
        reading and rewriting the full list (LangGraph requirement).
      - `current_sq_index` is an integer counter for sequential fan-out over
        sub_questions. Each retrieve_context call processes one sub-question.
        This is simpler to debug than parallel Send() API for an MVP.
      - `MAX_ITERATIONS` is read-only once set; prevents infinite loops.
    """

    # ── Input ───────────────────────────────────────────────────────────────────
    original_query: str

    # ── Decomposition phase ─────────────────────────────────────────────────────
    sub_questions: list[SubQuestion]
    decomposition_retry_count: int      # How many times decompose_query has been retried

    # ── Retrieval phase ─────────────────────────────────────────────────────────
    current_sq_index: int               # Index into sub_questions for current retrieval
    total_chunks_retrieved: int

    # ── Evaluation phase ────────────────────────────────────────────────────────
    overall_sufficiency: float          # Aggregate score across all sub-questions
    insufficiency_reasons: list[str]    # Reasons why retrieval was insufficient

    # ── Synthesis phase ─────────────────────────────────────────────────────────
    draft_report: Optional[str]
    final_report: Optional[str]

    # ── Message history (required by LangChain message utilities) ───────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Control ─────────────────────────────────────────────────────────────────
    error: Optional[str]                # Set if an unrecoverable error occurs
    iteration_count: int                # Guards against unexpected loops
    MAX_ITERATIONS: int                 # Hard ceiling — default: 3


def initial_state(query: str) -> AgentState:
    """Create the starting state for a new query."""
    return AgentState(
        original_query=query,
        sub_questions=[],
        decomposition_retry_count=0,
        current_sq_index=0,
        total_chunks_retrieved=0,
        overall_sufficiency=0.0,
        insufficiency_reasons=[],
        draft_report=None,
        final_report=None,
        messages=[],
        error=None,
        iteration_count=0,
        MAX_ITERATIONS=3,
    )
