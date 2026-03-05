"""
src/agent/nodes/retrieve.py
────────────────────────────
Node: retrieve_context

For the current sub-question (identified by state["current_sq_index"]),
performs hybrid search and stores the results in the sub-question's
`retrieved_chunks` field.

The node processes ONE sub-question per invocation.
The conditional edge in edges.py loops this node until all sub-questions
have been retrieved for.

Why one sub-question at a time (sequential fan-out)?
  - Simpler to debug: state["current_sq_index"] shows exactly where we are
  - Avoids the more complex Send() API (which would run them in parallel)
  - For MVP clarity: sequential is fine for 3-6 sub-questions at 100ms each
  - In production, you'd use Send() for parallel retrieval

Retrieval uses all three stages:
  1. BM25 on query text (financial term exact match)
  2. HNSW vector ANN (semantic recall)
  3. Azure semantic reranking cross-encoder (quality refinement)
"""

from __future__ import annotations

from src.retrieval.search_client import FinancialSearchClient
from ..state import AgentState, SubQuestion

_search_client: FinancialSearchClient | None = None


def _get_search_client() -> FinancialSearchClient:
    global _search_client
    if _search_client is None:
        _search_client = FinancialSearchClient()
    return _search_client


def run(state: AgentState) -> dict:
    """
    Retrieve context for state["sub_questions"][current_sq_index].
    Advances current_sq_index by 1.
    """
    idx = state["current_sq_index"]
    sub_questions = list(state["sub_questions"])  # Make a copy

    if idx >= len(sub_questions):
        # Safety guard — should not happen with correct edge routing
        return {"current_sq_index": idx}

    sq = dict(sub_questions[idx])  # Copy the sub-question

    client = _get_search_client()

    # Hybrid search with optional document scope filter
    results = client.hybrid_search(
        query_text=sq["question"],
        document_type_filter=sq.get("document_scope"),
    )

    sq["retrieved_chunks"] = results
    sq["retry_count"] = sq.get("retry_count", 0)

    # Update the sub_questions list
    sub_questions[idx] = sq

    total_retrieved = state.get("total_chunks_retrieved", 0) + len(results)

    return {
        "sub_questions": sub_questions,
        "current_sq_index": idx + 1,
        "total_chunks_retrieved": total_retrieved,
    }
