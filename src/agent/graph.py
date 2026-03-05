"""
src/agent/graph.py
───────────────────
LangGraph StateGraph assembly — the architectural centrepiece of the MVP.

This file contains ONLY wiring: no business logic, no prompt templates,
no API calls. It imports nodes and edges and connects them.

Graph overview:
  decompose_query
    ├─[valid]        → retrieve_context
    ├─[retry]        → decompose_query       (self-loop, malformed JSON retry)
    └─[failed]       → END

  retrieve_context
    ├─[more_questions] → retrieve_context   (sequential fan-out)
    └─[all_done]       → evaluate_sufficiency

  evaluate_sufficiency
    ├─[sufficient]    → synthesize_report
    ├─[needs_refine]  → refine_retrieval    (retry loop)
    └─[max_retries]   → synthesize_report   (best-effort fallback)

  refine_retrieval → retrieve_context       (re-retrieval after reformulation)
  synthesize_report → format_output → END

Mermaid visualization:
  Call `graph.get_graph().draw_mermaid()` at runtime, or see README.md.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import decompose, retrieve, evaluate, refine, synthesize, format_output
from .edges import (
    route_after_decompose,
    route_after_retrieve,
    route_after_evaluate,
)


def build_graph():
    """
    Assemble and compile the Financial Analyst Agent graph.

    Returns a CompiledGraph that can be invoked with:
        graph = build_graph()
        result = graph.invoke(initial_state("my query"))
    """
    g = StateGraph(AgentState)

    # ── Register Nodes ─────────────────────────────────────────────────────────
    g.add_node("decompose_query",       decompose.run)
    g.add_node("retrieve_context",      retrieve.run)
    g.add_node("evaluate_sufficiency",  evaluate.run)
    g.add_node("refine_retrieval",      refine.run)
    g.add_node("synthesize_report",     synthesize.run)
    g.add_node("format_output",         format_output.run)

    # ── Entry Point ────────────────────────────────────────────────────────────
    g.set_entry_point("decompose_query")

    # ── Static Edges (always take this path) ──────────────────────────────────
    # refine → retrieve: after reformulation, always re-retrieve
    g.add_edge("refine_retrieval",  "retrieve_context")

    # synthesize → format → done
    g.add_edge("synthesize_report", "format_output")
    g.add_edge("format_output",     END)

    # ── Conditional Edge 1: decompose_query ───────────────────────────────────
    # Validates decomposition output; retries on malformed JSON
    g.add_conditional_edges(
        "decompose_query",
        route_after_decompose,
        {
            "valid":  "retrieve_context",
            "retry":  "decompose_query",    # self-loop
            "failed": END,
        },
    )

    # ── Conditional Edge 2: retrieve_context ──────────────────────────────────
    # Sequential fan-out: processes one sub-question per call
    g.add_conditional_edges(
        "retrieve_context",
        route_after_retrieve,
        {
            "more_questions": "retrieve_context",    # process next sub-question
            "all_done":       "evaluate_sufficiency",
        },
    )

    # ── Conditional Edge 3: evaluate_sufficiency ──────────────────────────────
    # The key retry loop: self-assesses context quality before synthesizing
    g.add_conditional_edges(
        "evaluate_sufficiency",
        route_after_evaluate,
        {
            "sufficient":   "synthesize_report",
            "needs_refine": "refine_retrieval",
            "max_retries":  "synthesize_report",   # proceed despite low scores
        },
    )

    return g.compile()


# Module-level compiled graph (lazy initialization)
_graph = None


def get_graph():
    """Return the compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
