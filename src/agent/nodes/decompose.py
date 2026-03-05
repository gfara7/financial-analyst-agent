"""
src/agent/nodes/decompose.py
─────────────────────────────
Node: decompose_query

Receives the user's complex query and uses gpt-4o to break it into
3-6 targeted sub-questions, each scoped to a specific document source.

Why decompose?
  A complex financial query like "What are the key credit risk factors for large
  US banks and how does Basel III address them?" spans multiple documents:
    - JPMorgan 10-K: what specific credit risks the bank reports
    - Fed FSR: systemic view of credit risk across the banking system
    - Basel III: regulatory framework and capital requirements for credit risk
    - ECB Bulletin: macro context driving credit conditions

  A single retrieval query against all documents would dilute relevance.
  Targeted per-document sub-questions produce much higher precision.

Output: updates state["sub_questions"] with the decomposed list.
"""

from __future__ import annotations

import json
import re
import uuid

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_USER
from ..state import AgentState, SubQuestion


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_llm_deployment,
        temperature=0,
        max_tokens=1000,
    )


def run(state: AgentState) -> dict:
    """
    Decompose the original query into targeted sub-questions.

    Returns partial state update (only keys that change).
    """
    llm = _build_llm()

    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_USER.format(query=state["original_query"])),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # ── Parse JSON response ────────────────────────────────────────────────────
    sub_questions = _parse_decomposition(raw)

    if not sub_questions:
        # Return retry signal (no sub_questions → edges.py routes back)
        return {
            "sub_questions": [],
            "decomposition_retry_count": state.get("decomposition_retry_count", 0) + 1,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    return {
        "sub_questions": sub_questions,
        "current_sq_index": 0,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _parse_decomposition(raw: str) -> list[SubQuestion]:
    """Parse the LLM's JSON output into SubQuestion dicts."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    sub_questions: list[SubQuestion] = []
    valid_scopes = {"10k", "fsr", "basel3", "ecb_bulletin", None}

    for i, item in enumerate(data[:6]):  # Max 6 sub-questions
        question = item.get("question", "").strip()
        if len(question.split()) < 5:  # Minimum word check
            continue

        doc_scope = item.get("document_scope")
        if doc_scope not in valid_scopes:
            doc_scope = None

        sq: SubQuestion = {
            "id": item.get("id", f"sq_{i+1}"),
            "question": question,
            "document_scope": doc_scope,
            "retrieved_chunks": [],
            "retrieval_score": 0.0,
            "sufficiency_reason": "",
            "missing_context": "",
            "retry_count": 0,
        }
        sub_questions.append(sq)

    return sub_questions
