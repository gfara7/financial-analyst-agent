"""
src/agent/nodes/refine.py
──────────────────────────
Node: refine_retrieval

Reformulates sub-questions that received low sufficiency scores,
enabling the retrieve_context node to try again with better queries.

This node is the "intelligence" behind the retry loop:
  - It reads the `missing_context` field set by evaluate.py
  - It uses gpt-4o to reformulate the query with better coverage
  - Strategies: add synonyms, broaden scope, split complex questions

After this node, the graph routes back to retrieve_context, which will
re-retrieve for the reformulated sub-questions.

The retry is bounded by `max_retries_per_subquestion` in settings (default: 3).

Interview talking point:
  "The refine node demonstrates that the agent can self-correct. If it asked
   'What is the LCR requirement under Basel III?' and got irrelevant chunks,
   it reformulates to 'liquidity coverage ratio minimum threshold regulatory
   requirement' — broadening the terminology coverage. This is the same pattern
   used in production RAG systems at firms like Morgan Stanley and JPMorgan."
"""

from __future__ import annotations

import json
import re

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.prompts import REFINE_SYSTEM, REFINE_USER
from ..state import AgentState, SubQuestion


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_llm_deployment,
        temperature=0.2,   # Slight creativity for reformulation
        max_tokens=800,
    )


def run(state: AgentState) -> dict:
    """
    Reformulate sub-questions with insufficient context and reset retrieval index.
    """
    llm = _build_llm()
    sub_questions = list(state["sub_questions"])

    # Find sub-questions that need refinement
    to_refine = [
        sq for sq in sub_questions
        if sq.get("retrieval_score", 0.0) < settings.sufficiency_threshold
        and sq.get("retry_count", 0) < settings.max_retries_per_subquestion
    ]

    if not to_refine:
        # Nothing to refine — proceed to synthesis
        return {"current_sq_index": len(sub_questions)}

    # Build prompt context
    insufficient_json = json.dumps(
        [
            {
                "id": sq["id"],
                "original_question": sq["question"],
                "document_scope": sq.get("document_scope"),
                "sufficiency_score": sq.get("retrieval_score", 0.0),
                "what_is_missing": sq.get("missing_context", ""),
                "reason": sq.get("sufficiency_reason", ""),
            }
            for sq in to_refine
        ],
        indent=2,
    )

    messages = [
        SystemMessage(content=REFINE_SYSTEM),
        HumanMessage(content=REFINE_USER.format(insufficient_questions=insufficient_json)),
    ]

    raw = llm.invoke(messages).content.strip()
    refined = _parse_refined(raw)

    # Apply refined questions back to sub_questions list
    refined_map = {r["id"]: r for r in refined}
    updated_sqs = []
    for sq in sub_questions:
        sq = dict(sq)
        if sq["id"] in refined_map:
            refined_sq = refined_map[sq["id"]]
            sq["question"] = refined_sq.get("question", sq["question"])
            sq["document_scope"] = refined_sq.get("document_scope", sq.get("document_scope"))
            # Increment retry count and reset results
            sq["retry_count"] = sq.get("retry_count", 0) + 1
            sq["retrieved_chunks"] = []
            sq["retrieval_score"] = 0.0
        updated_sqs.append(sq)

    return {
        "sub_questions": updated_sqs,
        "current_sq_index": 0,    # Reset to re-retrieve for ALL sub-questions
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _parse_refined(raw: str) -> list[dict]:
    """Parse LLM's JSON reformulation output."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []
