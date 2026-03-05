"""
src/agent/nodes/evaluate.py
────────────────────────────
Node: evaluate_sufficiency

Uses gpt-4o as an LLM judge to score whether the retrieved context
is sufficient to answer each sub-question.

Why LLM-as-judge (not a simple metric)?
  Retrieval quality can't be judged by vector similarity scores alone.
  A chunk may be highly similar to the query but still miss the specific
  data point needed (e.g. retrieving a paragraph about credit risk but
  not the actual Tier 1 capital ratio number).

  The LLM judge asks: "Does this context actually answer the question?"
  It scores 0.0-1.0 and explains what is missing. This explanation is
  passed to the refine node if a retry is needed.

Output: updates overall_sufficiency and insufficiency_reasons.
  - If overall_sufficiency >= 0.6: proceed to synthesis
  - If < 0.6 and retry headroom: route to refine_retrieval
  - If retry limit reached: proceed anyway with best-effort

Interview talking point:
  "Without this node, the agent would synthesize over insufficient context
   and confidently hallucinate specific numbers that aren't there. The
   evaluate-refine loop is the mechanism that makes the agent self-aware
   of its own knowledge gaps."
"""

from __future__ import annotations

import json
import re

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.prompts import EVALUATE_SYSTEM, EVALUATE_USER
from ..state import AgentState, SubQuestion


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_llm_deployment,
        temperature=0,
        max_tokens=800,
    )


def run(state: AgentState) -> dict:
    """
    Evaluate retrieval sufficiency for all sub-questions.
    """
    llm = _build_llm()
    sub_questions = list(state["sub_questions"])

    scores: list[float] = []
    insufficiency_reasons: list[str] = []
    updated_sqs: list[SubQuestion] = []

    for sq in sub_questions:
        sq = dict(sq)

        if not sq["retrieved_chunks"]:
            sq["retrieval_score"] = 0.0
            sq["sufficiency_reason"] = "No chunks were retrieved."
            sq["missing_context"] = "All context is missing — retrieval returned empty results."
            scores.append(0.0)
            insufficiency_reasons.append(f"{sq['id']}: No chunks retrieved")
            updated_sqs.append(sq)
            continue

        # Build context string from top-3 retrieved chunks
        context_parts = []
        for i, chunk in enumerate(sq["retrieved_chunks"][:3]):
            text = chunk.get("text", "")
            source = f"[{chunk.get('document_type', '?')} p.{chunk.get('page_start', '?')}]"
            context_parts.append(f"Chunk {i+1} {source}:\n{text[:600]}")
        context_str = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=EVALUATE_SYSTEM),
            HumanMessage(
                content=EVALUATE_USER.format(
                    sq_id=sq["id"],
                    question=sq["question"],
                    context=context_str,
                )
            ),
        ]

        score_data = _parse_evaluation(llm.invoke(messages).content)
        sq_score = score_data.get("score", 0.5)
        sq["retrieval_score"] = sq_score
        sq["sufficiency_reason"] = score_data.get("reason", "")
        sq["missing_context"] = score_data.get("missing", "")

        scores.append(sq_score)
        if sq_score < settings.sufficiency_threshold:
            insufficiency_reasons.append(
                f"{sq['id']}: score={sq_score:.2f} — {sq['sufficiency_reason']}"
            )

        updated_sqs.append(sq)

    # Aggregate: minimum score (weakest link matters for synthesis quality)
    overall_sufficiency = min(scores) if scores else 0.0

    return {
        "sub_questions": updated_sqs,
        "overall_sufficiency": overall_sufficiency,
        "insufficiency_reasons": insufficiency_reasons,
    }


def _parse_evaluation(raw: str) -> dict:
    """Parse LLM evaluation JSON response."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()
    try:
        data = json.loads(raw)
        scores_list = data.get("scores", [])
        if scores_list:
            item = scores_list[0]
            return {
                "score": float(item.get("score", 0.5)),
                "reason": item.get("reason", ""),
                "missing": item.get("missing", ""),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # Fallback: extract a number from the text
    numbers = re.findall(r"\b(0\.\d+|1\.0|0)\b", raw)
    score = float(numbers[0]) if numbers else 0.5
    return {"score": score, "reason": raw[:200], "missing": ""}
