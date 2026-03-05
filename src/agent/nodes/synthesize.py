"""
src/agent/nodes/synthesize.py
──────────────────────────────
Node: synthesize_report

Uses gpt-4o to synthesize all retrieved context into a structured
research report following the analyst report format.

The report structure is deliberately rigid:
  - Executive Summary (3-5 bullets)
  - Thematic sections per sub-question
  - Regulatory-Risk Cross-Reference table
  - Key Quantitative Metrics list

Why rigid structure?
  1. Forces the LLM to organize, not just summarize
  2. Makes the output auditable — each section maps to specific retrieved context
  3. The Risk-Regulatory Cross-Reference table demonstrates cross-document synthesis
     (the key capability that justifies the multi-step agentic approach)

Citation rule: This node instructs the LLM to cite inline.
The format_output node then validates those citations against actual chunk metadata.
"""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.prompts import SYNTHESIZE_SYSTEM, SYNTHESIZE_USER
from ..state import AgentState


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_llm_deployment,
        temperature=0,
        max_tokens=3000,
    )


def run(state: AgentState) -> dict:
    """
    Synthesize a structured research report from all retrieved context.
    """
    llm = _build_llm()

    # Build context string organized by sub-question
    context_sections: list[str] = []

    for sq in state["sub_questions"]:
        section_header = f"### Sub-question {sq['id']}: {sq['question']}"
        if sq.get("document_scope"):
            section_header += f" [Source: {sq['document_scope']}]"

        chunks_text = []
        for i, chunk in enumerate(sq.get("retrieved_chunks", [])[:5]):
            doc_id = chunk.get("document_id", "unknown")
            page = chunk.get("page_start", "?")
            section = chunk.get("section_path_str", "")
            chunk_type = chunk.get("chunk_type", "prose")

            header = f"[Chunk {i+1}: {doc_id}, p.{page}"
            if section:
                header += f", §{section}"
            header += f", type={chunk_type}]"

            text = chunk.get("text", "")[:800]
            chunks_text.append(f"{header}\n{text}")

        if not chunks_text:
            chunks_text = ["(No chunks retrieved for this sub-question)"]

        context_sections.append(
            section_header + "\n\n" + "\n\n".join(chunks_text)
        )

    context_by_subquestion = "\n\n" + "=" * 60 + "\n\n".join(context_sections)

    messages = [
        SystemMessage(content=SYNTHESIZE_SYSTEM),
        HumanMessage(
            content=SYNTHESIZE_USER.format(
                original_query=state["original_query"],
                context_by_subquestion=context_by_subquestion,
            )
        ),
    ]

    response = llm.invoke(messages)
    draft_report = response.content.strip()

    return {"draft_report": draft_report}
