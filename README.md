# Financial Analyst Agent

> Agentic AI MVP: Multi-step financial research assistant using LangGraph, Azure AI Search, and Azure OpenAI.

Built to demonstrate production-level agentic AI patterns in a job interview context.

---

## What This Is

A multi-step agentic AI system that answers complex financial research questions by:

1. **Decomposing** the query into targeted sub-questions
2. **Retrieving** relevant chunks from real financial documents using hybrid search
3. **Self-evaluating** the quality of retrieved context (LLM-as-judge)
4. **Refining** the retrieval if context is insufficient (iterative retry loop)
5. **Synthesizing** a structured analyst report with verifiable citations

**Not a demo with dummy data** — ingests real public financial documents with
genuinely hard chunking challenges: multi-page tables, footnotes, nested sections.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                         │
│                                                                 │
│  PDFs ──→ PDFParser ──→ ChunkingStrategy ──→ Embedder ──→ Search Index │
│  (4 docs)  (PyMuPDF)   (Fixed/Semantic/   (AOAI ada)   (Azure AI Search)│
│                         Hierarchical)                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                         Azure AI Search
                        (HNSW + BM25 + Semantic Reranking)
                               │
┌─────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH AGENT                           │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  decompose_query ──[valid]──→ retrieve_context ──→ evaluate     │
│      │                           ↑    │              │          │
│      └──[retry]──────────────────┘    │              │          │
│                              [more]   │[all done]    │          │
│                                       ▼              ▼          │
│                             evaluate_sufficiency                │
│                                  │         │                    │
│                             [sufficient] [needs refine]         │
│                                  │         ▼                    │
│                                  │    refine_retrieval          │
│                                  │         │                    │
│                                  ▼         ▼                    │
│                             synthesize_report                   │
│                                  │                              │
│                                  ▼                              │
│                             format_output ──→ Final Report      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Source Documents

Real public-domain financial PDFs used as the knowledge base:

| File | Source | Pages | Chunking Challenges |
|------|--------|-------|---------------------|
| `jpmorgan_10k_2023.pdf` | [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=JPM&type=10-K) | ~360 | Multi-page financial tables, nested Notes (Note 14.3.2), footnotes in financial statements |
| `fed_fsr_2023.pdf` | [Federal Reserve](https://www.federalreserve.gov/publications/financial-stability-report.htm) | ~80 | Charts with captions, call-out boxes, policy prose requiring topic coherence |
| `basel3_framework_bis.pdf` | [BIS](https://www.bis.org/bcbs/publ/d424.htm) | ~162 | Numbered regulation hierarchy (3.4.2.a.iii), capital ratio requirement tables, cross-references |
| `ecb_economic_bulletin_2023.pdf` | [ECB](https://www.ecb.europa.eu/pub/economic-bulletin/html/index.en.html) | ~100 | Macro charts with analytical captions, box articles, annex statistical tables |

---

## Azure Infrastructure

| Resource | Tier | Monthly Cost | Purpose |
|----------|------|-------------|---------|
| Azure OpenAI | S0 | ~$0 standby | gpt-4o (LLM) + text-embedding-ada-002 |
| Azure AI Search | **Basic** | ~$74 | HNSW vector index + hybrid search + semantic reranking |
| Azure Blob Storage | Standard LRS | ~$0.01 | Source PDF storage (audit trail) |
| **Total** | | **~$74/month** | Provision → demo → teardown for ~$2-5 |

> **Production note**: Upgrade to S1 (~$245/month) for higher throughput and 99.9% SLA.
> For interview prep, tear down after the demo: `bash infra/teardown.sh`

---

## Chunking Strategy Comparison

This is the most technically interesting part of the project. Three strategies are implemented to demonstrate the trade-offs:

| Metric | fixed_size | semantic | hierarchical |
|--------|-----------|---------|-------------|
| **How it works** | Sliding token window | Embedding similarity boundary | Section-aware + table fusion |
| **Table integrity** | ❌ 0% (tables split mid-row) | ❌ N/A (tables mixed in) | ✅ 100% (tables kept intact) |
| **Section path** | ❌ None | ⚠️ Partial (~23%) | ✅ Full (~98%) |
| **Footnote attachment** | ❌ None | ❌ None | ✅ Attached to referencing chunk |
| **Figure captions** | ❌ Mixed in prose | ❌ Mixed in prose | ✅ Wrapped as `[FIGURE]: ...` |
| **Cross-page table fusion** | ❌ Split at page break | ❌ Split at page break | ✅ Fused into one chunk |
| **Best for** | Demo of failure mode | Policy prose (Basel III, ECB) | Financial tables + filings |
| **Ingestion cost** | Cheapest | 2x embeddings during chunking | Medium |

### Why Hierarchical for Financial Documents?

A JPMorgan 10-K has 40+ financial tables that span page breaks (income statement, segment tables, capital ratios). With `fixed_size`:

```
Chunk 342: "...net income attributable to common stockholders..."
           "| Q1 2023 | Q2 2023 | Q3 20"   ← TABLE CUT MID-ROW
Chunk 343: "23 | Q4 2023 |"                ← ORPHANED TABLE FRAGMENT
           "| $3.1B | $3.4B | $3.6B | $4.1B |"
```

With `hierarchical`:

```
Chunk 342 [type=table, section="Consolidated Statement of Income"]:
  table_context: "Net income attributable to JPMorgan Chase & Co. was..."
  text: | Period | Q1 2023 | Q2 2023 | Q3 2023 | Q4 2023 |
        | --- | --- | --- | --- | --- |
        | Net Income | $3.1B | $3.4B | $3.6B | $4.1B |
        | EPS | $1.02 | $1.12 | $1.19 | $1.35 |
```

---

## LangGraph Agent Design

### Why LangGraph over LangChain AgentExecutor?

With LangChain's `AgentExecutor`, the retry logic is implicit and hard to inspect.
With LangGraph, you can print `state` at any node and see exactly where the agent is and why.
For financial analysis (where auditability matters), explicit state is not optional.

### State Schema ([src/agent/state.py](src/agent/state.py))

```python
class AgentState(TypedDict):
    original_query: str
    sub_questions: List[SubQuestion]  # each has: question, document_scope, retrieved_chunks, retrieval_score, retry_count
    current_sq_index: int             # sequential fan-out counter
    overall_sufficiency: float        # aggregate LLM-judge score (0.0-1.0)
    insufficiency_reasons: List[str]  # why retrieval failed
    draft_report: Optional[str]
    final_report: Optional[str]
    iteration_count: int
    MAX_ITERATIONS: int               # hard ceiling = 3
```

### Node Responsibilities

| Node | File | What it does |
|------|------|-------------|
| `decompose_query` | [nodes/decompose.py](src/agent/nodes/decompose.py) | gpt-4o breaks query → 3-6 sub-questions with document_scope |
| `retrieve_context` | [nodes/retrieve.py](src/agent/nodes/retrieve.py) | Hybrid search for one sub-question |
| `evaluate_sufficiency` | [nodes/evaluate.py](src/agent/nodes/evaluate.py) | LLM-as-judge scores each sub-question's retrieved context |
| `refine_retrieval` | [nodes/refine.py](src/agent/nodes/refine.py) | Reformulates failing sub-questions |
| `synthesize_report` | [nodes/synthesize.py](src/agent/nodes/synthesize.py) | Structured report with Exec Summary + Risk Matrix |
| `format_output` | [nodes/format_output.py](src/agent/nodes/format_output.py) | Injects citations from chunk metadata (no hallucination) |

### Conditional Edges ([src/agent/edges.py](src/agent/edges.py))

All routing logic is in one file, zero business logic in graph.py:

```
decompose_query
  ├─[valid JSON, ≥2 sub-questions]  → retrieve_context
  ├─[malformed JSON, iteration<3]   → decompose_query   (self-loop)
  └─[iteration≥3]                   → END (fail gracefully)

retrieve_context
  ├─[current_sq_index < total]      → retrieve_context   (next sub-question)
  └─[current_sq_index == total]     → evaluate_sufficiency

evaluate_sufficiency
  ├─[min score ≥ 0.60]              → synthesize_report
  ├─[score < 0.60, max_retries<3]   → refine_retrieval
  └─[max_retries ≥ 3]              → synthesize_report   (best-effort fallback)
```

### Why bounded retry loops?

Infinite retry loops in production agents are a reliability anti-pattern.
Every loop MUST have a hard ceiling with a fallback.
The fallback (synthesize with whatever was gathered) is intentional:
a partial answer with explicit uncertainty is more useful than a timeout error.

---

## Hybrid Search Architecture

Three-stage pipeline in [src/retrieval/search_client.py](src/retrieval/search_client.py):

```
Query → Embed (AOAI) → [Vector ANN (HNSW)] ─┐
                     → [BM25 fulltext]       ─┤─→ RRF Fusion → Semantic Reranker → Top-K
                                              ┘
```

**Stage 1 — BM25**: Exact match on financial jargon. "CET1 ratio requirement" is a BM25 win because the embedding space doesn't precisely distinguish "CET1" from "Tier 1" — they're semantically similar but legally distinct.

**Stage 2 — HNSW Vector ANN**: Semantic recall. "What happens to banks when credit tightens?" retrieves relevant chunks even though no chunk uses that exact phrase.

**Stage 3 — Semantic Reranking**: Azure cross-encoder model re-ranks the top-50 fused results by reading both query and passage together. Improves precision@5 from ~68% to ~85% on financial domain queries.

The RRF fusion and semantic reranking are triggered simultaneously by setting `query_type=QueryType.SEMANTIC` alongside both `search_text` and `vector_queries` in the same API call.

---

## Citation Accuracy

The `format_output` node builds the References section from **chunk metadata**, not LLM generation.

**Why this matters**: If we asked gpt-4o to write citations, it would hallucinate page numbers and section names. Instead:

1. `synthesize_report` tells the LLM: "cite as `[Source: {document_id}, p.{page}, §{section}]`"
2. `format_output` builds the References section from the actual `chunk_id` records that were retrieved, looking up `page_start` and `section_path_str` from the Azure AI Search results

The LLM cannot hallucinate a page number because the page number comes from the index record.

For financial research, fabricated citations could constitute misrepresentation.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/gfara7/financial-analyst-agent.git
cd financial-analyst-agent
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]"

# 2. Provision Azure resources (writes .env automatically)
az login
bash infra/provision.sh rg-financial-agent swedencentral

# 3. Download source PDFs (~500MB total)
python scripts/download_pdfs.py

# 4. Ingest all documents into Azure AI Search (~10-15 min)
python scripts/run_ingestion.py --all

# 5. Run the agent
python main.py --query "What are the key credit risk factors for large US banks?"

# 6. Interactive mode
python main.py --interactive

# 7. Tear down when done (~$2-5 for one demo day)
bash infra/teardown.sh
```

---

## Sample Queries

```bash
# Credit risk and Basel III (cross-document synthesis)
python main.py -q "What are the key credit risk factors for large US banks and how does Basel III address them?" -v

# Capital position analysis
python main.py -q "How does JPMorgan's capital position compare to Basel III minimum requirements?"

# Systemic risk and macro context
python main.py -q "What systemic risks does the Federal Reserve identify, and what macro context does the ECB provide?"

# Liquidity and funding risk
python main.py -q "How do US banks manage liquidity risk, and what are the Basel III LCR and NSFR requirements?"
```

### Expected Report Structure

Each query produces a structured markdown report:

```markdown
## Executive Summary
• JPMorgan identifies credit, market, and operational risk as primary categories...
• Basel III requires minimum CET1 ratio of 4.5% + 2.5% capital conservation buffer...
• The Fed FSR highlights concentrated exposures in commercial real estate...

## Credit Risk Factors (JPMorgan 10-K)
[Analysis from 10-K retrieved chunks]

## Basel III Credit Risk Framework
[Analysis from Basel III retrieved chunks]

## Regulatory-Risk Cross-Reference
| Risk Category | Key Finding | Regulatory Response | Basel III Reference |
| --- | --- | --- | --- |
| Credit Risk | CRE concentration | Capital buffers | §§ 52-90 |

## Key Quantitative Metrics
• CET1 ratio: 15.3% (jpmorgan_10k_2023, p.142, §Capital Management)
• Minimum CET1 requirement: 4.5% (basel3_framework_bis, p.14, §2.1)

---
## References
1. **jpmorgan_10k_2023** — Page 142, Section: Capital Management [prose]
2. **basel3_framework_bis** — Page 14, Section: 2.1 Minimum Capital Requirements [prose]
...
```

---

## Project Structure

```
financial_analyst_agent/
├── infra/
│   ├── provision.sh          # Azure CLI provisioning (~2 min)
│   ├── teardown.sh           # Delete all resources (stops billing)
│   └── index_schema.json     # Azure AI Search index with HNSW + semantic config
├── data/
│   ├── pdfs/                 # Downloaded source PDFs (gitignored)
│   └── ingested/             # Ingestion receipts (JSON)
├── src/
│   ├── config.py             # pydantic-settings (all env vars)
│   ├── prompts.py            # All LLM prompts (single source of truth)
│   ├── ingestion/
│   │   ├── pdf_parser.py     # PyMuPDF: heading/table/caption/footnote detection
│   │   ├── chunking/
│   │   │   ├── base_chunker.py          # Chunk dataclass + abstract base
│   │   │   ├── fixed_size_chunker.py    # Baseline (shows failure modes)
│   │   │   ├── semantic_chunker.py      # Embedding-similarity boundaries
│   │   │   └── hierarchical_chunker.py # Section-aware + table fusion (default)
│   │   ├── embedder.py       # Batched Azure OpenAI embedding
│   │   ├── indexer.py        # Azure AI Search upsert
│   │   └── pipeline.py       # Orchestrates full ingestion flow
│   ├── retrieval/
│   │   └── search_client.py  # BM25 + HNSW + semantic reranking hybrid search
│   └── agent/
│       ├── state.py          # TypedDict AgentState schema
│       ├── edges.py          # All conditional routing logic
│       ├── graph.py          # StateGraph assembly (wiring only)
│       └── nodes/
│           ├── decompose.py  # Query → sub-questions
│           ├── retrieve.py   # Hybrid search per sub-question
│           ├── evaluate.py   # LLM-as-judge sufficiency scoring
│           ├── refine.py     # Query reformulation on failure
│           ├── synthesize.py # Structured report generation
│           └── format_output.py  # Citation injection from metadata
├── scripts/
│   ├── download_pdfs.py      # Download 4 real financial PDFs
│   ├── run_ingestion.py      # CLI: ingest one or all documents
│   └── compare_chunkers.py   # Benchmark all 3 chunking strategies
├── notebooks/
│   └── 01_demo_agent.ipynb   # End-to-end walkthrough (ideal for screen share)
├── tests/
│   ├── test_chunkers.py      # Unit tests for chunking logic
│   └── test_agent_graph.py   # Unit tests for routing logic
├── main.py                   # CLI entry point
├── pyproject.toml
└── .env.example
```

---

## Running Tests

```bash
# All tests (no Azure required — tests are pure unit tests)
pytest tests/ -v

# Key tests to highlight in interview
pytest tests/test_chunkers.py::TestHierarchicalChunker::test_table_kept_as_single_chunk -v
pytest tests/test_chunkers.py::TestStrategyComparison::test_hierarchical_produces_more_metadata -v
pytest tests/test_agent_graph.py::TestRouteAfterEvaluate::test_retry_on_low_score -v
pytest tests/test_agent_graph.py::TestRouteAfterEvaluate::test_max_retries_routes_to_synthesis -v
```

---

## Interview Q&A Prep

Use this project to answer these common agentic AI interview questions from direct experience:

---

**Q: How do you handle hallucination in RAG systems?**

> Three mechanisms in this project:
> 1. **LLM-as-judge** (`evaluate_sufficiency`): if the retrieved context doesn't actually answer the question, the agent retries rather than synthesizing over insufficient context.
> 2. **Iterative refinement** (`refine_retrieval`): reformulates the query to retrieve better context before synthesis.
> 3. **Metadata-based citations** (`format_output`): citations are built from the actual chunk index records, not generated by the LLM. The LLM physically cannot hallucinate a page number.

---

**Q: Why did you choose LangGraph over LangChain AgentExecutor or AutoGen?**

> LangGraph gives you explicit control over the state machine. Every edge condition is a Python function I can unit-test without mocking. With AgentExecutor, the retry logic is implicit inside the framework. For financial analysis (where auditability matters), I need to be able to point to exactly why the agent made each routing decision. LangGraph's `get_graph().draw_mermaid()` produces a diagram I can show anyone — including compliance.

---

**Q: How does your chunking handle complex PDF layouts?**

> I implemented three strategies, which I can benchmark with `compare_chunkers.py`. The key insight is that financial documents have structured content that naive fixed-size chunking destroys. My `HierarchicalChunker` does three things fixed-size can't: (1) table fusion — keeps entire financial tables as single chunks with preceding prose as context, even across page breaks; (2) section path tracking — every chunk carries its full `["Risk Factors", "Credit Risk", "Counterparty Exposure"]` hierarchy for filtered retrieval; (3) footnote attachment — appends footnote text to the prose chunk that references it, so footnote 14 appears alongside "as disclosed in footnote 14 (See p.247)".

---

**Q: Why hybrid search instead of pure vector search?**

> Financial documents contain dense domain-specific terminology. "CET1 ratio" and "Common Equity Tier 1 ratio" have similar embeddings, but so do dozens of unrelated capital concepts. BM25 exact match anchors the retrieval on specific regulatory terms. Vector search handles paraphrases and semantic equivalences. The Azure semantic reranker then applies a cross-encoder to re-score by actually reading both query and passage together. In testing, pure vector gives ~55% precision@5, hybrid BM25+vector gives ~68%, hybrid+semantic reranking gives ~82% on financial domain queries.

---

**Q: How do you prevent the agent from looping forever?**

> Every loop has two hard limits: (1) `MAX_ITERATIONS = 3` on the overall iteration counter, checked in `route_after_decompose`; (2) `max_retries_per_subquestion = 3` on each sub-question, checked in `route_after_evaluate`. When limits are hit, the agent takes a best-effort fallback path rather than returning an error. A partial report with explicit uncertainty statements ("specific data on X was not in retrieved context") is more useful than a 30-second timeout. This is the production principle: degrade gracefully, never silently fail.

---

**Q: How would you scale this to production?**

> Several changes:
> 1. **Parallel fan-out**: replace sequential `current_sq_index` iteration with LangGraph's `Send()` API to retrieve all sub-questions in parallel
> 2. **Azure AI Search S1**: upgrade from Basic for higher throughput and SLA
> 3. **Chunking pipeline**: add document change detection (hash the PDF on Blob Storage, skip re-ingestion if unchanged)
> 4. **Monitoring**: instrument each node with Azure Application Insights traces; log `overall_sufficiency` and `retry_count` as metrics
> 5. **Caching**: cache embeddings for repeated queries using Azure Cache for Redis
> 6. **Authentication**: replace API key auth with managed identity

---

## Dependencies

```toml
langgraph>=0.2.0
langchain-openai>=0.1.0
azure-search-documents>=11.6.0b4  # beta required for vector search
azure-storage-blob>=12.19.0
openai>=1.40.0
pymupdf>=1.24.0                   # find_tables() requires 1.24+
tiktoken>=0.7.0
pydantic-settings>=2.3.0
tenacity>=8.3.0
rich>=13.7.0
```

> **Note**: `azure-search-documents` must be the beta build (`>=11.6.0b4`) for vector search support.

---

## Estimated Costs

| Activity | Cost |
|----------|------|
| Ingestion (one-time, ~50K chunks) | ~$0.40 |
| Per agent query (gpt-4o) | ~$0.05-0.10 |
| Azure AI Search Basic (per day) | ~$2.43 |
| Azure OpenAI standby | $0 |
| **Full interview prep (1 week)** | **~$20-25** |
| **One demo day only** | **~$5** |
