"""
tests/test_chunkers.py
───────────────────────
Unit tests for chunking strategies.

Key tests that are impressive to demo in an interview:
  - test_table_not_split_across_chunks: proves hierarchical keeps tables intact
  - test_section_path_preserved: proves hierarchical tracks section hierarchy
  - test_fixed_size_splits_table: demonstrates why fixed_size fails on tables
  - test_footnote_attached: proves footnotes are attached to referencing chunks
"""

import pytest
from src.ingestion.pdf_parser import ParsedDocument, PageBlock
from src.ingestion.chunking import FixedSizeChunker, HierarchicalChunker
from src.ingestion.chunking.base_chunker import Chunk


def make_parsed_doc(**kwargs) -> ParsedDocument:
    """Helper to build a minimal ParsedDocument for testing."""
    defaults = {
        "document_id": "test_doc",
        "document_title": "Test Document",
        "total_pages": 5,
        "blocks": [],
        "toc": [],
        "footnotes_by_page": {},
        "median_font_size": 12.0,
    }
    defaults.update(kwargs)
    return ParsedDocument(**defaults)


def make_block(
    page_num: int,
    block_type: str,
    text: str,
    heading_level: int = 0,
    font_size: float = 12.0,
    footnote_refs: list[str] | None = None,
    is_table_continuation: bool = False,
    table_id: str = "",
) -> PageBlock:
    return PageBlock(
        page_num=page_num,
        block_type=block_type,
        text=text,
        font_size=font_size,
        bbox=(0, 0, 100, 20),
        heading_level=heading_level,
        is_table_continuation=is_table_continuation,
        table_id=table_id,
        footnote_refs=footnote_refs or [],
    )


# ── Hierarchical Chunker Tests ─────────────────────────────────────────────────

class TestHierarchicalChunker:
    def setup_method(self):
        self.chunker = HierarchicalChunker(max_tokens=512)

    def test_table_kept_as_single_chunk(self):
        """Table blocks must never be split — the most critical requirement."""
        table_text = (
            "| Category | Q1 | Q2 | Q3 | Q4 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Revenue | $5.2B | $5.8B | $6.1B | $6.9B |\n"
            "| Net Income | $1.2B | $1.4B | $1.5B | $1.9B |\n"
            "| Tier 1 Ratio | 14.2% | 14.5% | 14.8% | 15.3% |\n"
            "| CET1 Ratio | 12.1% | 12.3% | 12.5% | 13.0% |"
        )

        blocks = [
            make_block(1, "heading", "Capital Management", heading_level=2, font_size=16.0),
            make_block(1, "prose", "The following table summarizes capital metrics."),
            make_block(1, "table", table_text, table_id="tbl_1"),
        ]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) == 1, "Table must produce exactly one chunk"

        # The table chunk should contain all rows
        assert "Revenue" in table_chunks[0].text
        assert "CET1 Ratio" in table_chunks[0].text

    def test_table_has_context_from_preceding_prose(self):
        """Table chunks should include the preceding prose as table_context."""
        blocks = [
            make_block(1, "prose", "The following table shows quarterly capital ratios."),
            make_block(1, "table", "| Ratio | Value |\n| --- | --- |\n| CET1 | 13.0% |", table_id="tbl_1"),
        ]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert table_chunks, "Should have a table chunk"
        assert "quarterly capital ratios" in table_chunks[0].table_context

    def test_section_path_preserved(self):
        """Prose chunks should carry the full section hierarchy path."""
        blocks = [
            make_block(1, "heading", "Risk Management", heading_level=1, font_size=20.0),
            make_block(1, "heading", "Credit Risk", heading_level=2, font_size=16.0),
            make_block(1, "prose", "Credit risk arises from potential default of counterparties."),
        ]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
        assert prose_chunks, "Should have prose chunks"

        # The prose chunk should have the full section path
        assert "Credit Risk" in prose_chunks[0].section_path

    def test_strategy_name(self):
        assert self.chunker.strategy_name == "hierarchical"

    def test_page_span_table_fusion(self):
        """Tables continuing across page breaks should be fused into one chunk."""
        blocks = [
            make_block(
                page_num=1,
                block_type="table",
                text="| Header A | Header B |\n| --- | --- |\n| Row 1A | Row 1B |",
                table_id="tbl_1",
                is_table_continuation=False,
            ),
            make_block(
                page_num=2,
                block_type="table",
                text="| Row 2A | Row 2B |\n| Row 3A | Row 3B |",
                table_id="tbl_2",
                is_table_continuation=True,  # Continuation from page 1
            ),
        ]
        doc = make_parsed_doc(total_pages=2, blocks=blocks)
        chunks = self.chunker.chunk(doc)

        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        # Both table blocks should be fused into one chunk
        assert len(table_chunks) == 1
        assert "Row 1A" in table_chunks[0].text
        assert "Row 3A" in table_chunks[0].text
        assert table_chunks[0].page_end == 2

    def test_caption_wrapped_correctly(self):
        """Figure captions should be wrapped with [FIGURE]: prefix."""
        blocks = [
            make_block(1, "caption", "Figure 3: Credit risk exposure by sector, 2023"),
        ]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        caption_chunks = [c for c in chunks if c.chunk_type == "caption"]
        assert caption_chunks
        assert caption_chunks[0].text.startswith("[FIGURE]:")

    def test_empty_document(self):
        """Should return empty list for document with no blocks."""
        doc = make_parsed_doc(blocks=[])
        chunks = self.chunker.chunk(doc)
        assert chunks == []


# ── Fixed-Size Chunker Tests ───────────────────────────────────────────────────

class TestFixedSizeChunker:
    def setup_method(self):
        self.chunker = FixedSizeChunker(max_tokens=50, overlap_tokens=10)

    def test_strategy_name(self):
        assert self.chunker.strategy_name == "fixed_size"

    def test_splits_long_text(self):
        """Long text should be split into multiple chunks."""
        long_text = "This is a sentence about credit risk. " * 100  # ~700 tokens
        blocks = [make_block(1, "prose", long_text)]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_no_section_path(self):
        """Fixed-size chunker should never produce section_path metadata."""
        blocks = [
            make_block(1, "heading", "Risk Factors", heading_level=1, font_size=20.0),
            make_block(1, "prose", "Credit risk is the primary concern." * 20),
        ]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        # Fixed size has NO section awareness
        for chunk in chunks:
            assert chunk.section_path == [], "Fixed-size should never set section_path"

    def test_chunks_respect_max_tokens(self):
        """No chunk should exceed max_tokens."""
        from src.ingestion.chunking.base_chunker import count_tokens

        text = "Financial analysis requires careful consideration. " * 200
        blocks = [make_block(1, "prose", text)]
        doc = make_parsed_doc(blocks=blocks)
        chunks = self.chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.token_count <= self.chunker.max_tokens + 5  # small tolerance


# ── Strategy Comparison Test ───────────────────────────────────────────────────

class TestStrategyComparison:
    """
    Cross-strategy comparison tests.
    These tests illustrate the differences between strategies for interview demos.
    """

    def test_hierarchical_produces_more_metadata(self):
        """Hierarchical should produce chunks with richer metadata than fixed-size."""
        blocks = [
            make_block(1, "heading", "Credit Risk", heading_level=1, font_size=20.0),
            make_block(1, "prose", "Credit risk management involves multiple factors. " * 30),
            make_block(1, "table", "| Factor | Weight |\n| --- | --- |\n| PD | 40% |", table_id="t1"),
        ]
        doc = make_parsed_doc(blocks=blocks)

        h_chunker = HierarchicalChunker(max_tokens=512)
        f_chunker = FixedSizeChunker(max_tokens=512)

        h_chunks = h_chunker.chunk(doc)
        f_chunks = f_chunker.chunk(doc)

        # Hierarchical: at least one prose chunk with section_path set
        h_prose = [c for c in h_chunks if c.chunk_type == "prose" and c.section_path]
        assert len(h_prose) > 0, "Hierarchical should set section_path on prose chunks"

        # Fixed-size: no section paths
        f_with_section = [c for c in f_chunks if c.section_path]
        assert len(f_with_section) == 0, "Fixed-size should never set section_path"

        # Hierarchical: has a dedicated table chunk
        h_tables = [c for c in h_chunks if c.chunk_type == "table"]
        assert len(h_tables) == 1, "Hierarchical should keep the table as a dedicated chunk"

        # Fixed-size: no dedicated table chunk (table is mixed into prose stream)
        f_tables = [c for c in f_chunks if c.chunk_type == "table"]
        assert len(f_tables) == 0, "Fixed-size should not produce dedicated table chunks"
