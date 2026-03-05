"""
src/ingestion/pdf_parser.py
────────────────────────────
Structured PDF extraction using PyMuPDF (fitz).

Why PyMuPDF over pdfplumber?
  - Direct access to font metadata (size, flags) → reliable heading detection
  - find_tables() API (stable since 1.23) → avoids regex table heuristics
  - Image rectangle coordinates → figure caption detection

Block types produced:
  heading  — section title at any level (H1-H4)
  prose    — regular paragraph text
  table    — Markdown-formatted table string (via tabulate)
  caption  — figure/chart/exhibit caption text
  footnote — footnote text from page bottom region
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF


BlockType = Literal["heading", "prose", "table", "caption", "footnote"]

# Minimum fraction of page height that defines the "footnote zone" (bottom 12%)
FOOTNOTE_ZONE_FRACTION = 0.88

# Caption trigger words (case-insensitive)
CAPTION_TRIGGERS = re.compile(
    r"^(figure|fig\.|chart|exhibit|table|panel|source:|note:)\s+",
    re.IGNORECASE,
)

# Superscript footnote reference patterns in body text
FOOTNOTE_REF_PATTERN = re.compile(r"(\d+)\s*(?=\s|$|[,;.])")


@dataclass
class PageBlock:
    """A classified text unit from a single PDF page."""

    page_num: int                   # 1-indexed
    block_type: BlockType
    text: str                       # Raw text or Markdown for tables
    font_size: float                # Dominant font size in block
    bbox: tuple                     # (x0, y0, x1, y1) in page points
    heading_level: int              # 0 = not heading; 1-4 = H1-H4
    is_table_continuation: bool     # True if table started on the previous page
    table_id: str                   # Non-empty for table blocks (for page-span fusion)
    footnote_refs: list[str]        # Footnote numbers referenced in this prose block


@dataclass
class ParsedDocument:
    """Full structured extraction result from a PDF."""

    document_id: str
    document_title: str
    total_pages: int
    blocks: list[PageBlock]             # Ordered by page → y-position
    toc: list[dict]                     # [{level, title, page}]
    footnotes_by_page: dict[int, list[str]]  # page_num -> list of footnote texts
    median_font_size: float


class PDFParser:
    """
    Structured extraction from financial PDFs.

    Design decisions:
      - Heading detection uses font size RELATIVE to the document median,
        not absolute values. This adapts to different document styles.
      - Tables are converted to Markdown immediately; the raw bbox is kept
        for page-span detection.
      - Footnotes are detected by vertical position (bottom 12% of page)
        AND by having smaller-than-median font size.
    """

    def parse(self, pdf_path: str | Path, document_id: str) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        # ── Pass 1: Collect all font sizes for document-level median ────────
        all_font_sizes: list[float] = []
        for page in doc:
            for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
                if block["type"] == 0:  # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > 4:  # ignore tiny artifacts
                                all_font_sizes.append(span["size"])

        median_fs = statistics.median(all_font_sizes) if all_font_sizes else 12.0

        # ── Extract document title from metadata or first large text ─────────
        meta = doc.metadata or {}
        doc_title = meta.get("title", "") or pdf_path.stem.replace("_", " ").title()

        # ── Extract Table of Contents ─────────────────────────────────────────
        toc_raw = doc.get_toc(simple=False)  # [[level, title, page, dest], ...]
        toc = [{"level": t[0], "title": t[1], "page": t[2]} for t in toc_raw]

        # ── Pass 2: Page-by-page block extraction ─────────────────────────────
        all_blocks: list[PageBlock] = []
        footnotes_by_page: dict[int, list[str]] = {}
        table_counter = 0
        pending_table_bboxes: dict[int, tuple] = {}  # table_id -> bottom bbox

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_height = page.rect.height
            footnote_y_threshold = page_height * FOOTNOTE_ZONE_FRACTION

            # ── Detect tables on this page ────────────────────────────────────
            table_regions: list[dict] = []
            try:
                tables = page.find_tables()
                for tbl in tables:
                    table_counter += 1
                    table_id = f"tbl_{table_counter}"
                    md_text = _table_to_markdown(tbl)
                    if not md_text.strip():
                        continue

                    # Check if this is a continuation of a table from the previous page
                    is_continuation = _is_table_continuation(
                        tbl.bbox, page_num, pending_table_bboxes
                    )

                    block = PageBlock(
                        page_num=page_num,
                        block_type="table",
                        text=md_text,
                        font_size=median_fs,
                        bbox=tbl.bbox,
                        heading_level=0,
                        is_table_continuation=is_continuation,
                        table_id=table_id,
                        footnote_refs=[],
                    )
                    all_blocks.append(block)
                    table_regions.append({"bbox": tbl.bbox, "table_id": table_id})

                    # Track for next-page continuation detection
                    # If table extends to within 5% of page bottom, it may continue
                    if tbl.bbox[3] > page_height * 0.90:
                        pending_table_bboxes[table_id] = tbl.bbox
                    else:
                        pending_table_bboxes.pop(table_id, None)

            except Exception:
                # find_tables() may fail on some PDFs; degrade gracefully
                pass

            # ── Get image rectangles for caption detection ────────────────────
            image_bboxes: list[tuple] = []
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                image_bboxes.extend([r for r in rects])

            # ── Process text blocks ───────────────────────────────────────────
            footnote_texts: list[str] = []
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in text_dict["blocks"]:
                if block["type"] != 0:  # skip non-text (images)
                    continue

                block_bbox = tuple(block["bbox"])

                # Skip if this bbox is inside a detected table region
                if _bbox_overlaps_any(block_bbox, [r["bbox"] for r in table_regions]):
                    continue

                # Collect text and dominant font size from block
                block_text, dominant_fs = _extract_block_text(block)
                if not block_text.strip():
                    continue

                # ── Classify block type ───────────────────────────────────────
                block_y_center = (block_bbox[1] + block_bbox[3]) / 2

                # Footnote zone (bottom of page + small font)
                if (
                    block_y_center > footnote_y_threshold
                    and dominant_fs < median_fs * 0.85
                ):
                    footnote_texts.append(block_text.strip())
                    continue

                # Figure/chart caption
                if CAPTION_TRIGGERS.match(block_text.strip()) or _is_below_image(
                    block_bbox, image_bboxes
                ):
                    pb = PageBlock(
                        page_num=page_num,
                        block_type="caption",
                        text=block_text.strip(),
                        font_size=dominant_fs,
                        bbox=block_bbox,
                        heading_level=0,
                        is_table_continuation=False,
                        table_id="",
                        footnote_refs=[],
                    )
                    all_blocks.append(pb)
                    continue

                # Heading detection: font significantly larger than median
                heading_level = _detect_heading_level(dominant_fs, median_fs, block_text)
                if heading_level > 0:
                    pb = PageBlock(
                        page_num=page_num,
                        block_type="heading",
                        text=block_text.strip(),
                        font_size=dominant_fs,
                        bbox=block_bbox,
                        heading_level=heading_level,
                        is_table_continuation=False,
                        table_id="",
                        footnote_refs=[],
                    )
                    all_blocks.append(pb)
                    continue

                # Regular prose
                footnote_refs = FOOTNOTE_REF_PATTERN.findall(block_text)
                pb = PageBlock(
                    page_num=page_num,
                    block_type="prose",
                    text=block_text.strip(),
                    font_size=dominant_fs,
                    bbox=block_bbox,
                    heading_level=0,
                    is_table_continuation=False,
                    table_id="",
                    footnote_refs=footnote_refs,
                )
                all_blocks.append(pb)

            if footnote_texts:
                footnotes_by_page[page_num] = footnote_texts

        total_pages = len(doc)
        doc.close()

        return ParsedDocument(
            document_id=document_id,
            document_title=doc_title,
            total_pages=total_pages,
            blocks=all_blocks,
            toc=toc,
            footnotes_by_page=footnotes_by_page,
            median_font_size=median_fs,
        )


# ── Helper Functions ───────────────────────────────────────────────────────────

def _extract_block_text(block: dict) -> tuple[str, float]:
    """Extract plain text and dominant font size from a PyMuPDF text block dict."""
    lines = []
    font_sizes: list[float] = []
    for line in block["lines"]:
        line_text = ""
        for span in line["spans"]:
            line_text += span["text"]
            font_sizes.append(span["size"])
        lines.append(line_text)
    text = "\n".join(lines)
    dominant_fs = statistics.median(font_sizes) if font_sizes else 12.0
    return text, dominant_fs


def _detect_heading_level(font_size: float, median_fs: float, text: str) -> int:
    """
    Determine heading level (1-4) based on font size ratio to median.
    Returns 0 if not a heading.

    Thresholds (tunable):
      H1: >= 1.6x median
      H2: >= 1.3x median
      H3: >= 1.15x median
      H4: >= 1.05x median AND text looks like a section number
    """
    ratio = font_size / median_fs if median_fs > 0 else 1.0
    stripped = text.strip()

    if ratio >= 1.60:
        return 1
    if ratio >= 1.30:
        return 2
    if ratio >= 1.15:
        return 3
    # H4: slightly larger, AND starts with a numbered section pattern like "3.4.2"
    section_num_pattern = re.compile(r"^\d+(\.\d+){1,3}\s+\w")
    if ratio >= 1.05 and section_num_pattern.match(stripped):
        return 4

    # Also treat ALL-CAPS short lines as headings even at normal font size
    if stripped.isupper() and 3 <= len(stripped.split()) <= 8:
        return 3

    return 0


def _table_to_markdown(tbl) -> str:
    """Convert a PyMuPDF Table object to a Markdown table string."""
    try:
        from tabulate import tabulate

        rows = tbl.extract()
        if not rows or len(rows) < 2:
            return ""

        # First row as headers if it looks like a header
        headers = rows[0]
        body = rows[1:]

        # Replace None cells with empty string
        headers = [str(h) if h is not None else "" for h in headers]
        body = [[str(c) if c is not None else "" for c in row] for row in body]

        return tabulate(body, headers=headers, tablefmt="pipe")
    except Exception:
        # Fallback: simple pipe-delimited
        rows = tbl.extract()
        if not rows:
            return ""
        lines = []
        for row in rows:
            cells = [str(c) if c is not None else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _bbox_overlaps_any(bbox: tuple, regions: list[tuple]) -> bool:
    """Return True if bbox significantly overlaps any region in the list."""
    x0, y0, x1, y1 = bbox
    for rx0, ry0, rx1, ry1 in regions:
        # Check for >30% overlap
        inter_x = max(0, min(x1, rx1) - max(x0, rx0))
        inter_y = max(0, min(y1, ry1) - max(y0, ry0))
        inter_area = inter_x * inter_y
        bbox_area = max(1, (x1 - x0) * (y1 - y0))
        if inter_area / bbox_area > 0.30:
            return True
    return False


def _is_below_image(block_bbox: tuple, image_bboxes: list) -> bool:
    """True if this text block is directly below an image (within 20pt)."""
    x0, y0, x1, y1 = block_bbox
    for ix0, iy0, ix1, iy1 in image_bboxes:
        # Horizontally overlapping and vertically close (below image)
        h_overlap = min(x1, ix1) - max(x0, ix0)
        if h_overlap > 0 and 0 <= y0 - iy1 <= 20:
            return True
    return False


def _is_table_continuation(
    table_bbox: tuple,
    page_num: int,
    pending_table_bboxes: dict[str, tuple],
) -> bool:
    """
    Check if this table appears to be continuing from the previous page.
    Heuristic: if any pending table from the previous page extended near the
    bottom, and this table starts near the top of the current page.
    """
    if not pending_table_bboxes:
        return False
    # Table starts in top 15% of page
    if table_bbox[1] < 100:  # within first 100pt (typically ~top 14% on A4/letter)
        return True
    return False
