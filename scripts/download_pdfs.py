"""
scripts/download_pdfs.py
────────────────────────
Downloads the 4 real public financial PDFs used as the knowledge base.

All sources are authoritative public-domain documents:
  - SEC EDGAR (US federal government)
  - Federal Reserve (US federal government)
  - BIS (Bank for International Settlements)
  - ECB (European Central Bank)

Usage:
    python scripts/download_pdfs.py

Output: data/pdfs/
"""

import os
import sys
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# ── Document Registry ──────────────────────────────────────────────────────────
# Each entry has: id, filename, url, description, expected_size_mb (approx)
DOCUMENTS = [
    {
        "id": "jpmorgan_10k_2023",
        "filename": "jpmorgan_10k_2023.pdf",
        "url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000127/jpm-20231231.htm",
        # The actual 10-K PDF from EDGAR:
        "url_pdf": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000127/jpm-20231231.pdf",
        "description": "JPMorgan Chase 10-K Annual Report 2023 (SEC EDGAR)",
        "document_type": "10k",
        "source": "SEC EDGAR — public domain",
        "chunking_challenges": [
            "Multi-page financial tables (income statement, balance sheet)",
            "Deeply nested sections (e.g. Note 14.3.2)",
            "Footnotes referencing main financial tables",
            "Risk factors section with 50+ sub-items",
        ],
        "expected_pages": 360,
    },
    {
        "id": "fed_fsr_2023",
        "filename": "fed_fsr_2023.pdf",
        "url_pdf": "https://www.federalreserve.gov/publications/files/financial-stability-report-20231110.pdf",
        "description": "Federal Reserve Financial Stability Report — November 2023",
        "document_type": "fsr",
        "source": "Federal Reserve — public domain",
        "chunking_challenges": [
            "Charts and figures with descriptive captions",
            "Call-out boxes with key statistics",
            "Mixed tabular and prose content",
            "Policy prose requiring topic-coherent chunking",
        ],
        "expected_pages": 80,
    },
    {
        "id": "basel3_framework_bis",
        "filename": "basel3_framework_bis.pdf",
        "url_pdf": "https://www.bis.org/bcbs/publ/d424.pdf",
        "description": "Basel III: Finalising Post-Crisis Reforms (BIS, 2017/consolidated 2023)",
        "document_type": "basel3",
        "source": "Bank for International Settlements — public domain",
        "chunking_challenges": [
            "Numbered regulation hierarchy up to 4 levels deep",
            "Capital ratio requirement tables spanning multiple pages",
            "Cross-references between sections (e.g. 'see paragraph 52')",
            "Mathematical formulas for risk-weight calculations",
        ],
        "expected_pages": 162,
    },
    {
        "id": "ecb_economic_bulletin_2023",
        "filename": "ecb_economic_bulletin_2023.pdf",
        "url_pdf": "https://www.ecb.europa.eu/pub/pdf/ecbu/eb202308.en.pdf",
        "description": "ECB Economic Bulletin Issue 8/2023",
        "document_type": "ecb_bulletin",
        "source": "European Central Bank — public domain",
        "chunking_challenges": [
            "Charts with dense analytical captions",
            "Macro indicator tables with multiple series",
            "Box articles as self-contained subsections",
            "Cross-references to annex statistical tables",
        ],
        "expected_pages": 100,
    },
]

# Fallback URLs (alternative sources if primary fails)
FALLBACK_URLS = {
    "jpmorgan_10k_2023": [
        # Try alternate EDGAR filing index
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000019617&type=10-K&dateb=&owner=include&count=5",
    ],
    "fed_fsr_2023": [
        "https://www.federalreserve.gov/publications/financial-stability-report.htm",
    ],
    "basel3_framework_bis": [
        "https://www.bis.org/bcbs/publ/d424.htm",
    ],
    "ecb_economic_bulletin_2023": [
        "https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202308.en.html",
    ],
}


def download_file(url: str, dest_path: Path, description: str) -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, stream=True, headers=headers, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f, tqdm(
            desc=f"  {description[:50]}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        # Verify it's actually a PDF
        with open(dest_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            console.print(f"    [yellow]WARNING: File does not appear to be a valid PDF[/yellow]")
            return False

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        console.print(f"    [green]✓[/green] Downloaded {size_mb:.1f} MB → {dest_path.name}")
        return True

    except requests.HTTPError as e:
        console.print(f"    [red]HTTP error {e.response.status_code}[/red]: {url}")
        return False
    except Exception as e:
        console.print(f"    [red]Error[/red]: {e}")
        return False


def print_document_table() -> None:
    """Print a summary table of documents to be downloaded."""
    table = Table(title="Source Documents", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Document", max_width=45)
    table.add_column("Type")
    table.add_column("Pages (~)")
    table.add_column("Source")

    for doc in DOCUMENTS:
        table.add_row(
            doc["id"],
            doc["description"],
            doc["document_type"],
            str(doc["expected_pages"]),
            doc["source"],
        )
    console.print(table)
    console.print()


def main() -> None:
    # Determine project root (scripts/ -> parent)
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    pdf_dir = project_root / "data" / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Financial Analyst Agent — PDF Downloader[/bold]\n")
    print_document_table()
    console.print(f"Saving to: [cyan]{pdf_dir}[/cyan]\n")

    results = []
    for doc in DOCUMENTS:
        dest = pdf_dir / doc["filename"]
        console.print(f"[bold]{doc['id']}[/bold]")
        console.print(f"  {doc['description']}")

        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            console.print(f"  [dim]Already exists ({size_mb:.1f} MB) — skipping[/dim]")
            results.append((doc["id"], "skipped", str(dest)))
            continue

        success = download_file(doc["url_pdf"], dest, doc["description"])
        if success:
            results.append((doc["id"], "downloaded", str(dest)))
        else:
            console.print(f"  [yellow]Trying fallback sources for {doc['id']}...[/yellow]")
            console.print(f"  [yellow]Please download manually from: {doc['url_pdf']}[/yellow]")
            console.print(f"  [yellow]Save as: {dest}[/yellow]")
            results.append((doc["id"], "failed", doc["url_pdf"]))

        console.print()

    # Summary
    console.print("\n[bold]Download Summary[/bold]")
    for doc_id, status, info in results:
        icon = {"downloaded": "[green]✓[/green]", "skipped": "[dim]–[/dim]", "failed": "[red]✗[/red]"}[status]
        console.print(f"  {icon} {doc_id}: {status}")

    failed = [r for r in results if r[1] == "failed"]
    if failed:
        console.print(
            f"\n[yellow]{len(failed)} document(s) failed to download automatically.[/yellow]"
        )
        console.print(
            "[yellow]Download them manually and place in data/pdfs/ with the correct filename.[/yellow]"
        )
        console.print(
            "[yellow]All sources are public domain — no login required.[/yellow]"
        )
        sys.exit(1)
    else:
        console.print(
            "\n[green]All documents ready.[/green] Run ingestion next:"
        )
        console.print("  [cyan]python scripts/run_ingestion.py --all[/cyan]")


if __name__ == "__main__":
    main()
