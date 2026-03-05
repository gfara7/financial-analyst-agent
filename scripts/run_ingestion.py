"""
scripts/run_ingestion.py
─────────────────────────
CLI script to run the document ingestion pipeline.

Usage:
    # Ingest all 4 documents
    python scripts/run_ingestion.py --all

    # Ingest a specific document
    python scripts/run_ingestion.py --pdf data/pdfs/jpmorgan_10k_2023.pdf --type 10k --id jpmorgan_10k_2023

    # Re-ingest (deletes existing chunks first)
    python scripts/run_ingestion.py --all --force
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.ingestion.pipeline import IngestionPipeline, CHUNKING_STRATEGIES

console = Console()

# ── Document registry (matches download_pdfs.py) ──────────────────────────────
ALL_DOCUMENTS = [
    {
        "id": "jpmorgan_10k_2023",
        "filename": "jpmorgan_10k_2023.pdf",
        "type": "10k",
    },
    {
        "id": "fed_fsr_2023",
        "filename": "fed_fsr_2023.pdf",
        "type": "fsr",
    },
    {
        "id": "basel3_framework_bis",
        "filename": "basel3_framework_bis.pdf",
        "type": "basel3",
    },
    {
        "id": "ecb_economic_bulletin_2023",
        "filename": "ecb_economic_bulletin_2023.pdf",
        "type": "ecb_bulletin",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest financial PDFs into Azure AI Search"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Ingest all 4 documents")
    group.add_argument("--pdf", type=str, help="Path to a single PDF file")
    parser.add_argument(
        "--type",
        choices=list(CHUNKING_STRATEGIES.keys()),
        help="Document type (required with --pdf)",
    )
    parser.add_argument("--id", type=str, help="Document ID (required with --pdf)")
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete existing chunks and re-ingest even if receipt exists (default: False)",
    )
    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent

    pipeline = IngestionPipeline()
    receipts = []

    if args.all:
        console.print("\n[bold]Ingesting all 4 documents...[/bold]")
        for doc in ALL_DOCUMENTS:
            pdf_path = project_root / "data" / "pdfs" / doc["filename"]
            if not pdf_path.exists():
                console.print(
                    f"[red]Missing: {pdf_path}[/red] — run scripts/download_pdfs.py first"
                )
                continue
            receipt_path = project_root / "data" / "ingested" / f"{doc['id']}.json"
            if receipt_path.exists() and not args.force:
                console.print(f"\n[dim]Skipping {doc['id']} — already ingested (use --force to re-ingest)[/dim]")
                continue
            receipt = pipeline.ingest(
                pdf_path=pdf_path,
                document_type=doc["type"],
                document_id=doc["id"],
                delete_existing=args.force,
            )
            receipts.append(receipt)
    else:
        if not args.type or not args.id:
            console.print("[red]--type and --id are required when using --pdf[/red]")
            sys.exit(1)
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            console.print(f"[red]File not found: {pdf_path}[/red]")
            sys.exit(1)
        receipt = pipeline.ingest(
            pdf_path=pdf_path,
            document_type=args.type,
            document_id=args.id,
            delete_existing=args.force,
        )
        receipts.append(receipt)

    # ── Summary table ──────────────────────────────────────────────────────────
    if receipts:
        console.print("\n")
        table = Table(title="Ingestion Summary", show_header=True, header_style="bold green")
        table.add_column("Document ID", style="cyan")
        table.add_column("Type")
        table.add_column("Pages")
        table.add_column("Chunks")
        table.add_column("Tokens")
        table.add_column("Strategy")
        table.add_column("Time (s)")

        for r in receipts:
            table.add_row(
                r["document_id"],
                r["document_type"],
                str(r["total_pages"]),
                str(r["chunks_uploaded"]),
                f"{r['total_tokens']:,}",
                r["chunking_strategy"],
                str(r["timing"]["total_s"]),
            )

        console.print(table)
        console.print("\n[green]Ingestion complete.[/green] Run the agent:")
        console.print(
            '  [cyan]python main.py --query "What are the key credit risk factors for large US banks?"[/cyan]'
        )


if __name__ == "__main__":
    main()
