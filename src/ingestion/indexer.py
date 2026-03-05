"""
src/ingestion/indexer.py
────────────────────────
Uploads embedded chunks to Azure AI Search.

Design decisions:
  - Uses merge-or-upload (upsert) so re-ingesting a document is idempotent
  - Batches uploads in groups of 1000 (Search API limit per request)
  - Creates the index from index_schema.json if it doesn't already exist
  - Returns a summary dict for the ingestion receipt
"""

from __future__ import annotations

import json
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.config import settings
from .chunking.base_chunker import Chunk

# Maximum documents per batch (Azure AI Search limit)
_UPLOAD_BATCH_SIZE = 1000

# Path to the index schema JSON
_SCHEMA_PATH = Path(__file__).parent.parent.parent / "infra" / "index_schema.json"


class Indexer:
    """
    Manages Azure AI Search index creation and document upload.
    """

    def __init__(self):
        credential = AzureKeyCredential(settings.azure_search_key)
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=credential,
        )
        self._search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=credential,
        )

    def ensure_index_exists(self) -> None:
        """Create the Azure AI Search index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self._index_client.list_indexes()]
        if settings.azure_search_index_name in existing_indexes:
            return

        if not _SCHEMA_PATH.exists():
            raise FileNotFoundError(
                f"Index schema not found at {_SCHEMA_PATH}. "
                "Run infra/provision.sh first or create infra/index_schema.json."
            )

        with open(_SCHEMA_PATH) as f:
            schema_dict = json.load(f)

        # Rename if the index name in settings differs from schema
        schema_dict["name"] = settings.azure_search_index_name

        index = SearchIndex.from_dict(schema_dict)
        self._index_client.create_index(index)
        print(f"  Created index: {settings.azure_search_index_name}")

    def upload_chunks(
        self, chunks: list[Chunk], show_progress: bool = True
    ) -> dict:
        """
        Upload chunks to the search index using merge-or-upload (upsert).

        Returns a summary: {"uploaded": N, "failed": M}
        """
        # Filter chunks that have embeddings
        ready = [c for c in chunks if c.embedding is not None]
        skipped = len(chunks) - len(ready)
        if skipped > 0:
            print(f"  Skipping {skipped} chunks without embeddings")

        if not ready:
            return {"uploaded": 0, "failed": 0}

        # Serialize to search documents
        documents = [c.to_search_document() for c in ready]

        # Upload in batches
        batches = [
            documents[i : i + _UPLOAD_BATCH_SIZE]
            for i in range(0, len(documents), _UPLOAD_BATCH_SIZE)
        ]

        uploaded = 0
        failed = 0

        with tqdm(
            total=len(ready),
            desc="  Indexing",
            unit="chunks",
            disable=not show_progress,
        ) as pbar:
            for batch in batches:
                result = self._upload_batch(batch)
                batch_uploaded = sum(1 for r in result if r.succeeded)
                batch_failed = sum(1 for r in result if not r.succeeded)
                uploaded += batch_uploaded
                failed += batch_failed
                pbar.update(len(batch))

        return {"uploaded": uploaded, "failed": failed}

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a given document_id.
        Used before re-ingesting to prevent duplicates.
        Returns count of deleted chunks.
        """
        # Search for all chunks from this document
        results = self._search_client.search(
            search_text="*",
            filter=f"document_id eq '{document_id}'",
            select=["chunk_id"],
            top=10000,
        )
        chunk_ids = [{"chunk_id": r["chunk_id"]} for r in results]
        if not chunk_ids:
            return 0

        self._search_client.delete_documents(documents=chunk_ids)
        return len(chunk_ids)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _upload_batch(self, documents: list[dict]):
        return self._search_client.merge_or_upload_documents(documents=documents)
