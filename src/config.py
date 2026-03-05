"""
src/config.py
─────────────
Central configuration via pydantic-settings.
All settings are loaded from environment variables or a .env file.
Import `settings` anywhere in the project:

    from src.config import settings
    print(settings.azure_openai_endpoint)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure OpenAI ───────────────────────────────────────────────────────────
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_llm_deployment: str = "gpt-4o"

    # ── Azure AI Search ────────────────────────────────────────────────────────
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index_name: str = "financial-docs"

    # ── Azure Blob Storage ─────────────────────────────────────────────────────
    azure_storage_connection_string: str
    azure_storage_container: str = "source-pdfs"

    # ── Agent behavior ─────────────────────────────────────────────────────────
    agent_max_iterations: int = 3
    retrieval_top_k: int = 10
    retrieval_vector_candidates: int = 50   # ANN candidates before fusion
    sufficiency_threshold: float = 0.60
    max_retries_per_subquestion: int = 3

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    semantic_similarity_threshold: float = 0.75

    # ── Embedding batch sizes ──────────────────────────────────────────────────
    embedding_batch_size: int = 100         # chunks per Azure OpenAI call


settings = Settings()
