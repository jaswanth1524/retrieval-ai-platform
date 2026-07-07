"""Application settings loaded from environment variables."""

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings shared by backend components."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docrag_documents"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_dense_vector_size: PositiveInt = 384

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/BM25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_model_tag: str = "fastembed:BAAI/bge-small-en-v1.5"

    rrf_k: PositiveInt = 60
    dense_retrieval_limit: PositiveInt = 50
    sparse_retrieval_limit: PositiveInt = 50
    fused_top_n: PositiveInt = 50
    rerank_top_k: PositiveInt = 8
    reranker_batch_size: PositiveInt = 64
    max_context_chunks: PositiveInt = 6

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: PositiveInt = 512
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_num_retries: int = Field(default=0, ge=0)
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = Field(default=None)
    openai_model: str = "gpt-4o-mini"

    max_upload_bytes: PositiveInt = 50 * 1024 * 1024
    # Despite the name, this counts whitespace-delimited words (api/documents.py splits
    # on str.split()), not the dense embedding model's subword tokens. bge-small hard-
    # truncates at 512 subword tokens, and English words expand to ~1.3-1.8 subword
    # tokens each, so 300 words stays safely under that limit after expansion; 500
    # silently dropped the tail of most full-size chunks from the embedded vector.
    chunk_size_tokens: PositiveInt = 300
    chunk_overlap_tokens: int = Field(default=75, ge=0)

    # Inert in the primary paths (dev uses the Vite proxy, prod serves the frontend
    # same-origin via StaticFiles) — a fallback for a contributor who points a
    # separately-running frontend directly at this API instead of using the proxy.
    cors_allow_origins: list[str] = Field(default_factory=list)


def get_settings() -> AppSettings:
    """Return settings from the current process environment."""

    return AppSettings()
