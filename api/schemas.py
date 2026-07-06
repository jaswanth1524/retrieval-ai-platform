"""FastAPI request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class PublicConfigResponse(BaseModel):
    """Non-secret runtime configuration exposed to the UI."""

    qdrant_collection: str
    embedding_provider: str
    dense_embedding_model: str
    sparse_embedding_model: str
    reranker_model: str
    embedding_model_tag: str
    llm_provider: str
    llm_model: str
    openai_model: str
    rrf_k: int
    dense_retrieval_limit: int
    sparse_retrieval_limit: int
    fused_top_n: int
    rerank_top_k: int
    max_context_chunks: int


class DocumentIngestResponse(BaseModel):
    """Document upload and ingestion response."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str


class QuestionRequest(BaseModel):
    """Question-answering request."""

    question: str = Field(min_length=1)


class CitationResponse(BaseModel):
    """Citation metadata returned with an answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str


class QuestionResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    sources: list[CitationResponse]

