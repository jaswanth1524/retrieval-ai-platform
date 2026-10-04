"""Annotated FastAPI dependency aliases shared by the route modules."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from api.admission import QuestionSlots
from api.dependencies import (
    get_app_settings,
    get_feedback_store,
    get_ingest_backlog,
    get_ingest_executor,
    get_ingest_job_store,
    get_ingest_service,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_question_slots,
    get_rag_pipeline,
    get_raw_document_store,
    get_trace_store,
    get_vector_repository,
)
from api.feedback import FeedbackStore
from api.jobs import IngestBacklog, JobStore
from api.pipeline import IngestService, RagPipeline
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.tracing import TraceStore

SettingsDep = Annotated[AppSettings, Depends(get_app_settings)]


IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]


RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]


OllamaCheckDep = Annotated[Callable[[AppSettings], bool], Depends(get_ollama_reachability_checker)]


QdrantClientDep = Annotated[QdrantClient, Depends(get_qdrant_client)]


QdrantCheckDep = Annotated[Callable[[QdrantClient], bool], Depends(get_qdrant_reachability_checker)]


VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]


IngestJobStoreDep = Annotated[JobStore, Depends(get_ingest_job_store)]


RawDocumentStoreDep = Annotated[RawDocumentStore | None, Depends(get_raw_document_store)]


FeedbackStoreDep = Annotated[FeedbackStore | None, Depends(get_feedback_store)]


IngestExecutorDep = Annotated[ThreadPoolExecutor, Depends(get_ingest_executor)]


IngestBacklogDep = Annotated[IngestBacklog, Depends(get_ingest_backlog)]


QuestionSlotsDep = Annotated[QuestionSlots, Depends(get_question_slots)]


TraceStoreDep = Annotated[TraceStore, Depends(get_trace_store)]
