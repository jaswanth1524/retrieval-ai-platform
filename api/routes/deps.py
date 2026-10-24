"""Annotated FastAPI dependency aliases shared by the route modules."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient

from api.admission import QuestionSlots
from api.chunking import TokenCounter
from api.dependencies import (
    get_document_summarizer,
    get_feedback_store,
    get_ingest_backlog,
    get_ingest_executor,
    get_ingest_job_store,
    get_ingest_service,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_query_log_store,
    get_question_slots,
    get_rag_pipeline,
    get_request_raw_store,
    get_request_settings,
    get_token_counter,
    get_trace_store,
    get_vector_repository,
)
from api.feedback import FeedbackStore
from api.ingest_jobs import IngestQueue
from api.jobs import IngestBacklog, JobStore
from api.pipeline import IngestService, RagPipeline
from api.query_log import QueryLogStore
from api.raw_documents import RawDocumentStore
from api.repository import VectorRepository
from api.settings import AppSettings
from api.summarize import DocumentSummarizer
from api.tracing import TraceStore

# The request's settings: the app's, pointed at the X-Workspace workspace when one is
# named (api/workspaces.py). Routes read the collection and directories from these.
SettingsDep = Annotated[AppSettings, Depends(get_request_settings)]


IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]


RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]


DocumentSummarizerDep = Annotated[DocumentSummarizer, Depends(get_document_summarizer)]


OllamaCheckDep = Annotated[Callable[[AppSettings], bool], Depends(get_ollama_reachability_checker)]


QdrantClientDep = Annotated[QdrantClient, Depends(get_qdrant_client)]


QdrantCheckDep = Annotated[Callable[[QdrantClient], bool], Depends(get_qdrant_reachability_checker)]


VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]


IngestJobStoreDep = Annotated[JobStore, Depends(get_ingest_job_store)]


RawDocumentStoreDep = Annotated[RawDocumentStore | None, Depends(get_request_raw_store)]


FeedbackStoreDep = Annotated[FeedbackStore | None, Depends(get_feedback_store)]


IngestExecutorDep = Annotated[ThreadPoolExecutor, Depends(get_ingest_executor)]


IngestBacklogDep = Annotated[IngestBacklog, Depends(get_ingest_backlog)]


def get_ingest_queue(
    job_store: IngestJobStoreDep,
    executor: IngestExecutorDep,
    backlog: IngestBacklogDep,
    ingest_service: IngestServiceDep,
) -> IngestQueue:
    """The four collaborators every route that queues an ingest takes, together.
    Sub-dependencies, so a test's override of any one of them still applies."""

    return IngestQueue(job_store, executor, backlog, ingest_service)


IngestQueueDep = Annotated[IngestQueue, Depends(get_ingest_queue)]


QuestionSlotsDep = Annotated[QuestionSlots, Depends(get_question_slots)]


TraceStoreDep = Annotated[TraceStore, Depends(get_trace_store)]
QueryLogDep = Annotated[QueryLogStore | None, Depends(get_query_log_store)]


TokenCounterDep = Annotated[TokenCounter, Depends(get_token_counter)]
