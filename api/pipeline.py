"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

# The pipeline is built in layers, each in its own module: answer_traces (trace
# records) < pipeline_core (retrieval, the plain answer) < answer_stream (the stream).
# Every name this module used to define is re-exported here (``x as x``).
from api.answer_events import DeltaEvent as DeltaEvent
from api.answer_events import DoneEvent as DoneEvent
from api.answer_events import FollowupsEvent as FollowupsEvent
from api.answer_events import SourcesEvent as SourcesEvent
from api.answer_events import StageEvent as StageEvent
from api.answer_events import StreamEvent as StreamEvent
from api.answer_run import AnswerOverrides as AnswerOverrides
from api.answer_run import AnswerRun as AnswerRun
from api.answer_run import RetrievalPhase as RetrievalPhase
from api.answer_run import check_rerank_top_k as check_rerank_top_k
from api.answer_run import filter_from_record as filter_from_record
from api.answer_run import narrowing_record as narrowing_record
from api.answer_run import replay_cached as replay_cached
from api.answer_run import served_query as served_query
from api.answer_run import timings_dict as timings_dict
from api.answer_run import trace_model_name as trace_model_name
from api.answer_run import trace_status as trace_status
from api.answer_run import usage_cost as usage_cost
from api.answer_run import usage_dict as usage_dict
from api.answer_run import usage_tuple as usage_tuple
from api.answer_stream import StreamingAnswers
from api.answer_traces import TraceRecording as TraceRecording
from api.context_expansion import _TAG_LOOKUP_EXECUTOR as _TAG_LOOKUP_EXECUTOR
from api.context_expansion import _VARIANT_EXECUTOR as _VARIANT_EXECUTOR
from api.context_expansion import _clip_head as _clip_head
from api.context_expansion import _clip_tail as _clip_tail
from api.context_expansion import _map_concurrently as _map_concurrently
from api.context_expansion import _strip_overlap as _strip_overlap
from api.context_expansion import expand_with_neighbors as expand_with_neighbors
from api.continuation import CONTINUE_INSTRUCTION as CONTINUE_INSTRUCTION
from api.continuation import ContinuePlan as ContinuePlan
from api.continuation import ContinueRefused as ContinueRefused
from api.continuation import base_prompt as base_prompt
from api.continuation import continuation_messages as continuation_messages
from api.continuation import drop_restart as drop_restart
from api.continuation import join_continuation as join_continuation
from api.conversation_history import CondenseMemo as CondenseMemo
from api.conversation_history import _clip_earlier_answers as _clip_earlier_answers
from api.conversation_history import condense_memo as condense_memo
from api.generation import GroundedAnswer as GroundedAnswer
from api.ingest_service import DocumentIngestOutcome as DocumentIngestOutcome
from api.ingest_service import IngestService as IngestService
from api.passage_selection import RECENCY_HORIZON_SECONDS as RECENCY_HORIZON_SECONDS
from api.passage_selection import asks_about_the_past as asks_about_the_past
from api.passage_selection import recency_ordered as recency_ordered
from api.passage_selection import select_passages as select_passages
from api.passage_selection import without_recency_for_the_past as without_recency_for_the_past
from api.pipeline_core import PipelineCore as PipelineCore
from api.run_coordination import QuestionAbandoned as QuestionAbandoned
from api.run_coordination import cached_run as cached_run
from api.run_coordination import drain as drain
from api.run_coordination import lead_or_join as lead_or_join
from api.run_coordination import wait_for_model as wait_for_model


class RagPipeline(StreamingAnswers):
    """Composes retrieval, reranking, and generation behind a single call."""
