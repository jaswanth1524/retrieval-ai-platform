"""Prometheus metrics for latency and error observability."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Buckets tuned for a local-Ollama RAG pipeline: sub-second retrieval/rerank stages,
# multi-second-to-70s+ generation, and a total that can span both regimes.
STAGE_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 120)

question_stage_seconds = Histogram(
    "docrag_question_stage_seconds",
    "Wall-clock time spent in each question-answering stage.",
    labelnames=("stage",),
    buckets=STAGE_LATENCY_BUCKETS,
)

questions_total = Counter(
    "docrag_questions_total",
    "Total questions answered, labeled by outcome and (for outcome=error) error_type.",
    labelnames=("outcome", "error_type"),
)

ingest_jobs_total = Counter(
    "docrag_ingest_jobs_total",
    "Total background ingestion jobs, labeled by outcome and (for outcome=failed) error_type.",
    labelnames=("outcome", "error_type"),
)

ingest_chunks_total = Counter(
    "docrag_ingest_chunks_total",
    "Total document chunks successfully indexed.",
)

# error_type is "" for non-error outcomes — sum(...) by (outcome) still aggregates
# cleanly across it, and a scraper only cares about the breakdown on the error path.
NO_ERROR_TYPE = ""

ingest_jobs_evicted_total = Counter(
    "docrag_ingest_jobs_evicted_total",
    "Ingest jobs silently dropped from the in-memory store past its retention cap.",
)

traces_evicted_total = Counter(
    "docrag_traces_evicted_total",
    "Query traces silently dropped from the in-memory ring buffer past its retention cap.",
)


def observe_question_timings(timings: dict[str, float]) -> None:
    """Record one question's per-stage timings (input in milliseconds) as seconds."""

    question_stage_seconds.labels(stage="condense").observe(timings.get("condense_ms", 0.0) / 1000)
    # Two distinct stages at opposite ends of retrieval — query-variant generation runs
    # before embedding, neighbour expansion after rerank. They used to share one
    # "expand" label, which made the series unreadable.
    question_stage_seconds.labels(stage="query_expansion").observe(
        timings.get("query_expansion_ms", 0.0) / 1000
    )
    question_stage_seconds.labels(stage="context_expansion").observe(
        timings.get("context_expansion_ms", 0.0) / 1000
    )
    question_stage_seconds.labels(stage="embed").observe(timings["embed_ms"] / 1000)
    question_stage_seconds.labels(stage="search").observe(timings["search_ms"] / 1000)
    question_stage_seconds.labels(stage="rerank").observe(timings["rerank_ms"] / 1000)
    question_stage_seconds.labels(stage="generate").observe(timings["generate_ms"] / 1000)
    question_stage_seconds.labels(stage="total").observe(timings["total_ms"] / 1000)
