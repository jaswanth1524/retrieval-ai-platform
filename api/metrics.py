"""Prometheus metrics for latency and error observability."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

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

rate_limited_total = Counter(
    "docrag_rate_limited_total",
    "Requests refused with 429 by RATE_LIMIT_PER_MINUTE.",
)

answer_cache_hits_total = Counter(
    "docrag_answer_cache_hits_total",
    "Questions answered from the answer cache (not counted in the stage histograms).",
)

ingest_jobs_total = Counter(
    "docrag_ingest_jobs_total",
    "Total background ingestion jobs, labeled by outcome and (for outcome=failed) error_type.",
    labelnames=("outcome", "error_type"),
)

# Whole-job wall clock (parse through the last upsert), successful or not. Ingest of a
# large PDF on CPU runs minutes, so the buckets reach well past the question ones.
INGEST_LATENCY_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200)

ingest_job_seconds = Histogram(
    "docrag_ingest_job_seconds",
    "Wall-clock time of each background ingestion job, labeled by outcome.",
    labelnames=("outcome",),
    buckets=INGEST_LATENCY_BUCKETS,
)

# Where an ingest's time goes. "write" is everything under the filename lock:
# snapshot, upserts, storing the original and stale cleanup.
ingest_stage_seconds = Histogram(
    "docrag_ingest_stage_seconds",
    "Wall-clock time of each ingestion stage (parse, chunk, embed, write).",
    labelnames=("stage",),
    buckets=INGEST_LATENCY_BUCKETS,
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


build_info = Info("docrag_build", "The running DocRAG version.")

# How answers were grounded (citations.answer_status), cache hits included: the share of
# "uncited" and "insufficient_context" is the quality signal to watch.
answers_total = Counter(
    "docrag_answers_total",
    "Answers returned, labeled by grounding status (answered, uncited, ...).",
    labelnames=("status",),
)

answers_truncated_total = Counter(
    "docrag_answers_truncated_total",
    "Answers the provider cut off at the token limit (LLM_MAX_TOKENS).",
)

citation_retries_total = Counter(
    "docrag_citation_retries_total",
    "Answers that cited nothing and were regenerated once to add citations.",
)

feedback_total = Counter(
    "docrag_feedback_total",
    "Answer ratings recorded through POST /feedback, labeled by rating.",
    labelnames=("rating",),
)

llm_tokens_total = Counter(
    "docrag_llm_tokens_total",
    "Tokens the LLM provider reported using, labeled by provider and kind.",
    labelnames=("provider", "kind"),
)

question_slots_in_use = Gauge(
    "docrag_question_slots_in_use",
    "Questions and searches running now (bounded by MAX_CONCURRENT_QUESTIONS).",
)

ingest_backlog_bytes = Gauge(
    "docrag_ingest_backlog_bytes",
    "Bytes of uploads admitted and not yet indexed (bounded by INGEST_MAX_PENDING_BYTES).",
)

ingest_backlog_jobs = Gauge(
    "docrag_ingest_backlog_jobs",
    "Ingest jobs admitted and not yet finished.",
)

corpus_documents = Gauge(
    "docrag_corpus_documents",
    "Indexed documents, as of the last document listing.",
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
    # Its own series, not folded into "generate": it is a second LLM round trip that
    # fires on a minority of questions, so averaging it into generate would smear a
    # bimodal distribution into one meaningless number.
    question_stage_seconds.labels(stage="citation_retry").observe(
        timings.get("citation_retry_ms", 0.0) / 1000
    )
    # Queueing behind another question's rerank, a slice of "rerank" above.
    question_stage_seconds.labels(stage="rerank_wait").observe(
        timings.get("rerank_wait_ms", 0.0) / 1000
    )
    # Streamed questions only: /questions has no first token, and a 0 there would
    # drag the series toward an instant answer nobody saw.
    if timings.get("first_token_ms", 0.0) > 0:
        question_stage_seconds.labels(stage="first_token").observe(timings["first_token_ms"] / 1000)
    question_stage_seconds.labels(stage="total").observe(timings["total_ms"] / 1000)
