from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from api.embeddings import EmbeddedText
from api.generation import ChatMessage
from api.pipeline import AnswerOverrides, IngestService, RagPipeline, expand_with_neighbors
from api.repository import VectorRepository
from api.reranking import rerank_candidates
from api.retrieval import RetrievalConfigError, retrieve_candidates
from api.settings import AppSettings


class StaticEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings
        self.seen_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_texts = list(texts)
        return self.embeddings


class FakeReranker:
    def __init__(self) -> None:
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_documents = list(documents)
        return [1.0 for _ in documents]


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[ChatMessage] = []
        self.seen_provider: str | None = None
        self.seen_temperature: float | None = None

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.messages = list(messages)
        self.seen_provider = settings.llm_provider
        self.seen_temperature = settings.llm_temperature
        return self.answer

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        self.messages = list(messages)
        self.seen_provider = settings.llm_provider
        self.seen_temperature = settings.llm_temperature
        words = self.answer.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "


class RaisingGenerator:
    """Fails ``complete`` outright, and ``stream`` after yielding a partial answer —
    for exercising the pipeline's partial-trace-on-error paths."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise RuntimeError("generation boom")

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        yield "partial "
        raise RuntimeError("generation boom mid-stream")


class ListTraceStore:
    """Minimal in-memory ``TraceSink`` for asserting on what the pipeline records."""

    def __init__(self) -> None:
        self.traces: list[Any] = []

    def add(self, trace: Any) -> None:
        self.traces.append(trace)


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "pipeline_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
        "chunk_size_tokens": 20,
        "chunk_overlap_tokens": 0,
        "fused_top_n": 5,
        "rerank_top_k": 3,
        "max_context_chunks": 2,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_embedding(value: float) -> EmbeddedText:
    return EmbeddedText(
        dense=[value, 0.0, 0.0],
        sparse=models.SparseVector(indices=[1], values=[1.0]),
    )


def test_ingest_service_parses_chunks_and_indexes_them() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings)

    outcome = service.ingest("guide.txt", b"Intro\nalpha beta")

    assert outcome.filename == "guide.txt"
    assert outcome.sections_parsed == 1
    assert outcome.chunks_ingested == 1
    assert outcome.collection_name == "pipeline_documents"


def test_rag_pipeline_answers_from_ingested_document() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )

    reranker = FakeReranker()
    generator = FakeGenerator("Alpha is documented [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=reranker,
        generator=generator,
        settings=settings,
    )

    grounded = pipeline.answer("alpha")

    assert grounded.answer == "Alpha is documented [1]."
    assert [source.filename for source in grounded.sources] == ["guide.txt"]
    assert reranker.seen_documents == ["Intro alpha beta"]
    assert "Intro alpha beta" in generator.messages[1]["content"]
    # No override → generation uses the base provider.
    assert generator.seen_provider == "ollama"


def test_rag_pipeline_provider_override_does_not_mutate_base_settings() -> None:
    settings = make_settings()  # llm_provider defaults to "ollama"
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )

    generator = FakeGenerator("Alpha is documented [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    pipeline.answer("alpha", AnswerOverrides(llm_provider="openai"))

    # The override reached generation, but the shared singleton is untouched.
    assert generator.seen_provider == "openai"
    assert settings.llm_provider == "ollama"


def _ingest_three_matching_docs(
    repository: VectorRepository, settings: AppSettings
) -> None:
    for name, text in [
        ("guide1.txt", "First alpha document."),
        ("guide2.txt", "Second alpha document."),
        ("guide3.txt", "Third alpha document."),
    ]:
        IngestService(
            repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings
        ).ingest(name, text.encode())


def test_rag_pipeline_max_context_chunks_override_limits_sources() -> None:
    settings = make_settings(fused_top_n=5, rerank_top_k=3, max_context_chunks=2)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    _ingest_three_matching_docs(repository, settings)

    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Documented [1][2][3]."),
        settings=settings,
    )

    default_answer = pipeline.answer("alpha")
    overridden_answer = pipeline.answer("alpha", AnswerOverrides(max_context_chunks=1))

    assert len(default_answer.sources) == 2
    assert len(overridden_answer.sources) == 1
    # The base singleton's own max_context_chunks is untouched by the override.
    assert settings.max_context_chunks == 2


def test_rag_pipeline_rerank_top_k_override_limits_sources() -> None:
    settings = make_settings(fused_top_n=5, rerank_top_k=3, max_context_chunks=5)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    _ingest_three_matching_docs(repository, settings)

    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Documented [1][2][3]."),
        settings=settings,
    )

    overridden_answer = pipeline.answer("alpha", AnswerOverrides(rerank_top_k=1))

    # rerank_top_k=1 truncates before generation's own max_context_chunks=5 slice.
    assert len(overridden_answer.sources) == 1
    assert settings.rerank_top_k == 3


def test_rag_pipeline_llm_temperature_override_reaches_generation() -> None:
    settings = make_settings(llm_temperature=0.0)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro alpha beta"
    )

    generator = FakeGenerator("Alpha is documented [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    pipeline.answer("alpha", AnswerOverrides(llm_temperature=1.5))

    assert generator.seen_temperature == 1.5
    assert settings.llm_temperature == 0.0


def test_rag_pipeline_rejects_rerank_top_k_above_fused_top_n() -> None:
    settings = make_settings(fused_top_n=5, rerank_top_k=3, max_context_chunks=2)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro alpha beta"
    )

    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Documented [1]."),
        settings=settings,
    )

    with pytest.raises(RetrievalConfigError, match="fused_top_n"):
        pipeline.answer("alpha", AnswerOverrides(rerank_top_k=6))


def test_rag_pipeline_multiple_overrides_isolated_across_concurrent_style_calls() -> None:
    settings = make_settings(fused_top_n=5, rerank_top_k=3, max_context_chunks=2)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    _ingest_three_matching_docs(repository, settings)

    generator_a = FakeGenerator("A [1].")
    generator_b = FakeGenerator("B [1].")
    pipeline_a = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator_a,
        settings=settings,
    )
    pipeline_b = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator_b,
        settings=settings,
    )

    pipeline_a.answer(
        "alpha", AnswerOverrides(llm_provider="openai", max_context_chunks=1, llm_temperature=1.0)
    )
    pipeline_b.answer("alpha", AnswerOverrides(llm_provider="ollama"))

    assert generator_a.seen_provider == "openai"
    assert generator_a.seen_temperature == 1.0
    assert generator_b.seen_provider == "ollama"
    assert generator_b.seen_temperature == settings.llm_temperature
    # Neither call mutated the shared singleton.
    assert settings.llm_provider == "ollama"
    assert settings.max_context_chunks == 2


def test_rag_pipeline_answer_records_stage_timings() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
    )

    grounded = pipeline.answer("alpha")

    assert grounded.timings is not None
    assert grounded.timings.embed_ms >= 0
    assert grounded.timings.search_ms >= 0
    assert grounded.timings.rerank_ms >= 0
    assert grounded.timings.generate_ms >= 0
    assert grounded.timings.total_ms >= 0


def test_rag_pipeline_answer_filters_by_filenames() -> None:
    settings = make_settings(fused_top_n=5, rerank_top_k=5, max_context_chunks=5)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    for name, text in [("a.txt", "alpha in a"), ("b.txt", "alpha in b")]:
        IngestService(
            repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings
        ).ingest(name, text.encode())

    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Found in b [1]."),
        settings=settings,
    )

    grounded = pipeline.answer("alpha", filenames=["b.txt"])

    assert [source.filename for source in grounded.sources] == ["b.txt"]


def test_rag_pipeline_answer_stream_emits_sources_then_deltas_then_done() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
    )

    events = list(pipeline.answer_stream("alpha"))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["filename"] == "guide.txt"
    delta_events = events[1:-1]
    assert delta_events
    assert all(event["type"] == "delta" for event in delta_events)
    assert "".join(event["text"] for event in delta_events) == "Alpha is documented [1]."
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Alpha is documented [1]."
    assert [source["source_number"] for source in done["sources"]] == [1]
    assert done["timings"]["total_ms"] >= 0


def test_rag_pipeline_answer_stream_returns_insufficient_context_for_empty_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("unused"),
        settings=settings,
    )

    events = list(pipeline.answer_stream("alpha"))

    assert events[0] == {"type": "sources", "sources": [], "trace_id": None}
    assert events[-1]["type"] == "done"
    assert "not have enough information" in events[-1]["answer"]
    assert events[-1]["sources"] == []


def test_expand_with_neighbors_merges_adjacent_chunk_text() -> None:
    settings = make_settings(context_neighbor_radius=1, chunk_size_tokens=3, chunk_overlap_tokens=0)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    # Three chunks of 3 words each -> ordinals 1, 2, 3.
    embeddings = [make_embedding(1.0), make_embedding(1.0), make_embedding(1.0)]
    IngestService(
        repository, StaticEmbeddingProvider(embeddings), settings
    ).ingest("guide.txt", b"one two three four five six seven eight nine")

    candidates = retrieve_candidates(
        repository, settings, "one", StaticEmbeddingProvider([make_embedding(1.0)])
    )
    reranked = rerank_candidates("one", candidates, FakeReranker(), settings)
    middle = next(chunk for chunk in reranked if chunk.chunk_ordinal == 2)

    expanded = expand_with_neighbors(reranked, repository, settings)
    expanded_middle = next(chunk for chunk in expanded if chunk.chunk_ordinal == 2)

    assert middle.expanded_text is None
    assert expanded_middle.expanded_text is not None
    assert "one two three" in expanded_middle.expanded_text
    assert "seven eight nine" in expanded_middle.expanded_text


def test_expand_with_neighbors_disabled_by_zero_radius() -> None:
    settings = make_settings(context_neighbor_radius=0)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro alpha beta"
    )
    candidates = retrieve_candidates(
        repository, settings, "alpha", StaticEmbeddingProvider([make_embedding(1.0)])
    )
    reranked = rerank_candidates("alpha", candidates, FakeReranker(), settings)

    expanded = expand_with_neighbors(reranked, repository, settings)

    assert expanded == reranked


def test_rag_pipeline_answer_records_a_trace_when_a_store_is_configured() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
        trace_store=trace_store,
    )

    grounded = pipeline.answer("alpha")

    assert grounded.trace_id is not None
    assert len(trace_store.traces) == 1
    trace = trace_store.traces[0]
    assert trace.trace_id == grounded.trace_id
    assert trace.mode == "sync"
    assert trace.status == "ok"
    assert trace.question == "alpha"
    assert trace.answer == "Alpha is documented [1]."
    assert trace.cited_source_numbers == [1]
    assert trace.error is None
    assert trace.timings is not None and trace.timings["total_ms"] >= 0
    assert trace.config is not None
    assert trace.config.llm_provider == "ollama"
    assert len(trace.candidates) == 1
    assert trace.candidates[0].filename == "guide.txt"
    assert trace.candidates[0].kept is True
    assert trace.candidates[0].selected_for_context is True
    assert trace.prompt_messages is not None
    assert "Intro alpha beta" in trace.prompt_messages[1]["content"]


def test_rag_pipeline_answer_without_trace_store_leaves_trace_id_none() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
    )

    grounded = pipeline.answer("alpha")

    assert grounded.trace_id is None


def test_rag_pipeline_answer_records_partial_trace_on_generation_error() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=RaisingGenerator(),
        settings=settings,
        trace_store=trace_store,
    )

    with pytest.raises(RuntimeError, match="generation boom"):
        pipeline.answer("alpha")

    assert len(trace_store.traces) == 1
    trace = trace_store.traces[0]
    assert trace.status == "error"
    assert trace.error == "generation boom"
    assert trace.answer is None
    # The retrieval phase completed before generation failed, so the effective
    # config is still captured even though the trace as a whole is an error.
    assert trace.config is not None


def test_rag_pipeline_answer_stream_carries_matching_trace_id_in_sources_and_done() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
        trace_store=trace_store,
    )

    events = list(pipeline.answer_stream("alpha"))

    sources_event = events[0]
    done_event = events[-1]
    assert sources_event["trace_id"] is not None
    assert sources_event["trace_id"] == done_event["trace_id"]
    assert len(trace_store.traces) == 1
    trace = trace_store.traces[0]
    assert trace.trace_id == done_event["trace_id"]
    assert trace.mode == "stream"
    assert trace.status == "ok"
    assert trace.prompt_messages is not None


def test_rag_pipeline_answer_stream_records_partial_trace_on_mid_stream_error() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=RaisingGenerator(),
        settings=settings,
        trace_store=trace_store,
    )

    with pytest.raises(RuntimeError, match="generation boom mid-stream"):
        list(pipeline.answer_stream("alpha"))

    assert len(trace_store.traces) == 1
    trace = trace_store.traces[0]
    assert trace.mode == "stream"
    assert trace.status == "error"
    assert trace.error == "generation boom mid-stream"
    assert trace.answer == "partial"


def test_rag_pipeline_answer_stream_records_closed_trace_on_early_generator_close() -> None:
    """A client disconnecting mid-stream closes the generator (GeneratorExit) before
    any `except Exception`/success path runs — the `finally` clause must still record
    a trace rather than silently dropping it."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
        trace_store=trace_store,
    )

    generator = pipeline.answer_stream("alpha")
    next(generator)  # advance to the first yield (the sources event), then abandon it
    generator.close()

    assert len(trace_store.traces) == 1
    trace = trace_store.traces[0]
    assert trace.status == "error"
    assert trace.error == "Stream closed before completion."
