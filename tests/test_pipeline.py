from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from api.embeddings import EmbeddedText
from api.generation import CITATION_RETRY_REMINDER, ChatMessage, GenerationError
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
        self.seen_query: str | None = None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_query = query
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


class SequencedGenerator:
    """Returns a different ``complete``/``stream`` response on each successive call —
    for asserting on the condense call distinctly from the final-answer call."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.call_count = 0
        self.calls: list[tuple[list[ChatMessage], AppSettings]] = []

    def _next_answer(self) -> str:
        answer = self.answers[min(self.call_count, len(self.answers) - 1)]
        self.call_count += 1
        return answer

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls.append((list(messages), settings))
        return self._next_answer()

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        self.calls.append((list(messages), settings))
        answer = self._next_answer()
        words = answer.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "


class GenerationErrorOnceGenerator:
    """Raises ``GenerationError`` on the first ``complete`` call (the condense call),
    then answers normally — for exercising the condense-failure fallback path."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.call_count = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.call_count += 1
        if self.call_count == 1:
            raise GenerationError("condense boom")
        return self.answer

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        words = self.answer.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "


class EmptyThenAnswerGenerator:
    """Returns an empty/whitespace ``complete`` response the first call (the condense
    call), then answers normally — exercises the empty-condense-result fallback."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.call_count = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return "   "
        return self.answer

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        words = self.answer.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else word + " "


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


class WordTokenCounter:
    """Deterministic 1-token-per-word counter for predictable chunk boundaries."""

    def count(self, text: str) -> int:
        return len(text.split())


def test_ingest_service_parses_chunks_and_indexes_them() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings)

    outcome = service.ingest("guide.txt", b"Intro\nalpha beta")

    assert outcome.filename == "guide.txt"
    assert outcome.sections_parsed == 1
    assert outcome.chunks_ingested == 1
    assert outcome.collection_name == "pipeline_documents"


def test_ingest_service_stamps_byte_size_and_upload_time_on_every_chunk() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings)
    content = b"Intro\nalpha beta"

    service.ingest("guide.txt", content)

    metadata = repository.filename_metadata(settings)["guide.txt"]
    assert metadata.byte_size == len(content)
    assert metadata.uploaded_at is not None
    assert metadata.uploaded_at > 0


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
    # budget = 8 - 3 (prefix words) = 5, so each sentence becomes its own chunk
    # (ordinals 1, 2, 3) under the injected word counter.
    settings = make_settings(context_neighbor_radius=1, chunk_size_tokens=8, chunk_overlap_tokens=0)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    embeddings = [make_embedding(1.0), make_embedding(1.0), make_embedding(1.0)]
    IngestService(
        repository,
        StaticEmbeddingProvider(embeddings),
        settings,
        token_counter=WordTokenCounter(),
    ).ingest("guide.md", b"# Notes\nOne two three. Four five six. Seven eight nine.")

    candidates = retrieve_candidates(
        repository, settings, "one", StaticEmbeddingProvider([make_embedding(1.0)])
    )
    reranked = rerank_candidates("one", candidates, FakeReranker(), settings)
    middle = next(chunk for chunk in reranked if chunk.chunk_ordinal == 2)

    expanded = expand_with_neighbors(reranked, repository, settings)
    expanded_middle = next(chunk for chunk in expanded if chunk.chunk_ordinal == 2)

    assert middle.expanded_text is None
    assert expanded_middle.expanded_text is not None
    assert "One two three" in expanded_middle.expanded_text
    assert "Seven eight nine" in expanded_middle.expanded_text


def test_expand_with_neighbors_fetches_once_per_filename_not_once_per_chunk() -> None:
    """Neighbour lookup is batched per document.

    A per-chunk round trip meant up to rerank_top_k sequential Qdrant scrolls on every
    question, for data one query per file returns. This pins the call count so the N+1
    cannot come back, and re-asserts the merged text so batching didn't change output.
    """

    settings = make_settings(context_neighbor_radius=1, chunk_size_tokens=8, chunk_overlap_tokens=0)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    embeddings = [make_embedding(1.0) for _ in range(3)]
    IngestService(
        repository,
        StaticEmbeddingProvider(embeddings),
        settings,
        token_counter=WordTokenCounter(),
    ).ingest("guide.md", b"# Notes\nOne two three. Four five six. Seven eight nine.")

    candidates = retrieve_candidates(
        repository, settings, "one", StaticEmbeddingProvider([make_embedding(1.0)])
    )
    reranked = rerank_candidates("one", candidates, FakeReranker(), settings)
    assert len(reranked) >= 3, "need several chunks from one file for this to mean anything"

    calls: list[tuple[str, list[int]]] = []
    real_fetch = repository.fetch_neighbors

    def counting_fetch(
        settings_arg: AppSettings, filename: str, ordinals: Sequence[int]
    ) -> Any:
        calls.append((filename, list(ordinals)))
        return real_fetch(settings_arg, filename, ordinals)

    repository.fetch_neighbors = counting_fetch  # type: ignore[method-assign]
    expanded = expand_with_neighbors(reranked, repository, settings)

    # One call, because every chunk came from the same document.
    assert len(calls) == 1, calls
    assert calls[0][0] == "guide.md"
    # And it asked for the union of the neighbours, deduplicated and sorted.
    assert calls[0][1] == sorted(set(calls[0][1]))

    expanded_middle = next(chunk for chunk in expanded if chunk.chunk_ordinal == 2)
    assert expanded_middle.expanded_text is not None
    assert "One two three" in expanded_middle.expanded_text
    assert "Seven eight nine" in expanded_middle.expanded_text


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


_HISTORY: list[ChatMessage] = [
    {"role": "user", "content": "Tell me about doc A and doc B."},
    {"role": "assistant", "content": "Doc A covers setup [1]. Doc B covers billing [2]."},
]


def test_rag_pipeline_answer_with_history_condenses_before_retrieval() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    reranker = FakeReranker()
    generator = SequencedGenerator(["What is the second document about?", "Doc B is billing [1]."])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=reranker,
        generator=generator,
        settings=settings,
    )

    grounded = pipeline.answer("What about the second one?", history=_HISTORY)

    assert generator.call_count == 2
    assert embeddings.seen_texts == ["What is the second document about?"]
    assert reranker.seen_query == "What is the second document about?"
    assert grounded.answer == "Doc B is billing [1]."
    # Generation still sees the raw question plus the history, not the condensed one.
    final_call_messages = generator.calls[-1][0]
    assert any("What about the second one?" in m["content"] for m in final_call_messages)
    assert _HISTORY[0] in final_call_messages
    assert _HISTORY[1] in final_call_messages


def test_rag_pipeline_answer_without_history_makes_a_single_generation_call() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    generator = SequencedGenerator(["Alpha is documented [1]."])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    grounded = pipeline.answer("alpha")

    assert generator.call_count == 1
    assert grounded.answer == "Alpha is documented [1]."
    assert grounded.timings is not None
    assert grounded.timings.condense_ms == 0.0


def test_rag_pipeline_answer_condense_disabled_skips_condense_but_keeps_history() -> None:
    settings = make_settings(conversation_condense_enabled=False)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    generator = SequencedGenerator(["Alpha is documented [1]."])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    grounded = pipeline.answer("alpha", history=_HISTORY)

    assert generator.call_count == 1
    assert embeddings.seen_texts == ["alpha"]
    final_call_messages = generator.calls[-1][0]
    assert _HISTORY[0] in final_call_messages
    assert grounded.answer == "Alpha is documented [1]."


def test_rag_pipeline_answer_condense_generation_error_falls_back_to_raw_question() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=FakeReranker(),
        generator=GenerationErrorOnceGenerator("Alpha is documented [1]."),
        settings=settings,
        trace_store=trace_store,
    )

    grounded = pipeline.answer("alpha", history=_HISTORY)

    assert grounded.answer == "Alpha is documented [1]."
    assert embeddings.seen_texts == ["alpha"]
    trace = trace_store.traces[0]
    assert trace.condensed_question is None
    assert trace.history_message_count == len(_HISTORY)


def test_rag_pipeline_answer_condense_empty_result_falls_back_to_raw_question() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=FakeReranker(),
        generator=EmptyThenAnswerGenerator("Alpha is documented [1]."),
        settings=settings,
    )

    grounded = pipeline.answer("alpha", history=_HISTORY)

    assert grounded.answer == "Alpha is documented [1]."
    assert embeddings.seen_texts == ["alpha"]


def test_rag_pipeline_answer_history_truncated_to_max_history_messages() -> None:
    settings = make_settings(conversation_max_history_messages=2)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    long_history: list[ChatMessage] = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
    ]
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=SequencedGenerator(["condensed query", "Alpha is documented [1]."]),
        settings=settings,
        trace_store=trace_store,
    )

    pipeline.answer("alpha", history=long_history)

    trace = trace_store.traces[0]
    assert trace.history_message_count == 2
    final_call_messages = trace_store.traces[0].prompt_messages
    assert final_call_messages is not None
    assert {"role": "user", "content": "turn 1"} not in final_call_messages
    assert {"role": "user", "content": "turn 2"} in final_call_messages
    assert {"role": "assistant", "content": "reply 2"} in final_call_messages


def test_rag_pipeline_answer_records_condensed_question_in_trace() -> None:
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
        generator=SequencedGenerator(
            ["What is the second document about?", "Doc B is billing [1]."]
        ),
        settings=settings,
        trace_store=trace_store,
    )

    grounded = pipeline.answer("What about the second one?", history=_HISTORY)

    trace = trace_store.traces[0]
    assert trace.condensed_question == "What is the second document about?"
    assert trace.history_message_count == len(_HISTORY)
    assert grounded.timings is not None
    assert grounded.timings.condense_ms > 0.0


def test_rag_pipeline_answer_stream_condense_parity_with_sync() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    generator = SequencedGenerator(["What is the second document about?", "Doc B is billing [1]."])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    events = list(pipeline.answer_stream("What about the second one?", history=_HISTORY))

    # The condense call (via `complete`) happens before the sources event, which
    # itself happens before any `stream` delta call.
    assert generator.call_count == 2
    assert embeddings.seen_texts == ["What is the second document about?"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["timings"]["condense_ms"] > 0.0


class CountingRepository:
    """Wraps a real repository to count hybrid_search calls (multi-query assertions)."""

    def __init__(self, inner: VectorRepository) -> None:
        self.inner = inner
        self.search_calls = 0

    def ensure_ready(self, settings: AppSettings) -> None:
        self.inner.ensure_ready(settings)

    def hybrid_search(
        self, settings: AppSettings, query_embedding: Any, filenames: Any = None
    ) -> Any:
        self.search_calls += 1
        return self.inner.hybrid_search(settings, query_embedding, filenames)

    def fetch_neighbors(self, settings: AppSettings, filename: str, ordinals: Any) -> Any:
        return self.inner.fetch_neighbors(settings, filename, ordinals)


def test_multi_query_expansion_runs_extra_searches_and_records_variants() -> None:
    settings = make_settings(query_expansion_enabled=True, query_expansion_count=2)
    client = QdrantClient(":memory:")
    inner = VectorRepository(client)
    IngestService(inner, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    counting = CountingRepository(inner)
    trace_store = ListTraceStore()
    generator = SequencedGenerator(["rewrite one\nrewrite two", "Alpha is documented [1]."])
    pipeline = RagPipeline(
        repository=counting,  # type: ignore[arg-type]
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        trace_store=trace_store,
    )

    result = pipeline.answer("alpha")

    assert result.answer == "Alpha is documented [1]."
    # Original query + 2 variants => 3 hybrid searches.
    assert counting.search_calls == 3
    assert trace_store.traces[0].query_variants == ["rewrite one", "rewrite two"]


def test_no_expansion_runs_single_search_and_no_variants() -> None:
    settings = make_settings(query_expansion_enabled=False)
    client = QdrantClient(":memory:")
    inner = VectorRepository(client)
    IngestService(inner, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    counting = CountingRepository(inner)
    trace_store = ListTraceStore()
    pipeline = RagPipeline(
        repository=counting,  # type: ignore[arg-type]
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator("Alpha is documented [1]."),
        settings=settings,
        trace_store=trace_store,
    )

    pipeline.answer("alpha")

    assert counting.search_calls == 1
    assert trace_store.traces[0].query_variants == []


def test_streaming_citation_retry_swaps_corrected_answer_into_done() -> None:
    settings = make_settings(citation_retry_enabled=True)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    trace_store = ListTraceStore()
    # Streamed answer has no citation; the retry supplies one.
    generator = SequencedGenerator(["alpha is documented plainly", "Alpha is documented [1]."])
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        trace_store=trace_store,
    )

    events = list(pipeline.answer_stream("alpha"))
    done = next(event for event in events if event["type"] == "done")

    assert done["answer"] == "Alpha is documented [1]."
    assert [source["source_number"] for source in done["sources"]] == [1]
    assert trace_store.traces[0].citation_retry_used is True


def _retrying_pipeline(
    trace_store: ListTraceStore, generator: SequencedGenerator
) -> RagPipeline:
    settings = make_settings(citation_retry_enabled=True)
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    return RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        trace_store=trace_store,
    )


def test_streaming_trace_records_the_retry_prompt_not_the_first_one() -> None:
    """When the citation retry fires, the trace must show the prompt that produced the
    stored answer — the retry's continuation, not the prompt that was streamed.

    A debug trace showing a prompt the model was never asked is worse than no trace at
    all in a product whose purpose includes debugging retrieval.
    """

    trace_store = ListTraceStore()
    generator = SequencedGenerator(["alpha is documented plainly", "Alpha is documented [1]."])
    pipeline = _retrying_pipeline(trace_store, generator)

    list(pipeline.answer_stream("alpha"))

    trace = trace_store.traces[0]
    assert trace.citation_retry_used is True
    assert trace.answer == "Alpha is documented [1]."
    assert trace.prompt_messages[-1] == {"role": "user", "content": CITATION_RETRY_REMINDER}
    assert trace.prompt_messages[-2] == {
        "role": "assistant",
        "content": "alpha is documented plainly",
    }
    # It is the message list the generator was really called with for that answer.
    assert trace.prompt_messages == generator.calls[-1][0]


def test_sync_trace_records_the_retry_prompt_not_a_rebuild() -> None:
    trace_store = ListTraceStore()
    generator = SequencedGenerator(["alpha is documented plainly", "Alpha is documented [1]."])
    pipeline = _retrying_pipeline(trace_store, generator)

    grounded = pipeline.answer("alpha")

    trace = trace_store.traces[0]
    assert grounded.citation_retry_used is True
    assert trace.prompt_messages[-1] == {"role": "user", "content": CITATION_RETRY_REMINDER}
    assert trace.prompt_messages == generator.calls[-1][0]


def test_sync_trace_records_the_plain_prompt_when_no_retry_happens() -> None:
    """The no-retry path still records the exact messages generation used."""

    trace_store = ListTraceStore()
    generator = SequencedGenerator(["Alpha is documented [1]."])
    pipeline = _retrying_pipeline(trace_store, generator)

    pipeline.answer("alpha")

    trace = trace_store.traces[0]
    assert trace.citation_retry_used is False
    assert trace.prompt_messages == generator.calls[-1][0]
    assert trace.prompt_messages[-1]["role"] == "user"
    assert CITATION_RETRY_REMINDER not in trace.prompt_messages[-1]["content"]
