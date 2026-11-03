"""Cached answers scoped to named documents survive changes to other documents."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient

from api.answer_cache import AnswerCache
from api.corpus import bump_corpus_generation
from api.embeddings import EmbeddedText
from api.ingest_service import IngestService
from api.pipeline import RagPipeline
from api.repository import VectorRepository
from tests.test_pipeline import (
    FakeGenerator,
    FakeReranker,
    StaticEmbeddingProvider,
    make_embedding,
    make_settings,
)


class OnePerText:
    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        return [make_embedding(1.0) for _ in texts]


def _pipeline() -> tuple[RagPipeline, IngestService, FakeGenerator]:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, OnePerText(), settings)
    for name in ("a.txt", "b.txt"):
        service.ingest(name, f"Intro\n{name} alpha".encode())
    generator = FakeGenerator("Alpha [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        answer_cache=AnswerCache(max_entries=32, ttl_seconds=600),
    )
    return pipeline, service, generator


def test_a_scoped_answer_survives_uploads_of_other_documents_but_not_of_its_own() -> None:
    pipeline, service, _ = _pipeline()
    assert not pipeline.answer("alpha?", filenames=["a.txt"]).cached

    service.ingest("c.txt", b"Intro\nc.txt alpha")  # another document
    assert pipeline.answer("alpha?", filenames=["a.txt"]).cached
    # A whole-corpus question is still answered again after any change.
    assert not pipeline.answer("alpha?").cached
    service.ingest("d.txt", b"Intro\nd.txt alpha")
    assert not pipeline.answer("alpha?").cached

    service.ingest("a.txt", b"Intro\na.txt alpha, edited")  # its own document
    assert not pipeline.answer("alpha?", filenames=["a.txt"]).cached
    assert pipeline.answer("alpha?", filenames=["a.txt"]).cached


def test_a_change_the_log_cannot_name_misses_and_tags_keep_the_old_rule() -> None:
    pipeline, service, _ = _pipeline()
    pipeline.answer("alpha?", filenames=["a.txt"])
    pipeline.answer("alpha?", filenames=["a.txt"], tags=["hr"])
    # A bump naming nothing (a workspace deleted, say): anything may have changed.
    bump_corpus_generation(collection=make_settings().qdrant_collection)
    assert not pipeline.answer("alpha?", filenames=["a.txt"]).cached

    pipeline.answer("alpha?", filenames=["a.txt"], tags=["hr"])
    service.ingest("c.txt", b"Intro\nc.txt alpha")
    # Tags: a new document could carry them, so any change still misses.
    assert not pipeline.answer("alpha?", filenames=["a.txt"], tags=["hr"]).cached
