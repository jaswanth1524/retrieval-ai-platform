from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from api.embeddings import (
    EmbeddingError,
    LocalEmbeddingProvider,
    coerce_dense_vector,
    coerce_sequence,
    coerce_sparse_vector,
)
from tests.factories import make_test_settings


class ArrayLike:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class SparseLike:
    def __init__(self, indices: object, values: object) -> None:
        self.indices = indices
        self.values = values


def test_coerce_dense_vector_accepts_array_like_values() -> None:
    assert coerce_dense_vector(ArrayLike([1, "2.5", 3])) == [1.0, 2.5, 3.0]


def test_coerce_sparse_vector_builds_qdrant_sparse_vector() -> None:
    vector = coerce_sparse_vector(
        SparseLike(indices=ArrayLike([10, "20"]), values=ArrayLike([0.5, "1.25"]))
    )

    assert vector.indices == [10, 20]
    assert vector.values == [0.5, 1.25]


def test_sparse_vector_rejects_mismatched_indices_and_values() -> None:
    with pytest.raises(EmbeddingError, match="indices"):
        coerce_sparse_vector(SparseLike(indices=[1, 2], values=[0.1]))


def test_coerce_sequence_rejects_strings() -> None:
    with pytest.raises(EmbeddingError, match="not iterable"):
        coerce_sequence("abc", label="bad vector")


class RaisingDenseModel:
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]:
        raise RuntimeError("model download failed")


class EmptySparseModel:
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]:
        return []


class RecordingDenseModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]:
        docs = [documents] if isinstance(documents, str) else list(documents)
        self.seen.extend(docs)
        return [[0.1, 0.2, 0.3] for _ in docs]


class RecordingSparseModel:
    def __init__(self) -> None:
        self.embed_seen: list[str] = []
        self.query_seen: list[str] = []

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]:
        docs = [documents] if isinstance(documents, str) else list(documents)
        self.embed_seen.extend(docs)
        return [SparseLike(indices=[1], values=[1.0]) for _ in docs]

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[object]:
        qs = [query] if isinstance(query, str) else list(query)
        self.query_seen.extend(qs)
        return [SparseLike(indices=[2], values=[1.0]) for _ in qs]


def test_embed_query_applies_instruction_and_query_side_sparse() -> None:
    settings = make_test_settings(dense_query_instruction="PREFIX: ")
    dense = RecordingDenseModel()
    sparse = RecordingSparseModel()
    provider = LocalEmbeddingProvider(settings, dense_model=dense, sparse_model=sparse)

    result = provider.embed_query("what is docrag")

    assert dense.seen == ["PREFIX: what is docrag"]
    assert sparse.query_seen == ["what is docrag"]  # raw query, no instruction
    assert sparse.embed_seen == []  # document-side embed never used for a query
    assert result.dense == [0.1, 0.2, 0.3]
    assert result.sparse.indices == [2]


def test_embed_query_without_instruction_uses_raw_query() -> None:
    settings = make_test_settings(dense_query_instruction="")
    dense = RecordingDenseModel()
    sparse = RecordingSparseModel()
    provider = LocalEmbeddingProvider(settings, dense_model=dense, sparse_model=sparse)

    provider.embed_query("hello")

    assert dense.seen == ["hello"]


def test_local_embedding_provider_wraps_model_failure_as_embedding_error() -> None:
    """Both models are lazy_load=True, so a bad model name or failed download only
    surfaces on this first call, not at construction — this call must translate it
    into EmbeddingError rather than an unwrapped exception."""

    provider = LocalEmbeddingProvider(
        make_test_settings(),
        dense_model=RaisingDenseModel(),
        sparse_model=EmptySparseModel(),
    )

    with pytest.raises(EmbeddingError) as raised:
        provider.embed_texts(["hello"])
    # A fixed message for the client; the cause (which can name paths) stays chained
    # for the log.
    assert str(raised.value) == "Embedding model failed."
    assert "model download failed" in str(raised.value.__cause__)


class SlowFirstLoadDenseModel:
    """Counts how many threads are inside ``embed`` at once, like fastembed's
    unlocked lazy load would let two of them load the model together."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]:
        docs = [documents] if isinstance(documents, str) else list(documents)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return [[0.1, 0.2, 0.3] for _ in docs]


def test_the_first_call_per_model_is_not_run_twice_at_once() -> None:
    """An ingest batch and a question arriving together at cold start both loaded the
    same lazy ONNX model; the first call per model is serialized."""

    dense = SlowFirstLoadDenseModel()
    provider = LocalEmbeddingProvider(
        make_test_settings(), dense_model=dense, sparse_model=RecordingSparseModel()
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(provider.embed_texts, ["a"]), pool.submit(provider.embed_query, "q")]
        for future in futures:
            future.result()

    assert dense.peak == 1


def test_embed_query_reuses_a_recent_query_embedding() -> None:
    """Regenerate and Retry re-ask the same text: the second embed is served from
    the cache, a different text is embedded fresh, and the cache is bounded."""

    from api import embeddings as embeddings_module

    settings = make_test_settings(dense_query_instruction="PREFIX: ")
    dense = RecordingDenseModel()
    sparse = RecordingSparseModel()
    provider = LocalEmbeddingProvider(settings, dense_model=dense, sparse_model=sparse)

    first = provider.embed_query("what is docrag")
    second = provider.embed_query("what is docrag")
    provider.embed_query("something else")

    assert second is first
    assert dense.seen == ["PREFIX: what is docrag", "PREFIX: something else"]
    assert sparse.query_seen == ["what is docrag", "something else"]

    for index in range(embeddings_module.QUERY_EMBEDDING_CACHE_SIZE):
        provider.embed_query(f"filler {index}")
    provider.embed_query("what is docrag")
    assert dense.seen[-1] == "PREFIX: what is docrag"  # evicted, so embedded again


def test_the_passage_prefix_reaches_only_the_dense_corpus_side() -> None:
    settings = make_test_settings(
        dense_passage_prefix="passage: ", dense_query_instruction="query: "
    )
    dense = RecordingDenseModel()
    sparse = RecordingSparseModel()
    provider = LocalEmbeddingProvider(settings, dense_model=dense, sparse_model=sparse)

    provider.embed_texts(["Alpha.", "Beta."])
    provider.embed_query("alpha?")

    assert dense.seen == ["passage: Alpha.", "passage: Beta.", "query: alpha?"]
    assert sparse.embed_seen == ["Alpha.", "Beta."]
    assert sparse.query_seen == ["alpha?"]


def test_bm25_gets_a_stemmer_language_only_when_it_is_not_english() -> None:
    from api.embeddings import sparse_model_options

    assert sparse_model_options(make_test_settings()) == {}
    assert sparse_model_options(make_test_settings(sparse_language=" German ")) == {
        "language": "german"
    }
    other = make_test_settings(
        sparse_embedding_model="prithivida/Splade_PP_en_v1", sparse_language="german"
    )
    assert sparse_model_options(other) == {}
