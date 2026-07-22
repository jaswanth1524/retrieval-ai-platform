from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from api.embeddings import (
    EmbeddingError,
    LocalEmbeddingProvider,
    coerce_dense_vector,
    coerce_sequence,
    coerce_sparse_vector,
)
from api.settings import AppSettings


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
    settings = AppSettings(_env_file=None, dense_query_instruction="PREFIX: ")  # type: ignore[call-arg]
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
    settings = AppSettings(_env_file=None, dense_query_instruction="")  # type: ignore[call-arg]
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
        AppSettings(_env_file=None),  # type: ignore[call-arg]
        dense_model=RaisingDenseModel(),
        sparse_model=EmptySparseModel(),
    )

    with pytest.raises(EmbeddingError, match="model download failed"):
        provider.embed_texts(["hello"])

