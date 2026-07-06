from __future__ import annotations

import pytest

from api.embeddings import (
    EmbeddingError,
    coerce_dense_vector,
    coerce_sequence,
    coerce_sparse_vector,
)


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

