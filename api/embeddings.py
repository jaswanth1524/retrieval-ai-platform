"""Local embedding providers and vector conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from api.settings import AppSettings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation or conversion fails."""


class DenseEmbeddingModel(Protocol):
    """Embedding model surface used for dense vectors."""

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]: ...


class SparseEmbeddingModel(Protocol):
    """Embedding model surface used for sparse vectors."""

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[object]: ...

    def query_embed(
        self,
        query: str | Iterable[str],
        **kwargs: Any,
    ) -> Iterable[object]: ...


@dataclass(frozen=True)
class EmbeddedText:
    """Dense and sparse vectors for one source text."""

    dense: list[float]
    sparse: models.SparseVector


class LocalEmbeddingProvider:
    """Generate local dense and sparse embeddings with FastEmbed."""

    def __init__(
        self,
        settings: AppSettings,
        dense_model: DenseEmbeddingModel | None = None,
        sparse_model: SparseEmbeddingModel | None = None,
    ) -> None:
        self.settings = settings
        self.dense_model = dense_model or TextEmbedding(
            model_name=settings.dense_embedding_model,
            lazy_load=True,
        )
        self.sparse_model = sparse_model or SparseTextEmbedding(
            model_name=settings.sparse_embedding_model,
            lazy_load=True,
        )
        # One long-lived pair of workers instead of a fresh pool per call. Every query
        # embeds, so spawning and joining two OS threads was pure per-request overhead
        # on the hot path. The provider is a process-wide singleton (see
        # dependencies.get_embedding_provider), so the pool lives as long as the app;
        # daemon threads let the process exit without an explicit shutdown.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="docrag-embed")

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        """Embed texts with local dense and sparse models.

        The two ONNX-backed models run on separate threads — both release the GIL
        during inference, so this overlaps their latency instead of paying it twice
        sequentially (roughly halving embedding wall-clock for a batch).
        """

        if not texts:
            return []

        # Both models are constructed with lazy_load=True, so the actual model
        # download/load happens on this first call, not at __init__ — this is the
        # boundary that must translate a load failure into EmbeddingError.
        try:
            dense_future = self._executor.submit(lambda: list(self.dense_model.embed(texts)))
            sparse_future = self._executor.submit(lambda: list(self.sparse_model.embed(texts)))
            dense_vectors = [coerce_dense_vector(vector) for vector in dense_future.result()]
            sparse_vectors = [coerce_sparse_vector(vector) for vector in sparse_future.result()]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding model failed: {exc}") from exc

        if len(dense_vectors) != len(texts):
            raise EmbeddingError(
                f"Dense embedding count mismatch: got {len(dense_vectors)}, expected {len(texts)}."
            )
        if len(sparse_vectors) != len(texts):
            raise EmbeddingError(
                "Sparse embedding count mismatch: "
                f"got {len(sparse_vectors)}, expected {len(texts)}."
            )

        return [
            EmbeddedText(dense=dense, sparse=sparse)
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def embed_query(self, text: str) -> EmbeddedText:
        """Embed one query symmetrically to the corpus side.

        Two asymmetries the corpus path has that a raw query lacks are fixed here:
        - Dense: prepend the model's query instruction (bge-small recommends
          "Represent this sentence for searching relevant passages: "; fastembed does
          not apply it automatically). Empty ``dense_query_instruction`` disables it.
        - Sparse: use BM25's ``query_embed`` (unique-token, weight-1.0 query encoding)
          rather than the document-side ``embed`` (TF-saturation/length weighting) —
          the model pair's intended query path.
        """

        instruction = self.settings.dense_query_instruction
        dense_text = f"{instruction}{text}" if instruction else text
        try:
            dense_future = self._executor.submit(lambda: list(self.dense_model.embed([dense_text])))
            sparse_future = self._executor.submit(
                lambda: list(self.sparse_model.query_embed([text]))
            )
            dense_vectors = [coerce_dense_vector(v) for v in dense_future.result()]
            sparse_vectors = [coerce_sparse_vector(v) for v in sparse_future.result()]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding model failed: {exc}") from exc

        if len(dense_vectors) != 1 or len(sparse_vectors) != 1:
            raise EmbeddingError(
                "Query embedding must produce exactly one dense and one sparse vector."
            )
        return EmbeddedText(dense=dense_vectors[0], sparse=sparse_vectors[0])


def coerce_dense_vector(vector: object) -> list[float]:
    """Convert a FastEmbed dense vector object into plain floats."""

    values = coerce_sequence(vector, label="dense vector")
    dense = [float(value) for value in values]
    if not dense:
        raise EmbeddingError("Dense embedding vector is empty.")
    return dense


def coerce_sparse_vector(vector: object) -> models.SparseVector:
    """Convert a FastEmbed sparse vector object into Qdrant's sparse vector shape."""

    indices_obj = getattr(vector, "indices", None)
    values_obj = getattr(vector, "values", None)
    if indices_obj is None or values_obj is None:
        raise EmbeddingError("Sparse embedding must expose indices and values.")

    indices = [int(index) for index in coerce_sequence(indices_obj, label="sparse indices")]
    values = [float(value) for value in coerce_sequence(values_obj, label="sparse values")]
    if len(indices) != len(values):
        raise EmbeddingError(
            f"Sparse embedding has {len(indices)} indices but {len(values)} values."
        )
    return models.SparseVector(indices=indices, values=values)


def coerce_sequence(value: object, *, label: str) -> list[Any]:
    """Convert array-like values into a plain list."""

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise EmbeddingError(f"{label} is not iterable.")
    return list(value)
