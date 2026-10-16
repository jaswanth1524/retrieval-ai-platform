"""Local embedding providers and vector conversion helpers."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from api.request_id import submit_in_context
from api.settings import AppSettings

logger = logging.getLogger(__name__)

# Recent query embeddings. Regenerate, Retry and a repeated question re-embed the same
# text; a dense+BM25 pass is small but it is on every question's critical path.
QUERY_EMBEDDING_CACHE_SIZE = 256


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


def sparse_model_options(settings: AppSettings) -> dict[str, Any]:
    """Extra options for the sparse model: BM25's stemmer language (SPARSE_LANGUAGE).

    Only for BM25 — the other sparse models fastembed has take no language — and only
    when it isn't English, so the default setup builds exactly the model it always did.
    """

    if settings.sparse_embedding_model.lower() != "qdrant/bm25":
        return {}
    language = settings.sparse_language.strip().lower()
    return {} if language in ("", "english") else {"language": language}


def passage_texts(texts: Sequence[str], prefix: str) -> list[str]:
    return [f"{prefix}{text}" for text in texts] if prefix else list(texts)


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
            threads=settings.embedding_threads or None,
        )
        self.sparse_model = sparse_model or SparseTextEmbedding(
            model_name=settings.sparse_embedding_model,
            lazy_load=True,
            **sparse_model_options(settings),
        )
        # One long-lived pair of workers instead of a fresh pool per call. Every query
        # embeds, so spawning and joining two OS threads was pure per-request overhead
        # on the hot path. The provider is a process-wide singleton (see
        # dependencies.get_embedding_provider), so the pool lives as long as the app.
        # It is never shut down: an ingest still running at app shutdown embeds its
        # remaining batches here; idle workers exit with the interpreter.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="docrag-embed")
        # Queries get their own pair: ingest submits whole 64-chunk batches to the pool
        # above, and a question's two tiny embeds queued behind them waited for the
        # entire batch to finish.
        self._query_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="docrag-embed-query"
        )
        # fastembed loads a lazy model on its first embed with no lock of its own, so
        # an ingest batch and a question arriving together at cold start loaded the
        # same ONNX session twice (twice the load time and memory). The first call per
        # model runs under its lock; later calls skip it.
        self._load_locks = {"dense": Lock(), "sparse": Lock()}
        self._loaded: set[str] = set()
        self._query_cache: OrderedDict[tuple[str, str], EmbeddedText] = OrderedDict()
        self._query_cache_lock = Lock()

    def _run_model[T](self, model: str, call: Callable[[], T]) -> T:
        """Run one embed call, serializing it with any other first call to ``model``."""

        if model in self._loaded:
            return call()
        with self._load_locks[model]:
            result = call()
            self._loaded.add(model)
            return result

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        """Embed texts with local dense and sparse models.

        The two ONNX-backed models run on separate threads — both release the GIL
        during inference, so this overlaps their latency instead of paying it twice
        sequentially (roughly halving embedding wall-clock for a batch).
        """

        # The corpus side only (e5-style models expect "passage: "); queries take
        # DENSE_QUERY_INSTRUCTION instead, and BM25 never sees either.
        prefix = self.settings.dense_passage_prefix

        if not texts:
            return []

        # Both models are constructed with lazy_load=True, so the actual model
        # download/load happens on this first call, not at __init__ — this is the
        # boundary that must translate a load failure into EmbeddingError.
        try:
            dense_future = submit_in_context(
                self._executor,
                lambda: self._run_model(
                    "dense", lambda: list(self.dense_model.embed(passage_texts(texts, prefix)))
                ),
            )
            sparse_future = submit_in_context(
                self._executor,
                lambda: self._run_model("sparse", lambda: list(self.sparse_model.embed(texts))),
            )
            dense_vectors = [coerce_dense_vector(vector) for vector in dense_future.result()]
            sparse_vectors = [coerce_sparse_vector(vector) for vector in sparse_future.result()]
        except EmbeddingError:
            raise
        except Exception as exc:
            # The cause (often a path or an ONNX runtime message) stays in the log.
            logger.warning("Embedding model failed.", exc_info=True)
            raise EmbeddingError("Embedding model failed.") from exc

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
        cache_key = (dense_text, text)
        with self._query_cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                return cached
        try:
            dense_future = submit_in_context(
                self._query_executor,
                lambda: self._run_model(
                    "dense", lambda: list(self.dense_model.embed([dense_text]))
                ),
            )
            sparse_future = submit_in_context(
                self._query_executor,
                lambda: self._run_model(
                    "sparse", lambda: list(self.sparse_model.query_embed([text]))
                ),
            )
            dense_vectors = [coerce_dense_vector(v) for v in dense_future.result()]
            sparse_vectors = [coerce_sparse_vector(v) for v in sparse_future.result()]
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.warning("Query embedding failed.", exc_info=True)
            raise EmbeddingError("Embedding model failed.") from exc

        if len(dense_vectors) != 1 or len(sparse_vectors) != 1:
            raise EmbeddingError(
                "Query embedding must produce exactly one dense and one sparse vector."
            )
        embedded = EmbeddedText(dense=dense_vectors[0], sparse=sparse_vectors[0])
        with self._query_cache_lock:
            self._query_cache[cache_key] = embedded
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > QUERY_EMBEDDING_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return embedded


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
