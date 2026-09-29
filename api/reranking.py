"""Cross-encoder reranking for fused retrieval candidates."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

from api.documents import contextual_text
from api.retrieval import RetrievalError, RetrievedChunk
from api.settings import AppSettings


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid; avoids OverflowError for large-magnitude logits."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class RerankingError(RuntimeError):
    """Raised when cross-encoder reranking fails."""


class CrossEncoderModel(Protocol):
    """Cross-encoder model surface used for reranking."""

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]: ...


@dataclass(frozen=True)
class RerankedChunk:
    """A retrieval candidate after cross-encoder reranking.

    ``expanded_text``, when set by neighbor-context expansion, is what actually goes
    into the generation prompt (small-to-big retrieval) — ``text`` stays the original
    chunk content used for the citation excerpt shown to the user.
    """

    point_id: str
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float
    chunk_ordinal: int | None = None
    expanded_text: str | None = None


_custom_model_lock = threading.Lock()

AUTO_ONNX_FILE = "auto"
# Exports "auto" picks per model. Only combinations actually measured against the
# registered export belong here — any other model keeps fastembed's own file.
_AUTO_ONNX_EXPORTS = {
    "jinaai/jina-reranker-v2-base-multilingual": "onnx/model_int8.onnx",
}


def _resolve_model(model: str, onnx_file: str) -> tuple[str, str | None]:
    """``(model_name, cache_dir)`` to load ``model`` with, registering an alternate
    ONNX export when asked.

    fastembed registers exactly one ONNX file per model (jina: fp32 ``onnx/model.onnx``)
    while the model repos also ship quantized exports. A non-empty ``onnx_file``
    registers the same repo under a distinct name pointing at that file, in its own
    cache directory: fastembed treats an existing snapshot of the repo as complete once
    its recorded files verify, so with the fp32 export already cached it never fetched
    the alternate file and failed to load it.
    """

    if onnx_file == AUTO_ONNX_FILE:
        onnx_file = _AUTO_ONNX_EXPORTS.get(model, "")
    if not onnx_file:
        return model, None
    name = f"{model}#{onnx_file}"
    with _custom_model_lock:
        supported = TextCrossEncoder._list_supported_models()
        if name not in {description.model for description in supported}:
            TextCrossEncoder.add_custom_model(
                model=name, sources=ModelSource(hf=model), model_file=onnx_file
            )
    base = os.getenv("FASTEMBED_CACHE_PATH", os.path.join(tempfile.gettempdir(), "fastembed_cache"))
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", onnx_file)
    return name, os.path.join(base, "alternate-onnx", slug)


class LocalCrossEncoderReranker:
    """Rerank candidates locally with FastEmbed's TextCrossEncoder."""

    def __init__(
        self,
        settings: AppSettings,
        model: CrossEncoderModel | None = None,
    ) -> None:
        self.settings = settings
        model_name, cache_dir = _resolve_model(settings.reranker_model, settings.reranker_onnx_file)
        self.model = model or TextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            lazy_load=True,
            threads=settings.reranker_threads or None,
        )
        # One forward pass at a time. Two concurrent questions each allocating their
        # own activations is how the container got OOM-killed before; serialized, the
        # peak stays at one batch's worth and each pass gets every core.
        self._model_lock = threading.Lock()
        self._cache_size = int(settings.rerank_cache_size)
        self._cache: OrderedDict[tuple[str, bytes], float] = OrderedDict()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _cache_key(query: str, document: str) -> tuple[str, bytes]:
        return query, hashlib.blake2b(document.encode(), digest_size=16).digest()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return cross-encoder scores for query-document pairs, reusing cached pairs."""

        keys = [self._cache_key(query, document) for document in documents]
        scores: list[float | None] = [None] * len(documents)
        if self._cache_size:
            with self._cache_lock:
                for index, key in enumerate(keys):
                    cached = self._cache.get(key)
                    if cached is not None:
                        self._cache.move_to_end(key)
                        scores[index] = cached
        missing = [index for index, score in enumerate(scores) if score is None]
        if missing:
            fresh = self._score_uncached(query, [documents[index] for index in missing])
            if len(fresh) != len(missing):
                raise RerankingError(
                    f"Reranker score count mismatch: got {len(fresh)}, expected {len(missing)}."
                )
            for index, value in zip(missing, fresh, strict=True):
                scores[index] = value
            if self._cache_size:
                with self._cache_lock:
                    for index, value in zip(missing, fresh, strict=True):
                        self._cache[keys[index]] = value
                        self._cache.move_to_end(keys[index])
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        return [float(score) for score in scores if score is not None]

    def _score_uncached(self, query: str, documents: Sequence[str]) -> list[float]:
        try:
            # The model is constructed with lazy_load=True, so the actual model
            # download/load happens on this first call, not at __init__ — this is
            # the boundary that must translate a load failure into RerankingError.
            with self._model_lock:
                scores = list(
                    self.model.rerank(
                        query=query,
                        documents=documents,
                        batch_size=int(self.settings.reranker_batch_size),
                    )
                )
        except Exception as exc:
            raise RerankingError(f"Reranker model failed: {exc}") from exc
        return [float(score) for score in scores]


class Reranker(Protocol):
    """Reranker surface used by the application pipeline."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankOutcome:
    """Full result of a rerank pass, for callers that need the dropped candidates too.

    ``scored`` is every candidate actually sent to the cross-encoder, sorted
    descending — including ones later dropped by ``rerank_min_score`` or the
    ``rerank_top_k`` cut. ``kept`` is the same list ``rerank_candidates`` has always
    returned: filtered and sliced to the top-K.
    """

    scored: list[RerankedChunk]
    kept: list[RerankedChunk]


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievedChunk],
    reranker: Reranker,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """Rerank the fused top-N candidates, drop low-relevance ones, keep the top-K.

    Thin wrapper over ``rerank_candidates_detailed`` for callers that only need the
    survivors — see that function's docstring for the full scoring/filtering rules.
    """

    return rerank_candidates_detailed(query, candidates, reranker, settings).kept


def rerank_candidates_detailed(
    query: str,
    candidates: Sequence[RetrievedChunk],
    reranker: Reranker,
    settings: AppSettings,
) -> RerankOutcome:
    """Rerank the fused top-N candidates, returning both the scored and kept lists.

    ``rerank_score`` is the raw cross-encoder logit sigmoid-normalized to [0, 1] so
    ``rerank_min_score`` is comparable across reranker models. Because the candidate
    list is sorted descending before filtering, dropping everything below the
    threshold and then slicing to ``rerank_top_k`` is equivalent to slicing first and
    filtering after — either way the result is the sorted prefix that clears the bar,
    which may be shorter than ``rerank_top_k`` or empty.

    ``rerank_candidates`` (the settings field, clamped to ``fused_top_n``) controls
    how many of the fused candidates are actually sent through the cross-encoder — the
    reranker is the most expensive retrieval-side stage, so this is the knob for
    trading candidate coverage against latency without touching the fusion spec itself.
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")
    if not candidates:
        return RerankOutcome(scored=[], kept=[])

    fused_top_n = int(settings.fused_top_n)
    candidates_considered = min(int(settings.rerank_candidates), fused_top_n)
    rerank_top_k = int(settings.rerank_top_k)
    min_score = float(settings.rerank_min_score)
    candidates_to_score = list(candidates[:candidates_considered])
    # Scored with the same "filename › section" prefix the chunk was embedded with, so
    # the reranker can't discard a match that rested on the document or heading name.
    documents = [
        contextual_text(candidate.filename, candidate.section, candidate.text)
        for candidate in candidates_to_score
    ]
    scores = reranker.score(normalized_query, documents)

    if len(scores) != len(candidates_to_score):
        raise RerankingError(
            f"Reranker score count mismatch: got {len(scores)}, "
            f"expected {len(candidates_to_score)}."
        )

    reranked = [
        RerankedChunk(
            point_id=candidate.point_id,
            filename=candidate.filename,
            page=candidate.page,
            section=candidate.section,
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            retrieval_score=candidate.score,
            rerank_score=_sigmoid(float(score)),
            chunk_ordinal=candidate.chunk_ordinal,
        )
        for candidate, score in zip(candidates_to_score, scores, strict=True)
    ]
    sorted_chunks = sorted(
        reranked,
        key=lambda chunk: (-chunk.rerank_score, -chunk.retrieval_score, chunk.chunk_id),
    )
    filtered = [chunk for chunk in sorted_chunks if chunk.rerank_score >= min_score]
    return RerankOutcome(scored=sorted_chunks, kept=filtered[:rerank_top_k])
