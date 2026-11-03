"""Nothing an answer carries, and no per-request knob, can be left out of the cache."""

from __future__ import annotations

import dataclasses
from typing import Any

from api.answer_cache import CachedAnswer
from api.citations import SourceCitation
from api.generation import StageTimings
from api.pipeline import AnswerOverrides, RagPipeline, replay_cached
from api.repository import PassageFilter
from tests.factories import make_test_settings

SOURCE = SourceCitation(
    source_number=1,
    filename="a.txt",
    page=1,
    section="Intro",
    chunk_id="c1",
    text="Alpha.",
)
TIMINGS = StageTimings(embed_ms=0, search_ms=0, rerank_ms=0, generate_ms=0, total_ms=0)

# A value for every field other than its default.
NOT_DEFAULT: dict[str, Any] = {
    "answer": "Alpha [1].",
    "sources": (SOURCE,),
    "trace_id": "t1",
    "answer_status": "uncited",
    "top_rerank_score": 0.5,
    "truncated": True,
    "weak_matches": True,
    "citation_backfilled": True,
    "passages_left_out": 2,
    "coverage": (3, 9),
    "followups": ("Why?",),
}


def test_every_field_survives_each_way_an_answer_is_cached_and_served() -> None:
    names = {field.name for field in dataclasses.fields(CachedAnswer)}
    assert names == set(NOT_DEFAULT), "add the new field to NOT_DEFAULT and the converters"

    answer = CachedAnswer(**{**NOT_DEFAULT, "coverage": None})
    assert CachedAnswer.from_grounded(answer.as_grounded(TIMINGS)) == answer

    # Through the stream: the replayed `done` frame stored again, follow-ups after it.
    done = next(
        event
        for event in replay_cached(
            answer,
            0.0,
            top_rerank_score=answer.top_rerank_score,
            weak_matches=answer.weak_matches,
        )
        if event["type"] == "done"
    )
    assert CachedAnswer.from_done(done, answer.followups) == answer

    summary = CachedAnswer(
        **{**NOT_DEFAULT, "top_rerank_score": None, "weak_matches": False, "followups": ()}
    )
    done = next(
        event
        for event in replay_cached(
            summary, 0.0, chunks_read=summary.coverage[0], chunks_total=summary.coverage[1]
        )
        if event["type"] == "done"
    )
    assert CachedAnswer.from_done(done) == summary


def test_every_answer_override_changes_the_cache_key() -> None:
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline._settings = make_test_settings(fused_top_n=50)  # noqa: SLF001
    pipeline._answer_cache = object()  # type: ignore[assignment]  # noqa: SLF001
    changed: dict[str, Any] = {
        "llm_provider": "openai_compatible",
        "rerank_top_k": 7,
        "max_context_chunks": 1,
        "llm_temperature": 0.7,
        "answer_format": "bullets",
        "answer_language": "French",
        "narrow": PassageFilter(page_min=2),
        "passages_only": True,
        "strong_matches_only": True,
    }
    assert {field.name for field in dataclasses.fields(AnswerOverrides)} == set(changed)
    pipeline._settings = pipeline._settings.model_copy(  # noqa: SLF001
        update={
            "openai_compatible_base_url": "http://127.0.0.1:8080/v1",
            "openai_compatible_model": "local",
        }
    )
    plain = pipeline._cache_key("q", AnswerOverrides(), None, None, None)  # noqa: SLF001
    for name, value in changed.items():
        key = pipeline._cache_key("q", AnswerOverrides(**{name: value}), None, None, None)  # noqa: SLF001
        assert key is not None and key != plain, name
