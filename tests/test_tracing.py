from __future__ import annotations

from api.reranking import RerankedChunk
from api.retrieval import RetrievedChunk
from api.tracing import QueryTrace, TraceStore, build_trace_candidates


def _retrieved(point_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=point_id,
        filename="doc.txt",
        page=1,
        section="Intro",
        chunk_id=f"chunk-{point_id}",
        text="text",
        score=score,
    )


def _reranked(point_id: str, rerank_score: float, *, expanded: bool = False) -> RerankedChunk:
    return RerankedChunk(
        point_id=point_id,
        filename="doc.txt",
        page=1,
        section="Intro",
        chunk_id=f"chunk-{point_id}",
        text="text",
        retrieval_score=0.5,
        rerank_score=rerank_score,
        expanded_text="expanded" if expanded else None,
    )


def _trace(trace_id: str) -> QueryTrace:
    return QueryTrace(
        trace_id=trace_id,
        created_at=0.0,
        question="What is the refund window?",
        mode="sync",
        status="ok",
        config=None,
        candidates=[],
        prompt_messages=None,
        answer="Refunds within 30 days [1].",
        cited_source_numbers=[1],
        timings={"total_ms": 42.0},
        error=None,
    )


def test_trace_store_add_and_get_roundtrip() -> None:
    store = TraceStore(max_retained=10)
    trace = _trace("t1")

    store.add(trace)

    fetched = store.get("t1")
    assert fetched is not None
    assert fetched.question == "What is the refund window?"
    assert fetched.answer == "Refunds within 30 days [1]."


def test_query_trace_condense_fields_default_to_none_and_zero() -> None:
    trace = _trace("t1")

    assert trace.condensed_question is None
    assert trace.history_message_count == 0


def test_query_trace_condense_fields_roundtrip_through_the_store() -> None:
    store = TraceStore(max_retained=10)
    trace = QueryTrace(
        trace_id="t1",
        created_at=0.0,
        question="What about the second one?",
        mode="sync",
        status="ok",
        config=None,
        candidates=[],
        prompt_messages=None,
        answer="Doc B covers billing [1].",
        cited_source_numbers=[1],
        timings={"total_ms": 42.0, "condense_ms": 5.0},
        error=None,
        condensed_question="What is the second document about?",
        history_message_count=2,
    )

    store.add(trace)

    fetched = store.get("t1")
    assert fetched is not None
    assert fetched.condensed_question == "What is the second document about?"
    assert fetched.history_message_count == 2


def test_trace_store_get_returns_none_for_unknown_id() -> None:
    store = TraceStore(max_retained=10)

    assert store.get("missing") is None


def test_trace_store_get_returns_a_snapshot_not_the_live_object() -> None:
    store = TraceStore(max_retained=10)
    trace = _trace("t1")
    store.add(trace)

    snapshot = store.get("t1")
    trace.answer = "mutated after add"

    assert snapshot is not None
    assert snapshot.answer == "Refunds within 30 days [1]."


def test_trace_store_list_traces_returns_newest_first() -> None:
    store = TraceStore(max_retained=10)
    store.add(_trace("t1"))
    store.add(_trace("t2"))
    store.add(_trace("t3"))

    ids = [trace.trace_id for trace in store.list_traces()]

    assert ids == ["t3", "t2", "t1"]


def test_trace_store_prunes_oldest_beyond_retention_limit() -> None:
    store = TraceStore(max_retained=5)
    for i in range(6):
        store.add(_trace(f"t{i}"))

    assert store.get("t0") is None
    assert store.get("t5") is not None
    assert len(store.list_traces()) == 5


def test_build_trace_candidates_marks_selected_and_expanded_chunks() -> None:
    fused = [_retrieved("a", 0.9), _retrieved("b", 0.5)]
    scored = [_reranked("a", 0.8), _reranked("b", 0.2)]
    kept = [scored[0]]
    selected = [_reranked("a", 0.8, expanded=True)]

    candidates = build_trace_candidates(fused, scored, kept, selected, min_score=0.3)

    by_id = {c.point_id: c for c in candidates}
    assert by_id["a"].kept is True
    assert by_id["a"].drop_reason is None
    assert by_id["a"].selected_for_context is True
    assert by_id["a"].neighbor_expanded is True
    assert by_id["b"].kept is False
    assert by_id["b"].drop_reason == "below_min_score"
    assert by_id["b"].selected_for_context is False
    assert by_id["b"].neighbor_expanded is False


def test_build_trace_candidates_top_k_cut_reason() -> None:
    fused = [_retrieved("a", 0.9)]
    scored = [_reranked("a", 0.8)]
    kept: list[RerankedChunk] = []  # scored above min_score but cut by rerank_top_k
    selected: list[RerankedChunk] = []

    candidates = build_trace_candidates(fused, scored, kept, selected, min_score=0.1)

    assert candidates[0].drop_reason == "top_k_cut"


def test_build_trace_candidates_not_scored_reason() -> None:
    fused = [_retrieved("a", 0.9), _retrieved("b", 0.1)]
    scored = [_reranked("a", 0.8)]  # "b" never sent to the cross-encoder
    kept = [scored[0]]
    selected: list[RerankedChunk] = []

    candidates = build_trace_candidates(fused, scored, kept, selected, min_score=0.1)

    by_id = {c.point_id: c for c in candidates}
    assert by_id["b"].rerank_score is None
    assert by_id["b"].drop_reason == "not_scored"
