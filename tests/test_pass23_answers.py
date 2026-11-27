"""Pass 23: answer path, verify and batch fixes."""

from __future__ import annotations

import threading
from typing import Any

from api.dependencies import clear_dependency_caches
from api.reranking import RerankedChunk
from api.verify import check_sentence, recheck, sentences_of
from tests.routes_http.support import upload_and_wait
from tests.test_pass17_answers import _sse_events
from tests.test_pass21_features import _DOC, _client


def _chunk(text: str, chunk_id: str = "c1") -> RerankedChunk:
    return RerankedChunk(
        point_id=chunk_id,
        filename="a.txt",
        page=None,
        section=None,
        chunk_id=chunk_id,
        text=text,
        retrieval_score=0.0,
        rerank_score=0.0,
    )


def _check(sentence: str, passage: str) -> str:
    return check_sentence(sentence, lambda _: [_chunk(passage)]).status


def test_cjk_numbers_are_read_the_same_on_both_sides() -> None:
    passage = "项目成本为1,000,000元人民币。"
    assert _check("项目成本为2,000,000元人民币。", passage) == "numbers_differ"
    assert _check("项目成本为1,000,000元人民币。", passage) == "supported"
    # Spaced in the answer, unspaced in the document, and the other way round.
    assert _check("每晚酒店费用为 3.5 万元。", "每晚酒店费用为3.5万元。") == "supported"
    assert _check("每晚酒店费用为3.5万元。", "每晚酒店费用为 3.5 万元。") == "supported"
    assert _check("员工每年有30天带薪假期。", "员工每年有28天带薪假期。") == "numbers_differ"


def test_latin_words_written_against_characters_are_words_of_their_own() -> None:
    from api.verify import _claim_words

    words, index_of = _claim_words("使用Kubernetes部署服务。")
    assert "kubernetes" in words
    passage = "我们使用Docker部署服务。"
    index = index_of(passage)
    assert "docker" in index.words(0, len(passage))


def test_cjk_sentences_split_around_their_citations() -> None:
    assert sentences_of("员工每年有28天假期。[1] 加班需要经理批准。[2]") == [
        "员工每年有28天假期。[1]",
        "加班需要经理批准。[2]",
    ]
    assert sentences_of("他说：“报销上限为62欧元。”[1]下一句话在这里。") == [
        "他说：“报销上限为62欧元。”[1]",
        "下一句话在这里。",
    ]
    assert sentences_of("假期为28天。[1-3]加班需要批准。") == [
        "假期为28天。[1-3]",
        "加班需要批准。",
    ]


def test_a_recheck_flags_a_citation_to_no_source_as_the_self_check_does() -> None:
    text = "Staff take annual leave of 28 days after asking their manager."
    checks, _ = recheck(f"{text[:-1]} [9].", {1: [_chunk(text)]})
    assert [check.status for check in checks] == ["unsupported"]


def test_a_recheck_of_five_documents_searches_only_the_one_that_changed(
    monkeypatch: Any,
) -> None:
    from api.pipeline_core import PipelineCore

    answer = "Staff take annual leave of 28 days after asking their manager [1]."
    scopes: list[Any] = []
    original = PipelineCore.retrieve

    def watching(self: Any, question: str, *args: Any, **kwargs: Any) -> Any:
        scopes.append(kwargs.get("filenames"))
        return original(self, question, *args, **kwargs)

    with _client(answer, chunk_size_tokens=200) as client:
        sources = []
        for number in range(1, 6):
            name = f"doc{number}.txt"
            assert upload_and_wait(client, name, _DOC + b" " + name.encode())["state"] == "done"
            asked = client.post(
                "/questions", json={"question": "leave?", "filenames": [name]}
            ).json()
            sources.append(
                {
                    "source_number": number,
                    "filename": name,
                    "chunk_id": asked["sources"][0]["chunk_id"],
                }
            )
        changed = _DOC.replace(b"28", b"35")
        assert upload_and_wait(client, "doc1.txt", changed)["state"] == "done"
        monkeypatch.setattr(PipelineCore, "retrieve", watching)
        result = client.post(
            "/questions/recheck", json={"answer": answer, "sources": sources}
        ).json()
        assert [s["state"] for s in result["sources"]] == ["changed"] + ["unchanged"] * 4
        assert scopes == [["doc1.txt"]]
    clear_dependency_caches()


def test_a_continue_that_brings_nothing_leaves_the_answer_cut_off() -> None:
    from api.dependencies import get_generator
    from api.llm import reset_finish_reason

    class _CutThenNothing:
        calls = 0

        def complete(self, messages: Any, settings: Any) -> str:
            return "".join(self.stream(messages, settings))

        def stream(self, messages: Any, settings: Any) -> Any:
            type(self).calls += 1
            if type(self).calls == 1:
                yield "Staff take annual leave of 28 days [1] and"
                reset_finish_reason("length")
            else:
                reset_finish_reason("stop")

    with _client("unused", llm_auto_continue=0) as client:
        client.app.dependency_overrides[get_generator] = lambda: _CutThenNothing()
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        first = _sse_events(client.post("/questions/stream", json={"question": "leave?"}).text)
        done = next(e for e in first if e["type"] == "done")
        assert done["truncated"] is True
        for _ in range(2):  # still cut off, so it can be continued again
            events = _sse_events(
                client.post("/questions/continue", json={"trace_id": done["trace_id"]}).text
            )
            done = next(e for e in events if e["type"] == "done")
            assert done["truncated"] is True, done
    clear_dependency_caches()


def test_a_question_with_no_neighbour_text_is_not_checked_against_neighbours() -> None:
    from api.pipeline import AnswerOverrides
    from api.pipeline_core import PipelineCore

    pipeline = PipelineCore.__new__(PipelineCore)
    from tests.factories import make_test_settings

    pipeline._settings = make_test_settings(context_neighbor_radius=1)
    assert pipeline.widener(AnswerOverrides(passages_only=True)) is None
    assert pipeline.widener(AnswerOverrides()) is not None


def test_a_users_cancel_lets_the_question_being_answered_finish() -> None:
    from api.batch_jobs import BatchJobStore, run_batch_job

    job = BatchJobStore().add("c", ["a", "b"])
    assert job is not None
    stopped: list[bool] = []

    def answer(question: str, should_stop: Any) -> str:
        job.cancel.set()
        stopped.append(should_stop())
        return question

    run_batch_job(job, lambda: lambda: None, answer, lambda *_: None)
    assert stopped == [False] and job.answers == ["a"] and job.state == "cancelled"


def test_a_deleted_workspace_or_shutdown_stops_the_question_being_answered() -> None:
    from api.batch_jobs import BatchJobStore, run_batch_job

    store = BatchJobStore()
    job, other = store.add("gone", ["a", "b"]), store.add("stays", ["a"])
    assert job is not None and other is not None

    def answer(question: str, should_stop: Any) -> str:
        store.abort("gone")
        assert should_stop()
        raise RuntimeError("abandoned")

    run_batch_job(job, lambda: lambda: None, answer, lambda *_: "failed")
    assert job.answers == [] and job.state == "cancelled"
    assert not other.cancel.is_set()
    store.abort()
    assert other.abort.is_set()


def test_batches_are_capped_per_workspace_and_in_all() -> None:
    from api.batch_jobs import (
        MAX_RUNNING_BATCH_JOBS,
        MAX_RUNNING_BATCH_JOBS_TOTAL,
        BatchJobStore,
    )

    store = BatchJobStore()
    for number in range(MAX_RUNNING_BATCH_JOBS_TOTAL):
        assert store.add(f"w{number // MAX_RUNNING_BATCH_JOBS}", ["q"]) is not None
    assert store.add("another", ["q"]) is None
    full = BatchJobStore()
    for _ in range(MAX_RUNNING_BATCH_JOBS):
        assert full.add("one", ["q"]) is not None
    assert full.add("one", ["q"]) is None and full.add("two", ["q"]) is not None


def test_batches_always_leave_a_question_slot_free() -> None:
    from api.admission import QuestionSlots

    slots = QuestionSlots(3)
    held = [slots.try_acquire("batch") for _ in range(3)]
    assert [release is not None for release in held] == [True, True, False]
    chat = slots.try_acquire("question")
    assert chat is not None
    assert slots.try_acquire("question") is None
    chat()
    first = held[0]
    assert first is not None
    first()
    assert slots.try_acquire("batch") is not None
    # One slot: a batch may take it.
    assert QuestionSlots(1).try_acquire("batch") is not None
    # A refused batch gives back its own place.
    busy = QuestionSlots(2)
    chats = [busy.try_acquire(), busy.try_acquire()]
    assert busy.try_acquire("batch") is None
    for release in chats:
        assert release is not None
        release()
    assert busy.try_acquire("batch") is not None
    assert QuestionSlots(0).try_acquire("batch") is not None


def test_a_batch_read_as_done_has_all_its_answers() -> None:
    from api.batch_jobs import BatchJob
    from api.routes.questions import _batch_job_response

    class _Finishing(list):  # type: ignore[type-arg]
        """The job's thread finishes just as the answers are copied."""

        job: BatchJob

        def __iter__(self) -> Any:
            snapshot = list.copy(self)
            self.job.state = "done"
            return iter(snapshot)

    answers = _Finishing()
    job = BatchJob("id", "c", ["a"], answers=answers, state="running")
    answers.job = job
    assert _batch_job_response(job).state == "running"


def test_a_batchs_traces_keep_the_request_id() -> None:
    from api.batch_jobs import BatchJob
    from api.request_id import request_id_var
    from api.routes.questions import _start_batch_thread

    seen: list[Any] = []
    done = threading.Event()

    def answer(question: str, should_stop: Any) -> str:
        seen.append(request_id_var.get())
        done.set()
        return question

    token = request_id_var.set("req-23")
    try:
        _start_batch_thread(BatchJob("id", "c", ["a"]), lambda: lambda: None, answer, None)
    finally:
        request_id_var.reset(token)
    assert done.wait(5) and seen == ["req-23"]


def test_two_batches_sharing_one_slot_take_turns() -> None:
    from api.admission import QuestionSlots
    from api.batch_jobs import BATCH_YIELD_SECONDS, BatchJobStore, run_batch_job

    slots = QuestionSlots(2)  # one for batches
    store = BatchJobStore()
    jobs = [store.add("c", [name] * 3) for name in ("a", "b")]
    order: list[str] = []

    def answer(question: str, should_stop: Any) -> str:
        order.append(question)
        threading.Event().wait(0.05)
        return question

    runners = [
        threading.Thread(
            target=run_batch_job,
            args=(job, lambda: slots.try_acquire("batch"), answer, None, BATCH_YIELD_SECONDS),
        )
        for job in jobs
    ]
    for runner in runners:
        runner.start()
        threading.Event().wait(0.01)
    for runner in runners:
        runner.join(20)
    # Neither waits for the other to finish all of its own.
    assert len(order) == 6 and set(order[:3]) == {"a", "b"}, order


def test_with_one_slot_a_batch_stands_aside_for_a_refused_question() -> None:
    from api.admission import BATCH_STAND_ASIDE_SECONDS, QuestionSlots

    now = [100.0]
    slots = QuestionSlots(1, clock=lambda: now[0])
    batch = slots.try_acquire("batch")
    assert batch is not None and slots.try_acquire("question") is None  # told to retry
    batch()
    now[0] += 5  # the batch's next question must not take it before the retry
    assert slots.try_acquire("batch") is None
    chat = slots.try_acquire("question")
    assert chat is not None
    chat()
    # Served: the batch goes on at once.
    again = slots.try_acquire("batch")
    assert again is not None and slots.try_acquire("question") is None
    again()
    now[0] += BATCH_STAND_ASIDE_SECONDS + 0.1  # nobody came back
    assert slots.try_acquire("batch") is not None
    # With a slot to spare there is nothing to stand aside from.
    two = QuestionSlots(2, clock=lambda: now[0])
    held = [two.try_acquire("question"), two.try_acquire("question")]
    assert two.try_acquire("question") is None
    first = held[0]
    assert first is not None
    first()
    assert two.try_acquire("batch") is not None
