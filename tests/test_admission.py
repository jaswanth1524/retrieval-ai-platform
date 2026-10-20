from __future__ import annotations

from api.admission import QuestionSlots


def test_slots_refuse_past_the_cap_and_come_back_on_release() -> None:
    slots = QuestionSlots(2)
    first = slots.try_acquire()
    second = slots.try_acquire()

    assert first is not None and second is not None
    assert slots.try_acquire() is None
    first()
    first()  # idempotent: a second release must not free an extra slot
    third = slots.try_acquire()
    assert third is not None
    assert slots.try_acquire() is None
    second()
    third()  # the in-use gauge is process-wide: leave it as it was


def test_a_zero_cap_never_refuses() -> None:
    slots = QuestionSlots(0)

    releases = [slots.try_acquire() for _ in range(50)]

    assert all(release is not None for release in releases)
    for release in releases:
        assert release is not None
        release()


def test_summaries_can_never_take_every_slot() -> None:
    """Summaries run for minutes: with MAX_CONCURRENT_QUESTIONS of them running, every
    question got 429 until one finished."""

    slots = QuestionSlots(4, summary_limit=2)
    summaries = [slots.try_acquire("summary"), slots.try_acquire("summary")]

    assert all(release is not None for release in summaries)
    assert slots.try_acquire("summary") is None  # the summaries' own cap
    questions = [slots.try_acquire(), slots.try_acquire()]
    assert all(release is not None for release in questions)
    assert slots.try_acquire() is None  # the overall cap still holds

    first_summary = summaries[0]
    assert first_summary is not None
    first_summary()
    first_summary()  # idempotent for both semaphores
    # A summary slot came back, and so did the overall one it held.
    third = slots.try_acquire("summary")
    assert third is not None
    assert slots.try_acquire("summary") is None
    for release in [*summaries, *questions, third]:
        assert release is not None
        release()  # the in-use gauge is process-wide


def test_a_summary_refused_by_the_overall_cap_gives_its_own_slot_back() -> None:
    slots = QuestionSlots(1, summary_limit=1)
    question = slots.try_acquire()
    assert question is not None

    assert slots.try_acquire("summary") is None
    question()
    # The failed attempt didn't keep the only summary slot.
    summary = slots.try_acquire("summary")
    assert summary is not None
    summary()
