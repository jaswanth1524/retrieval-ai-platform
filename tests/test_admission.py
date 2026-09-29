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


def test_a_zero_cap_never_refuses() -> None:
    slots = QuestionSlots(0)

    releases = [slots.try_acquire() for _ in range(50)]

    assert all(release is not None for release in releases)
