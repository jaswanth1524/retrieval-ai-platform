"""Pass 24: a loose replay hit's citations follow its passages when they reorder."""

from __future__ import annotations

import json
from pathlib import Path

from eval.replay import (
    RecordingGenerator,
    ReplayGenerator,
    renumber_citations,
    source_fingerprints,
)

CURRENT = (
    "[{n}] filename=travel_policy.md; page=1; section=Per diem\n"
    "The daily meal allowance abroad is sixty euros."
)
ARCHIVED = (
    "[{n}] filename=travel_policy_2019_archived.md; page=1; section=Per diem\n"
    "The 2019 daily meal allowance abroad was forty-five euros."
)


def _messages(first: str, second: str) -> list[dict[str, str]]:
    context = first.format(n=1) + "\n\n" + second.format(n=2)
    return [
        {"role": "system", "content": "Answer only from the context."},
        {
            "role": "user",
            "content": (
                "Question:\nWhat is the meal allowance?\n\nContext:\n"
                f"{context}\n\nInstructions:\n- Cite sources."
            ),
        },
    ]


class _Model:
    def complete(self, messages, settings):  # type: ignore[no-untyped-def]
        return "It is sixty euros [1]; it was forty-five in 2019 [2]."

    def stream(self, messages, settings):  # type: ignore[no-untyped-def]
        yield self.complete(messages, settings)


def test_passages_are_fingerprinted_by_header_and_text_not_number() -> None:
    before = source_fingerprints(_messages(CURRENT, ARCHIVED))  # type: ignore[arg-type]
    after = source_fingerprints(_messages(ARCHIVED, CURRENT))  # type: ignore[arg-type]
    assert before[1] == after[2] and before[2] == after[1] and before[1] != before[2]


def test_a_reordered_prompt_gets_the_reply_with_its_citations_moved(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    recorder = RecordingGenerator(_Model(), path)  # type: ignore[arg-type]
    recorder.complete(_messages(CURRENT, ARCHIVED), None)  # type: ignore[arg-type]
    assert "sources" in json.loads(path.read_text())

    replay = ReplayGenerator(path)
    same = replay.complete(_messages(CURRENT, ARCHIVED), None)  # type: ignore[arg-type]
    assert same == "It is sixty euros [1]; it was forty-five in 2019 [2]."
    swapped = replay.complete(_messages(ARCHIVED, CURRENT), None)  # type: ignore[arg-type]
    assert swapped == "It is sixty euros [2]; it was forty-five in 2019 [1]."
    assert (replay.loose_hits, replay.renumbered) == (1, 1)


def test_an_older_recording_without_fingerprints_replays_as_before(tmp_path: Path) -> None:
    path = tmp_path / "replies.jsonl"
    RecordingGenerator(_Model(), path).complete(_messages(CURRENT, ARCHIVED), None)  # type: ignore[arg-type]
    entry = json.loads(path.read_text())
    del entry["sources"]
    path.write_text(json.dumps(entry) + "\n")
    replay = ReplayGenerator(path)
    swapped = replay.complete(_messages(ARCHIVED, CURRENT), None)  # type: ignore[arg-type]
    assert swapped == "It is sixty euros [1]; it was forty-five in 2019 [2]."
    assert replay.renumbered == 0


def test_lists_are_renumbered_and_a_passage_gone_keeps_its_number() -> None:
    recorded = {1: "a", 2: "b", 3: "c"}
    assert renumber_citations("x [1, 3] y [2]", recorded, {1: "c", 2: "b", 3: "a"}) == (
        "x [3, 1] y [2]"
    )
    assert renumber_citations("x [1]", recorded, {1: "a"}) is None
    assert renumber_citations("x [1] [3]", recorded, {2: "a"}) == "x [2] [3]"
