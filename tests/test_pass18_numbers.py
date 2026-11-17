"""Pass 18: citing sentences whose numbers no cited passage states."""

from __future__ import annotations

from api.citations import unsupported_numbers
from tests.test_generation import make_chunk

TRAVEL = make_chunk("c1", "Employees may claim up to 1,000 euros for travel each quarter.")
LEAVE = make_chunk("c2", "Annual leave is twenty-five days, plus five days of sick leave.")


def test_a_wrong_number_is_flagged_and_a_right_one_is_not() -> None:
    chunks = [TRAVEL, LEAVE]
    assert unsupported_numbers("Travel is capped at 1000 euros a quarter [1].", chunks) == 0
    assert unsupported_numbers("Travel is capped at 1,500 euros a quarter [1].", chunks) == 1
    assert unsupported_numbers("Leave is 25 days, plus 5 sick days [2].", chunks) == 0
    # The right figure, but from a passage this sentence doesn't cite.
    assert unsupported_numbers("Leave is 25 days [1].", chunks) == 1


def test_list_numbers_uncited_sentences_and_bad_markers_are_not_claims() -> None:
    chunks = [TRAVEL, LEAVE]
    answer = "1. Book travel within 1,000 euros [1].\n2. Take 25 days of leave [2]."
    assert unsupported_numbers(answer, chunks) == 0
    assert unsupported_numbers("There are 7 offices.", chunks) == 0
    assert unsupported_numbers("There are 7 offices [9].", chunks) == 0
    # A marker after the full stop belongs to the sentence before it.
    assert unsupported_numbers("Leave is 30 days. [2] That is all.", chunks) == 1


def test_an_answer_with_a_wrong_number_says_so_on_done_and_the_response() -> None:
    import json
    from collections.abc import Sequence

    from fastapi.testclient import TestClient
    from qdrant_client import QdrantClient

    from api.dependencies import clear_dependency_caches, get_generator
    from api.generation import ChatMessage
    from api.settings import AppSettings
    from tests.routes_http.support import hermetic_app, make_settings, upload_and_wait

    class Wrong:
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            return "Annual leave is 30 days [1]."

        def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):  # type: ignore[no-untyped-def]
            yield self.complete(messages, settings)

    clear_dependency_caches()
    app = hermetic_app(make_settings(citation_retry_enabled=False), QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: Wrong()
    with TestClient(app) as client:
        upload_and_wait(client, "leave.txt", b"Intro\nAnnual leave is twenty-five days.")
        sync = client.post("/questions", json={"question": "How long is annual leave?"}).json()
        assert sync["unsupported_numbers"] == 1
        stream = client.post("/questions/stream", json={"question": "How long is leave?"}).text
        done = next(
            json.loads(line[6:])
            for line in stream.splitlines()
            if line.startswith("data: ") and '"type": "done"' in line
        )
        assert done["unsupported_numbers"] == 1
        trace = client.get(f"/traces/{done['trace_id']}").json()
        assert trace["unsupported_numbers"] == 1
    clear_dependency_caches()


def test_a_number_naming_the_cited_document_is_supported() -> None:
    # gpt-5-mini, comparing the current and the 2019 travel policies: "The 2019 file is
    # marked archived [7]" — 2019 is in the file name and heading, not the passage.
    from api.citations import unsupported_numbers
    from api.reranking import RerankedChunk

    archived = RerankedChunk(
        point_id="a",
        filename="travel_policy_2019_archived.md",
        page=1,
        section="Status",
        chunk_id="a",
        text="Archived. This version was replaced by the current travel policy.",
        retrieval_score=0.0,
        rerank_score=0.0,
    )
    assert unsupported_numbers("The 2019 file is marked archived [1].", [archived]) == 0
    assert unsupported_numbers("The 2020 file is marked archived [1].", [archived]) == 1
    # The page is not a source of numbers: a figure changed to 1 still shows.
    assert unsupported_numbers("Archived after 1 review [1].", [archived]) == 1


def test_a_count_of_the_sources_and_an_amount_in_cents_are_supported() -> None:
    # From gpt-5-mini's brief of three policies (pass 18, live).
    from api.citations import unsupported_numbers
    from api.reranking import RerankedChunk

    def chunk(text: str) -> RerankedChunk:
        return RerankedChunk(
            point_id=text[:8],
            filename="expense_policy.md",
            page=1,
            section="Mileage",
            chunk_id=text[:8],
            text=text,
            retrieval_score=0.0,
            rerank_score=0.0,
        )

    review = chunk("The company reviews the policy every year.")
    assert (
        unsupported_numbers("All three say the policy is reviewed every year [1].", [review]) == 0
    )
    assert unsupported_numbers("The two documents agree on the review [1].", [review]) == 0
    # A count that is a claim about the passage still shows.
    assert unsupported_numbers("The policy is reviewed three times a year [1].", [review]) == 1
    mileage = chunk("Mileage for a private car is paid at thirty cents per kilometre.")
    assert unsupported_numbers("Mileage is reimbursed at €0.30 per kilometre [1].", [mileage]) == 0
    assert unsupported_numbers("Mileage is reimbursed at €0.35 per kilometre [1].", [mileage]) == 1


def test_a_number_spelled_with_a_typographic_hyphen_is_read_whole() -> None:
    from api.citations import _numbers

    assert _numbers("forty‑five euros") == {"45"}
    assert _numbers("forty‐five euros") == {"45"}
