"""Pass 21: declines recognised in the canned answer's newer languages."""

from __future__ import annotations

import pytest

from api.citations import states_insufficiency
from api.prompting import insufficient_context_answer
from tests.test_pipeline import make_settings

NEWER = ["dutch", "polish", "swedish", "japanese", "chinese", "korean", "hindi"]


@pytest.mark.parametrize("language", NEWER)
def test_the_canned_answer_in_each_newer_language_reads_as_a_decline(language: str) -> None:
    canned = insufficient_context_answer(make_settings(answer_language=language))
    assert states_insufficiency(canned), canned


@pytest.mark.parametrize(
    "answer",
    [
        "Medewerkers krijgen 28 dagen verlof per jaar [1].",
        "Pracownicy otrzymują 28 dni urlopu rocznie [1].",
        "Anställda får 28 dagars semester per år [1].",
        "従業員は年間28日の休暇を取得できます [1]。",
        "员工每年有28天假期，需要提前申请 [1]。",
        "직원은 매년 28일의 휴가를 받습니다 [1].",
        "कर्मचारियों को हर साल 28 दिन की छुट्टी मिलती है [1]।",
        "The form does not contain a signature field [1].",
    ],
)
def test_ordinary_answers_in_those_languages_are_not_declines(answer: str) -> None:
    assert not states_insufficiency(answer)


@pytest.mark.parametrize(
    "decline",
    [
        "De documenten bevatten onvoldoende informatie over dit onderwerp.",
        "文書には十分な情報がありません。",
        "根据提供的文档，无法回答这个问题。",
        "문서에 관련 정보가 부족합니다.",
    ],
)
def test_a_model_written_decline_in_those_languages_is_recognised(decline: str) -> None:
    assert states_insufficiency(decline)
