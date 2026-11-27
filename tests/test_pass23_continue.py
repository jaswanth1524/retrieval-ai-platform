"""Pass 23: joining a continuation (found live with gpt-4o-mini, older than the pass)."""

from __future__ import annotations

from typing import Any

import pytest

from api.continuation import drop_restart, join_continuation
from api.dependencies import clear_dependency_caches, get_generator
from api.llm import reset_finish_reason
from tests.routes_http.support import upload_and_wait
from tests.test_pass17_answers import _sse_events
from tests.test_pass21_features import _DOC, _client

# As captured: the 40-token limit cut just after "[", and the model, asked to go on,
# started at the next item.
_CUT = (
    "1. 机票须提前14天通过指定代理预订。[1]  \n2. 默认乘坐经济舱，超过9小时的航班可以乘坐商务舱。["
)
_NEXT = "3. 海外每日餐费补贴为60欧元。[1]"
_JOINED = _CUT[:-1] + "\n" + _NEXT


def _join(before: str, after: str) -> str:
    return join_continuation(before, drop_restart(before, after))


@pytest.mark.parametrize(
    ("before", "after", "joined"),
    [
        (_CUT, _NEXT, _JOINED),
        (_CUT + "1", _NEXT, _JOINED),
        (_CUT + "1, ", _NEXT, _JOINED),
        (_CUT[:-1] + "【1", _NEXT, _JOINED),
        # Cut before the marker: the next item still starts its own line.
        (_CUT[:-1], _NEXT, _JOINED),
        (_CUT, "\n" + _NEXT, _JOINED),
        (
            "1. Book early [1].\n2. Business class is allowed over nine hours. [",
            "3. The meal allowance is sixty euros. [1]",
            "1. Book early [1].\n2. Business class is allowed over nine hours.\n"
            "3. The meal allowance is sixty euros. [1]",
        ),
        (
            "- Book early [1].\n- Fly economy [",
            "- Keep receipts [1].",
            "- Book early [1].\n- Fly economy\n- Keep receipts [1].",
        ),
        # The model finishes the marker: joined as it is.
        (_CUT, "1]  \n3. 海外", _CUT + "1]  \n3. 海外"),
        # Not the next item: a link's text, a figure, another number.
        ("2. See [", "the policy](http://x)", "2. See [the policy](http://x)"),
        ("2. It costs [", "3.5 million](http://x)", "2. It costs [3.5 million](http://x)"),
        ("Staff may see [", "1. the list", "Staff may see [1. the list"),
        # Chinese and Japanese write no space between words.
        ("5. 出差结束后", "30天内提交报销。[1]", "5. 出差结束后30天内提交报销。[1]"),
        ("補助は60", "ユーロです。", "補助は60ユーロです。"),
        ("假期为28天。[1]", "此外还有病假。", "假期为28天。[1]此外还有病假。"),
        # A clause written again from its punctuation is written once.
        (
            "规定如下。海外每日餐费补贴为",
            "海外每日餐费补贴为60欧元。",
            "规定如下。海外每日餐费补贴为60欧元。",
        ),
        (
            "需提前预订[1]。默认乘坐经济舱，",
            "默认乘坐经济舱，可乘商务舱。",
            "需提前预订[1]。默认乘坐经济舱，可乘商务舱。",
        ),
        # Too short to tell from a word doubled on purpose.
        ("请你，看看", "看看这里", "请你，看看看看这里"),
        # Korean and English keep their spaces.
        ("휴가는", "28일입니다", "휴가는 28일입니다"),
        ("Staff get", "28 days.", "Staff get 28 days."),
    ],
)
def test_a_continuation_is_joined_as_the_text_reads(before: str, after: str, joined: str) -> None:
    assert _join(before, after) == joined


def test_a_prefilled_continuation_is_joined_as_written() -> None:
    assert join_continuation(_CUT, "1]  \n3. 海外", prefilled=True) == _CUT + "1]  \n3. 海外"


class _CutAtBracket:
    """The first reply is cut just after "[", the next starts at the next item."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Any, settings: Any) -> str:
        return "".join(self.stream(messages, settings))

    def stream(self, messages: Any, settings: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield "1. Staff take annual leave of 28 days after asking their manager [1].\n"
            yield "2. Staff take annual leave of 28 days. ["
            reset_finish_reason("length")
        else:
            yield "3. Staff ask their manager before annual leave [1]."
            reset_finish_reason("stop")


_WHOLE = (
    "1. Staff take annual leave of 28 days after asking their manager [1].\n"
    "2. Staff take annual leave of 28 days.\n"
    "3. Staff ask their manager before annual leave [1]."
)


@pytest.mark.parametrize("auto", [0, 1])
def test_a_stream_cut_at_a_bracket_continues_into_readable_text(auto: int) -> None:
    # OpenAI takes no prefill, so the continuation is asked for in words.
    with _client(
        "unused", llm_provider="openai", openai_api_key="x", auto_continue_max=auto
    ) as client:
        generator = _CutAtBracket()
        client.app.dependency_overrides[get_generator] = lambda: generator
        assert upload_and_wait(client, "leave.txt", _DOC)["state"] == "done"
        events = _sse_events(client.post("/questions/stream", json={"question": "leave?"}).text)
        done = next(e for e in events if e["type"] == "done")
        if not auto:
            assert done["truncated"]
            events = _sse_events(
                client.post("/questions/continue", json={"trace_id": done["trace_id"]}).text
            )
            done = next(e for e in events if e["type"] == "done")
        else:
            # What is streamed is the new text only: nothing sent twice.
            sent = "".join(e["text"] for e in events if e["type"] == "delta")
            assert sent.count("Staff ask their manager") == 1 and sent.count("2. Staff") == 1
        assert done["answer"] == _WHOLE and not done["truncated"], done["answer"]
        trace = client.get(f"/traces/{done['trace_id']}").json()
        assert trace["answer"] == _WHOLE
    clear_dependency_caches()


def test_the_sync_route_joins_it_the_same_way() -> None:
    from api.continuation import AutoContinuing
    from tests.factories import make_test_settings

    text = AutoContinuing(_CutAtBracket(), 1).complete(
        [{"role": "user", "content": "leave?"}],
        make_test_settings(llm_provider="openai", openai_api_key="x"),
    )
    assert text == _WHOLE
    reset_finish_reason()
