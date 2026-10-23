"""Local-model behaviour of the LLM layer: prompt-scaled timeouts, the Ollama
generation slot, and non-streamed calls streamed under the hood."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from api import llm
from api.llm import (
    ChatMessage,
    GenerationError,
    GenerationSlot,
    LiteLLMGenerator,
    expected_prefill_ms,
    measured_prefill_rate,
    request_timeout_seconds,
    usage_totals,
)
from api.prompting import estimate_prompt_tokens
from api.settings import AppSettings
from tests.factories import make_test_settings

LONG: list[ChatMessage] = [{"role": "user", "content": "word " * 3000}]


def ollama(**overrides: Any) -> AppSettings:
    return make_test_settings(llm_provider="ollama", **overrides)


def chunk(text: str = "", finish: str | None = None, usage: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}
    if usage is not None:
        out["usage"] = usage
    return out


class StreamingClient:
    """A completion client that streams fixed chunks, optionally after a delay."""

    def __init__(self, chunks: list[dict[str, Any]], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        self.calls.append(kwargs)

        def gen() -> Iterator[dict[str, Any]]:
            if self.delay:
                time.sleep(self.delay)
            yield from self.chunks

        return gen()


def test_a_local_call_gets_time_to_read_its_prompt_and_a_hosted_one_does_not() -> None:
    tokens = estimate_prompt_tokens(LONG)
    assert request_timeout_seconds(LONG, ollama()) == pytest.approx(60 + tokens / 40)
    compatible = make_test_settings(
        llm_provider="openai_compatible",
        openai_compatible_base_url="http://x/v1",
        openai_compatible_model="m",
    )
    assert request_timeout_seconds(LONG, compatible) == pytest.approx(60 + tokens / 40)
    assert request_timeout_seconds(LONG, ollama(llm_prefill_tokens_per_second=0)) == 60
    openai = make_test_settings(llm_provider="openai", openai_api_key="k")
    assert request_timeout_seconds(LONG, openai) == 60


def test_a_slower_measured_rate_lengthens_the_timeout_and_gives_an_eta() -> None:
    settings = ollama()
    assert expected_prefill_ms(LONG, settings) is None
    llm._record_prefill(settings, 2000, 100.0)  # 20 tokens/s, slower than the 40 set
    assert measured_prefill_rate(settings) == pytest.approx(20)
    tokens = estimate_prompt_tokens(LONG)
    assert request_timeout_seconds(LONG, settings) == pytest.approx(60 + tokens / 20)
    assert expected_prefill_ms(LONG, settings) == pytest.approx(tokens / 20 * 1000)
    # A faster measurement never shortens the timeout below the configured rate's.
    llm._record_prefill(settings, 200_000, 1.0)
    assert request_timeout_seconds(LONG, settings) == pytest.approx(60 + tokens / 40)
    # Short prompts say little about the read rate and aren't measured.
    llm._measured_rates.clear()
    llm._record_prefill(settings, 100, 10.0)
    assert measured_prefill_rate(settings) is None


def test_ollama_complete_streams_and_joins_with_usage_and_finish_reason() -> None:
    client = StreamingClient(
        [
            chunk("Hello "),
            chunk("world [1].", finish="length"),
            chunk(usage={"prompt_tokens": 900, "completion_tokens": 4}),
        ]
    )
    settings = ollama()
    llm.begin_usage()
    answer = LiteLLMGenerator(settings, completion_client=client).complete(LONG, settings)
    assert answer == "Hello world [1]."
    assert client.calls[0]["stream"] is True
    assert llm.stopped_at_token_limit()
    assert usage_totals() == (900, 4)


def test_hosted_complete_is_one_plain_call() -> None:
    response = {"choices": [{"message": {"content": "Answer [1]."}}]}
    calls: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> object:
        calls.append(kwargs)
        return response

    settings = make_test_settings(llm_provider="openai", openai_api_key="k")
    assert LiteLLMGenerator(settings, completion_client=client).complete(LONG, settings)
    assert "stream" not in calls[0]
    assert calls[0]["timeout"] == 60


def test_a_stream_measures_how_fast_the_prompt_was_read() -> None:
    settings = ollama()
    client = StreamingClient([chunk("a"), chunk("b")], delay=0.2)
    generator = LiteLLMGenerator(settings, completion_client=client)
    assert "".join(generator.stream(LONG, settings)) == "ab"
    rate = measured_prefill_rate(settings)
    assert rate is not None
    assert rate < estimate_prompt_tokens(LONG) / 0.2 * 1.01
    assert client.calls[0]["stream_options"] == {"include_usage": True}


def test_ollama_calls_wait_for_the_slot_and_time_out_only_once_sent() -> None:
    settings = ollama()
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def slow_client(**kwargs: Any) -> Iterator[dict[str, Any]]:
        def gen() -> Iterator[dict[str, Any]]:
            order.append("first sent")
            started.set()
            release.wait(5)
            yield chunk("one")

        return gen()

    def fast_client(**kwargs: Any) -> Iterator[dict[str, Any]]:
        order.append("second sent")
        return iter([chunk("two")])

    first = threading.Thread(
        target=lambda: list(LiteLLMGenerator(settings, slow_client).stream(LONG, settings))
    )
    first.start()
    assert started.wait(5)
    second_slot = GenerationSlot(settings)
    assert not second_slot.try_acquire()  # busy: the stream path announces the wait
    second = threading.Thread(
        target=lambda: LiteLLMGenerator(settings, fast_client).complete(LONG, settings)
    )
    second.start()
    time.sleep(0.3)
    assert order == ["first sent"]  # still waiting here, not inside Ollama
    release.set()
    first.join(5)
    second.join(5)
    assert order == ["first sent", "second sent"]
    assert second_slot.try_acquire()
    second_slot.release()


def test_the_slot_is_reentrant_per_thread_and_released_from_any_thread() -> None:
    settings = ollama()
    outer = GenerationSlot(settings)
    assert outer.acquire()
    # A call the question makes while holding the slot doesn't wait for itself.
    with GenerationSlot(settings) as inner:
        assert not inner.acquired
    # Another thread must wait...
    other = GenerationSlot(settings)
    got: list[bool] = []
    waiter = threading.Thread(target=lambda: got.append(other.acquire()))
    waiter.start()
    time.sleep(0.2)
    assert got == []
    # ...until the holder is released, here from a third thread (a stream closed by
    # the thread serving the response).
    threading.Thread(target=outer.release).start()
    waiter.join(5)
    assert got == [True]
    other.release()


def test_the_slot_is_off_for_hosted_providers_and_with_no_limit() -> None:
    for settings in (
        make_test_settings(llm_provider="openai", openai_api_key="k"),
        ollama(ollama_max_parallel=0),
    ):
        holders = [GenerationSlot(settings) for _ in range(3)]
        assert all(slot.try_acquire() for slot in holders)
        assert not any(slot.acquired for slot in holders)


def test_a_waiting_caller_that_left_or_waited_too_long_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ollama()
    holder = GenerationSlot(settings)
    assert holder.acquire()
    result: list[object] = []

    def wait(slot: GenerationSlot, **kwargs: Any) -> None:
        try:
            result.append(slot.acquire(**kwargs))
        except GenerationError as exc:
            result.append(str(exc))

    thread = threading.Thread(
        target=wait, args=(GenerationSlot(settings),), kwargs={"should_stop": lambda: True}
    )
    thread.start()
    thread.join(5)
    assert result == [False]
    monkeypatch.setattr(llm, "_SLOT_WAIT_SECONDS", 0.5)
    thread = threading.Thread(target=wait, args=(GenerationSlot(settings),))
    thread.start()
    thread.join(5)
    assert "busy" in str(result[-1])
    holder.release()


def test_a_timeout_names_the_time_it_actually_allowed() -> None:
    import httpx
    from litellm.exceptions import Timeout as LiteLLMTimeout

    def timing_out(**kwargs: Any) -> object:
        raise LiteLLMTimeout(
            message="timed out", model="m", llm_provider="ollama"
        ) from httpx.ReadTimeout("read timed out")

    settings = ollama()
    with pytest.raises(GenerationError) as raised:
        LiteLLMGenerator(settings, completion_client=timing_out).complete(LONG, settings)
    message = str(raised.value)
    assert f"{request_timeout_seconds(LONG, settings):.0f}s" in message
    assert "LLM_PREFILL_TOKENS_PER_SECOND" in message
    # A short prompt's allowance rounds to the base: the plain message.
    with pytest.raises(GenerationError, match="LLM_REQUEST_TIMEOUT_SECONDS=60s"):
        LiteLLMGenerator(settings, completion_client=timing_out).complete(
            [{"role": "user", "content": ""}], ollama(llm_prefill_tokens_per_second=0)
        )


def test_a_non_streamed_ollama_call_measures_its_read_too() -> None:
    settings = ollama()
    client = StreamingClient([chunk("a"), chunk("b")], delay=0.2)
    LiteLLMGenerator(settings, completion_client=client).complete(LONG, settings)
    assert measured_prefill_rate(settings) is not None


def test_one_cold_load_does_not_drag_the_measured_rate_but_two_slow_reads_do() -> None:
    settings = ollama()
    llm._record_prefill(settings, 7000, 100.0)  # 70 tokens/s
    llm._record_prefill(settings, 7000, 1000.0)  # 7/s: the model was loading
    assert measured_prefill_rate(settings) == pytest.approx(70)
    llm._record_prefill(settings, 7000, 100.0)
    assert measured_prefill_rate(settings) == pytest.approx(70)
    # Slow twice in a row: the server really is slower now.
    llm._record_prefill(settings, 7000, 1000.0)
    llm._record_prefill(settings, 7000, 1000.0)
    assert measured_prefill_rate(settings) == pytest.approx(7)
