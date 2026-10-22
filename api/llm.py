"""The LLM client: LiteLLM calls per provider, reasoning-model budgets, and provider errors."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, Protocol, TypedDict, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import litellm
from litellm import exceptions as litellm_exceptions
from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError
from litellm.exceptions import Timeout as LiteLLMTimeout

from api.metrics import llm_tokens_total
from api.settings import AppSettings

_OLLAMA_MODEL_PREFIXES = ("ollama/", "ollama_chat/")


class GenerationError(RuntimeError):
    """Raised when grounded generation fails."""


class GenerationConfigError(GenerationError):
    """Raised when the selected generation provider is not configured."""


class ChatMessage(TypedDict):
    """LiteLLM-compatible chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionClient(Protocol):
    """Callable surface used for LiteLLM completion."""

    def __call__(self, **kwargs: Any) -> object: ...


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception it was raised from (cycle-safe)."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_request_timeout(exc: BaseException) -> bool:
    """True when the provider answered too slowly, False when it was never reached.

    LiteLLM reports both a slow answer and an unroutable host as a timeout — through
    Ollama as an ``APIConnectionError`` wrapping ``litellm.Timeout``, through OpenAI as a
    bare ``litellm.Timeout`` (which is *not* a ``litellm.APIConnectionError``). Only the
    underlying httpx exception tells them apart: ``ConnectTimeout`` means nothing
    answered, so "raise the timeout" would be the wrong advice.
    """

    chain = list(_exception_chain(exc))
    if any(isinstance(link, httpx.ConnectTimeout) for link in chain):
        return False
    if any(isinstance(link, (httpx.TimeoutException, LiteLLMTimeout)) for link in chain):
        return True
    message = str(exc).lower()
    return "timed out" in message or "litellm.timeout" in message


logger = logging.getLogger(__name__)


# Fixed, user-facing reasons per provider failure. The provider's own message goes to
# the log only: it reached every Q&A caller verbatim, and LiteLLM's messages can carry
# the request URL (with any credentials in it) and provider account details.
_PROVIDER_FAILURE_REASONS: tuple[tuple[type[Exception], str], ...] = (
    (
        litellm_exceptions.AuthenticationError,
        "the provider rejected its credentials (check OPENAI_API_KEY / ANTHROPIC_API_KEY)",
    ),
    (
        litellm_exceptions.PermissionDeniedError,
        "the provider denied access to this model",
    ),
    (litellm_exceptions.RateLimitError, "the provider is rate limiting requests; retry shortly"),
    (
        litellm_exceptions.NotFoundError,
        "the provider does not know this model (check LLM_MODEL / OPENAI_MODEL / ANTHROPIC_MODEL)",
    ),
    (
        litellm_exceptions.ContextWindowExceededError,
        "the prompt is longer than the model's context window; ask with fewer context chunks",
    ),
    (litellm_exceptions.ContentPolicyViolationError, "the provider refused the request"),
    (litellm_exceptions.BadRequestError, "the provider rejected the request"),
    (
        litellm_exceptions.ServiceUnavailableError,
        "the provider is unavailable; retry shortly",
    ),
)


def _never_connected(exc: BaseException) -> bool:
    """Whether the request never reached the server (refused, unresolvable, unroutable).

    LiteLLM's ``openai/`` route reports a refused connection as ``InternalServerError``,
    not ``APIConnectionError``; only the cause chain says what happened.
    """

    return any(isinstance(link, httpx.ConnectError) for link in _exception_chain(exc))


def _provider_failure(exc: Exception) -> GenerationError:
    """A ``GenerationError`` naming the kind of failure, not the provider's raw text."""

    logger.warning("Generation provider request failed.", exc_info=exc)
    reason = next(
        (text for kind, text in _PROVIDER_FAILURE_REASONS if isinstance(exc, kind)),
        f"{type(exc).__name__}; see the server log for details",
    )
    return GenerationError(f"Generation provider request failed: {reason}.")


def redact_url(url: str) -> str:
    """``url`` without any ``user:password@`` part, for messages a client can see."""

    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit(parts._replace(netloc=netloc))


def _connection_error(
    exc: Exception, settings: AppSettings, timeout: float | None = None
) -> GenerationError:
    """Translate LiteLLM's connection/timeout errors, telling "too slow" apart from "unreachable".

    A slow local answer used to come back as "Cannot reach Ollama… start `ollama
    serve`", sending the user after a server that was up the whole time.
    """

    if _is_request_timeout(exc):
        base = float(settings.llm_request_timeout_seconds)
        if timeout is not None and timeout > base:
            # The base plus the time allowed to read this prompt
            # (LLM_PREFILL_TOKENS_PER_SECOND): name both knobs.
            return GenerationError(
                f"The generation provider sent nothing for {timeout:.0f}s "
                f"(LLM_REQUEST_TIMEOUT_SECONDS={base:g}s plus time to read the prompt). "
                "Local models on modest hardware can need longer — raise "
                "LLM_REQUEST_TIMEOUT_SECONDS or lower LLM_PREFILL_TOKENS_PER_SECOND."
            )
        return GenerationError(
            "The generation provider did not finish within "
            f"LLM_REQUEST_TIMEOUT_SECONDS={base:g}s. "
            "Local models on modest hardware can need longer — raise that setting."
        )
    if settings.llm_provider.lower().strip() == "ollama":
        return GenerationError(
            f"Cannot reach Ollama at {redact_url(settings.ollama_base_url)}. Start Ollama "
            "(`ollama serve`) and confirm OLLAMA_BASE_URL is reachable from "
            "wherever the API process runs — use http://localhost:11434 when "
            "the API runs directly on your host, or "
            "http://host.docker.internal:11434 only when the API itself runs "
            "inside Docker."
        )
    if settings.llm_provider.lower().strip() == "openai_compatible":
        logger.warning("OpenAI-compatible server unreachable.", exc_info=exc)
        return GenerationError(
            "Cannot reach the OpenAI-compatible server at "
            f"{redact_url(settings.openai_compatible_base_url)}. Check that it is running "
            "and that OPENAI_COMPATIBLE_BASE_URL (ending in /v1) is reachable from "
            "wherever the API process runs."
        )
    return _provider_failure(exc)


# How the most recent LiteLLM call on this thread ended, read by the pipeline right
# after it generates (the ChatGenerator protocol returns text only). Per thread: each
# question generates on its own thread, and a stream is consumed on the thread that
# started it.
_last_call = threading.local()


def reset_finish_reason(reason: str | None = None) -> None:
    _last_call.finish_reason = reason


def last_finish_reason() -> str | None:
    """The provider's ``finish_reason`` for this thread's latest call, if it gave one."""

    reason = getattr(_last_call, "finish_reason", None)
    return reason if isinstance(reason, str) else None


def begin_usage() -> None:
    """Start adding up this thread's reported token usage (see ``usage_totals``)."""

    _last_call.usage = [0, 0]


def add_usage(usage: tuple[int, int] | None) -> None:
    """Add usage counted on another thread (a summary's parallel parts) to this thread's
    totals since ``begin_usage``."""

    totals = getattr(_last_call, "usage", None)
    if usage is None or not isinstance(totals, list):
        return
    totals[0] += usage[0]
    totals[1] += usage[1]


def usage_totals() -> tuple[int, int] | None:
    """Prompt and completion tokens reported since ``begin_usage`` on this thread, or
    None when the provider reported none (or nothing was begun)."""

    totals = getattr(_last_call, "usage", None)
    if not isinstance(totals, list) or not any(totals):
        return None
    return int(totals[0]), int(totals[1])


def stopped_at_token_limit() -> bool:
    """True when this thread's latest call was cut off at ``max_tokens``."""

    return last_finish_reason() == "length"


def _record_finish_reason(response: object) -> None:
    choices = read_value(response, "choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
        reason = read_value(choices[0], "finish_reason")
        if isinstance(reason, str) and reason:
            _last_call.finish_reason = reason


def _count_usage(response: object, settings: AppSettings) -> None:
    """Add a response's reported token usage to ``docrag_llm_tokens_total``.

    Only what the provider reports: a stream carries usage on its last chunk when the
    provider sends it (LiteLLM fills it in for Ollama); ``stream_options`` isn't sent to
    ask for it, since some OpenAI-compatible servers reject the field.
    """

    usage = read_value(response, "usage")
    if usage is None:
        return
    provider = settings.llm_provider.lower().strip()
    totals = getattr(_last_call, "usage", None)
    for index, (kind, key) in enumerate(
        (("prompt", "prompt_tokens"), ("completion", "completion_tokens"))
    ):
        value = read_value(usage, key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            llm_tokens_total.labels(provider=provider, kind=kind).inc(value)
            if isinstance(totals, list):
                totals[index] += value


# Providers that read the prompt on the same machine and send nothing while they do.
_LOCAL_PROVIDERS = ("ollama", "openai_compatible")
# A prompt shorter than this is mostly request overhead: its speed says little about
# how fast the server reads.
_RATE_MIN_PROMPT_TOKENS = 500
_RATE_SMOOTHING = 0.3
_measured_rates: dict[tuple[str, str], float] = {}
_rates_lock = threading.Lock()


def _provider(settings: AppSettings) -> str:
    return settings.llm_provider.lower().strip()


def _rate_key(settings: AppSettings) -> tuple[str, str]:
    provider = _provider(settings)
    model = (
        settings.openai_compatible_model if provider == "openai_compatible" else settings.llm_model
    )
    return provider, model


def _estimated_prompt_tokens(messages: Sequence[ChatMessage]) -> int:
    # api.prompting imports this module; it owns the (deliberately high) estimate.
    from api.prompting import estimate_prompt_tokens

    return estimate_prompt_tokens(messages)


def measured_prefill_rate(settings: AppSettings) -> float | None:
    """Estimated prompt tokens per second this process has seen the local model read
    (time to the first streamed chunk), smoothed; None before a long enough prompt."""

    with _rates_lock:
        return _measured_rates.get(_rate_key(settings))


def _record_prefill(settings: AppSettings, prompt_tokens: int, seconds: float) -> None:
    if (
        _provider(settings) not in _LOCAL_PROVIDERS
        or prompt_tokens < _RATE_MIN_PROMPT_TOKENS
        or seconds <= 0
    ):
        return
    rate = prompt_tokens / seconds
    key = _rate_key(settings)
    with _rates_lock:
        previous = _measured_rates.get(key)
        _measured_rates[key] = (
            rate if previous is None else previous + _RATE_SMOOTHING * (rate - previous)
        )


def request_timeout_seconds(messages: Sequence[ChatMessage], settings: AppSettings) -> float:
    """``LLM_REQUEST_TIMEOUT_SECONDS``, plus time to read the prompt on a local server.

    LiteLLM gives httpx a single number for Ollama (an ``httpx.Timeout`` is turned back
    into its read part), applied as the longest silence allowed — and a local server is
    silent for the whole prompt read, ~70 tokens/s on an M1. The prompt's estimated
    tokens over ``LLM_PREFILL_TOKENS_PER_SECOND`` (or the slower rate measured here)
    are added, so a long prompt gets the time it needs while a server that stops
    mid-answer still fails after the base timeout.
    """

    base = float(settings.llm_request_timeout_seconds)
    configured = float(settings.llm_prefill_tokens_per_second)
    if configured <= 0 or _provider(settings) not in _LOCAL_PROVIDERS:
        return base
    measured = measured_prefill_rate(settings)
    rate = min(configured, measured) if measured else configured
    return base + _estimated_prompt_tokens(messages) / rate


def expected_prefill_ms(messages: Sequence[ChatMessage], settings: AppSettings) -> float | None:
    """How long the local model should take to read ``messages`` at its measured rate;
    None when no rate has been measured (or the provider isn't local)."""

    if _provider(settings) not in _LOCAL_PROVIDERS:
        return None
    measured = measured_prefill_rate(settings)
    if not measured:
        return None
    return _estimated_prompt_tokens(messages) / measured * 1000


# A call waiting for Ollama longer than this gives up: something is stuck.
_SLOT_WAIT_SECONDS = 900.0
_slots: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_slots_lock = threading.Lock()
# The slot keys the current thread holds: a call made while its question already holds
# the slot (condense, answer, citation retry, follow-ups) doesn't wait for itself.
_held = threading.local()


def _held_keys() -> set[tuple[str, int]]:
    keys = getattr(_held, "keys", None)
    if keys is None:
        keys = set()
        _held.keys = keys
    return cast(set[tuple[str, int]], keys)


class GenerationSlot:
    """One of ``OLLAMA_MAX_PARALLEL`` places for calls to one Ollama server.

    Ollama answers one request at a time by default and queues the rest — with each
    queued request's timeout running, so a second question timed out without ever
    being read. Calls wait here instead, and their timeout starts once they are sent.
    A no-op for other providers, with ``OLLAMA_MAX_PARALLEL=0``, and on a thread that
    already holds the slot (re-entrant per thread). ``release`` works from any thread:
    a stream may be closed by another thread than the one that opened it.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._key: tuple[str, int] | None = None
        self._semaphore: threading.BoundedSemaphore | None = None
        self._owner: set[tuple[str, int]] | None = None
        self.acquired = False
        if _provider(settings) == "ollama" and settings.ollama_max_parallel > 0:
            self._key = (settings.ollama_base_url.rstrip("/"), int(settings.ollama_max_parallel))
            with _slots_lock:
                self._semaphore = _slots.setdefault(
                    self._key, threading.BoundedSemaphore(self._key[1])
                )

    @property
    def needed(self) -> bool:
        """Whether this thread has to take a place (False: no limit, or already held)."""

        return self._semaphore is not None and self._key not in _held_keys()

    def try_acquire(self) -> bool:
        """Take a place if one is free now; True also when none is needed."""

        if not self.needed:
            return True
        assert self._semaphore is not None
        if self._semaphore.acquire(blocking=False):
            self._mark_held()
            return True
        return False

    def acquire(self, should_stop: Callable[[], bool] | None = None) -> bool:
        """Wait for a place; False when ``should_stop`` reported the caller gone first."""

        if not self.needed:
            return True
        assert self._semaphore is not None
        deadline = time.monotonic() + _SLOT_WAIT_SECONDS
        while not self._semaphore.acquire(timeout=1.0):
            if should_stop is not None and should_stop():
                return False
            if time.monotonic() >= deadline:
                raise GenerationError(
                    "The local model stayed busy with other requests for "
                    f"{_SLOT_WAIT_SECONDS / 60:g} minutes; try again shortly."
                )
        self._mark_held()
        return True

    def _mark_held(self) -> None:
        keys = _held_keys()
        assert self._key is not None
        keys.add(self._key)
        self._owner = keys
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        self.acquired = False
        assert self._semaphore is not None and self._owner is not None and self._key
        self._owner.discard(self._key)
        self._semaphore.release()

    def __enter__(self) -> GenerationSlot:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _stream_kwargs(settings: AppSettings) -> dict[str, object]:
    """Ask for token usage on a stream's last chunk where the provider supports it.

    Without it LiteLLM's Ollama stream reported no usage at all (measured), so the
    prompt size a local model read was never known. Not sent to OpenAI-compatible
    servers: some reject the field.
    """

    if _provider(settings) in ("ollama", "openai"):
        return {"stream_options": {"include_usage": True}}
    return {}


class LiteLLMGenerator:
    """Generate answers with LiteLLM while keeping provider selection configurable."""

    def __init__(
        self,
        settings: AppSettings,
        completion_client: CompletionClient = litellm.completion,
    ) -> None:
        self.settings = settings
        self.completion_client = completion_client

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        """Call the LiteLLM provider selected by ``settings`` and return assistant text.

        Provider selection reads the *passed* settings (a per-request override may
        differ from the ``settings`` captured at construction), so a single cached
        generator instance can serve both the Ollama and OpenAI paths.
        """

        model, provider_kwargs = completion_model_and_kwargs(settings)
        reset_finish_reason()
        # Ollama sends nothing until a non-streamed answer is complete, so its timeout
        # (the longest silence allowed) had to cover reading the prompt *and* writing
        # the whole answer. Streamed and joined, it covers the read, then each gap.
        streamed = _provider(settings) == "ollama"
        timeout = request_timeout_seconds(messages, settings)
        with GenerationSlot(settings):
            try:
                response = self.completion_client(
                    model=model,
                    messages=list(messages),
                    temperature=float(settings.llm_temperature),
                    max_tokens=effective_max_tokens(settings, model),
                    timeout=timeout,
                    num_retries=int(settings.llm_num_retries),
                    # Some models (e.g. the gpt-5 family) reject params other models
                    # accept fine (temperature != 1). Let LiteLLM drop an unsupported
                    # param for the selected model rather than raising, so provider
                    # quirks don't turn into a hard failure.
                    drop_params=True,
                    **({"stream": True, **_stream_kwargs(settings)} if streamed else {}),
                    **provider_kwargs,
                )
                if streamed and not _is_whole_response(response):
                    return "".join(self._read_stream(response, settings))
            except (LiteLLMAPIConnectionError, LiteLLMTimeout) as exc:
                raise _connection_error(exc, settings, timeout) from exc
            except GenerationError:
                raise
            except Exception as exc:
                # LiteLLM/provider SDKs raise many distinct exception types (auth,
                # connection, rate limit, ...); this boundary's job is translating all
                # of them into our domain error so api/error_handlers.py maps them to a
                # clean 502 instead of an opaque 500.
                if _never_connected(exc):
                    raise _connection_error(exc, settings, timeout) from exc
                raise _provider_failure(exc) from exc
        _record_finish_reason(response)
        _count_usage(response, settings)
        return extract_completion_text(response)

    def _read_stream(
        self,
        response: object,
        settings: AppSettings,
        on_first_chunk: Callable[[], None] | None = None,
    ) -> Iterator[str]:
        """The text deltas of a streamed response; records its finish reason and the
        usage its last report carries, and releases the response however it ends."""

        # completion_client's return type is `object` (it must also cover the
        # non-streaming response) — with stream=True it is actually an iterable of
        # chunks; cast narrows that for the loop below.
        last_with_usage: object | None = None
        first = True
        try:
            for chunk in cast(Iterable[object], response):
                if first:
                    first = False
                    if on_first_chunk is not None:
                        on_first_chunk()
                _record_finish_reason(chunk)
                if read_value(chunk, "usage") is not None:
                    last_with_usage = chunk
                delta = extract_delta_text(chunk)
                if delta:
                    yield delta
        finally:
            _close_stream(response)
        if last_with_usage is not None:
            # The last report only: some providers repeat running totals per chunk.
            _count_usage(last_with_usage, settings)

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        """Call the LiteLLM provider with ``stream=True`` and yield assistant text deltas.

        Same provider selection and error-translation boundary as ``complete`` — the
        request is identical except for ``stream=True``. Because this is a generator
        function, the try/except below covers the whole streamed response: LiteLLM's
        ``CustomStreamWrapper`` is lazy, so a connection failure raised while iterating
        chunks is caught here exactly like a failure raised by the initial call.
        """

        model, provider_kwargs = completion_model_and_kwargs(settings)
        reset_finish_reason()
        timeout = request_timeout_seconds(messages, settings)
        slot = GenerationSlot(settings)
        slot.acquire()
        try:
            started = time.monotonic()

            def read_rate() -> None:
                _record_prefill(
                    settings, _estimated_prompt_tokens(messages), time.monotonic() - started
                )

            response = self.completion_client(
                model=model,
                messages=list(messages),
                temperature=float(settings.llm_temperature),
                max_tokens=effective_max_tokens(settings, model),
                timeout=timeout,
                num_retries=int(settings.llm_num_retries),
                drop_params=True,
                stream=True,
                **_stream_kwargs(settings),
                **provider_kwargs,
            )
            yield from self._read_stream(response, settings, on_first_chunk=read_rate)
        except (LiteLLMAPIConnectionError, LiteLLMTimeout) as exc:
            raise _connection_error(exc, settings, timeout) from exc
        except Exception as exc:
            if _never_connected(exc):
                raise _connection_error(exc, settings, timeout) from exc
            raise _provider_failure(exc) from exc
        finally:
            slot.release()


def _is_whole_response(response: object) -> bool:
    """A complete (non-streamed) response: a client that ignored ``stream=True``."""

    choices = read_value(response, "choices")
    return (
        isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and bool(choices)
        and read_value(choices[0], "message") is not None
    )


def _close_stream(response: object) -> None:
    """Release the provider's HTTP response when a stream ends early (client left).

    LiteLLM's sync stream wrapper has ``aclose`` but no ``close``, and its line iterator
    is only released when garbage-collected — meanwhile the connection stays open and
    Ollama keeps generating an answer nobody reads, holding the GPU. Closing the
    iterators down the chain reaches httpx's response, which closes the socket.
    """

    stream = getattr(response, "completion_stream", None)
    for candidate in (
        getattr(stream, "streaming_response", None),
        getattr(stream, "response_iterator", None),
        stream,
        response,
    ):
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - cleanup must never mask the real outcome
                logger.debug("Closing the provider stream failed.", exc_info=True)


class ChatGenerator(Protocol):
    """Generator surface used by the grounded answer pipeline (both response modes)."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str: ...

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterable[str]: ...


@functools.lru_cache(maxsize=64)
def supports_reasoning(model: str) -> bool:
    """True when LiteLLM reports ``model`` as a reasoning model (gpt-5 family, o-series).

    Wrapped so an unknown model or an offline metadata lookup degrades to False rather
    than raising — non-reasoning is the safe default (plain max_tokens, no effort knob).

    Ollama models are never asked: LiteLLM resolves an unmapped ``ollama/<tag>`` with an
    uncached POST to ``/api/show`` on ``OLLAMA_API_BASE``/localhost — not our configured
    ``ollama_base_url`` — and against an unreachable host that blocked 75s per call, up
    to four calls per question. Local models keep the plain budget by design anyway.
    """

    # openai/<name> is a model served by an OpenAI-compatible server, not OpenAI: its
    # name says nothing LiteLLM's metadata could know about. anthropic/<name>: Claude
    # thinks only when asked to (extended thinking is never enabled here), so the
    # reasoning headroom would only inflate the budget.
    if model.startswith((*_OLLAMA_MODEL_PREFIXES, "openai/", "anthropic/")):
        return False
    try:
        return bool(litellm.supports_reasoning(model))
    except Exception:
        return False


def effective_max_tokens(settings: AppSettings, model: str) -> int:
    """Output-token budget for ``model``.

    Reasoning models spend hidden reasoning tokens out of the same budget as the visible
    answer, so they get ``llm_max_tokens`` plus ``reasoning_token_headroom`` — otherwise
    reasoning alone exhausts a small budget and the visible answer comes back empty. The
    additive form preserves the relative sizing of the cheaper condense/expansion calls,
    which lower ``llm_max_tokens`` via ``model_copy``.
    """

    base = int(settings.llm_max_tokens)
    if supports_reasoning(model):
        return base + int(settings.reasoning_token_headroom)
    return base


def completion_model_and_kwargs(settings: AppSettings) -> tuple[str, dict[str, str | int]]:
    """Return LiteLLM model name and provider-specific keyword arguments."""

    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        # ollama_chat (/api/chat), not ollama (/api/generate): the generate route files
        # every extra kwarg under "options", where Ollama rejects keep_alive as an
        # invalid option — so the model unloaded after Ollama's 5-minute default no
        # matter what was configured — and it flattens messages without the model's
        # own chat template.
        ollama_kwargs: dict[str, str | int] = {
            "base_url": settings.ollama_base_url,
            "keep_alive": settings.ollama_keep_alive,
        }
        if settings.ollama_num_ctx:
            ollama_kwargs["num_ctx"] = int(settings.ollama_num_ctx)
        return f"ollama_chat/{settings.llm_model}", ollama_kwargs
    if provider == "openai":
        if not settings.openai_api_key:
            raise GenerationConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        kwargs: dict[str, str | int] = {"api_key": settings.openai_api_key}
        effort = settings.openai_reasoning_effort.strip()
        if effort and supports_reasoning(settings.openai_model):
            kwargs["reasoning_effort"] = effort
        return settings.openai_model, kwargs
    if provider == "openai_compatible":
        if not settings.openai_compatible_base_url or not settings.openai_compatible_model:
            raise GenerationConfigError(
                "OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_MODEL are required when "
                "LLM_PROVIDER=openai_compatible."
            )
        # LiteLLM's openai/ route with a custom api_base: the OpenAI wire protocol
        # against any server. The client insists on some key; local servers ignore it.
        return f"openai/{settings.openai_compatible_model}", {
            "api_base": settings.openai_compatible_base_url,
            "api_key": settings.openai_compatible_api_key or "not-needed",
        }
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise GenerationConfigError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        return f"anthropic/{settings.anthropic_model}", {"api_key": settings.anthropic_api_key}
    raise GenerationConfigError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'.")


def extract_completion_text(response: object) -> str:
    """Extract assistant text from LiteLLM's response object or a compatible fake."""

    choices = read_value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise GenerationError("Generation provider response did not include choices.")

    first_choice = choices[0]
    message = read_value(first_choice, "message")
    content = read_value(message, "content")
    if not isinstance(content, str):
        raise GenerationError("Generation provider response did not include message content.")
    return content


def extract_delta_text(chunk: object) -> str:
    """Extract the incremental assistant text from one streamed LiteLLM chunk.

    Returns "" for chunks that carry no content delta (e.g. the final chunk, or a
    role-only opening chunk) rather than raising — those are a normal part of a
    stream, unlike a malformed non-streaming response.
    """

    choices = read_value(chunk, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return ""
    delta = read_value(choices[0], "delta")
    content = read_value(delta, "content")
    return content if isinstance(content, str) else ""


def read_value(source: object, key: str) -> object:
    """Read an attribute or dictionary key from a response object."""

    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)
