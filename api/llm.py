"""The LLM client: LiteLLM calls per provider, reasoning-model budgets, and provider errors."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, Protocol, TypedDict, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
import litellm
from litellm import exceptions as litellm_exceptions
from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError
from litellm.exceptions import Timeout as LiteLLMTimeout

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
        "the provider rejected its credentials (check OPENAI_API_KEY)",
    ),
    (
        litellm_exceptions.PermissionDeniedError,
        "the provider denied access to this model",
    ),
    (litellm_exceptions.RateLimitError, "the provider is rate limiting requests; retry shortly"),
    (
        litellm_exceptions.NotFoundError,
        "the provider does not know this model (check LLM_MODEL / OPENAI_MODEL)",
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


def _connection_error(exc: Exception, settings: AppSettings) -> GenerationError:
    """Translate LiteLLM's connection/timeout errors, telling "too slow" apart from "unreachable".

    A slow local answer used to come back as "Cannot reach Ollama… start `ollama
    serve`", sending the user after a server that was up the whole time.
    """

    if _is_request_timeout(exc):
        return GenerationError(
            "The generation provider did not finish within "
            f"LLM_REQUEST_TIMEOUT_SECONDS={settings.llm_request_timeout_seconds:g}s. "
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
        try:
            response = self.completion_client(
                model=model,
                messages=list(messages),
                temperature=float(settings.llm_temperature),
                max_tokens=effective_max_tokens(settings, model),
                timeout=float(settings.llm_request_timeout_seconds),
                num_retries=int(settings.llm_num_retries),
                # Some models (e.g. the gpt-5 family) reject params other models
                # accept fine (temperature != 1). Let LiteLLM drop an unsupported
                # param for the selected model rather than raising, so provider
                # quirks don't turn into a hard failure.
                drop_params=True,
                **provider_kwargs,
            )
        except (LiteLLMAPIConnectionError, LiteLLMTimeout) as exc:
            raise _connection_error(exc, settings) from exc
        except Exception as exc:
            # LiteLLM/provider SDKs raise many distinct exception types (auth,
            # connection, rate limit, ...); this boundary's job is translating all
            # of them into our domain error so main.py maps them to a clean 502
            # instead of an opaque 500.
            if _never_connected(exc):
                raise _connection_error(exc, settings) from exc
            raise _provider_failure(exc) from exc
        return extract_completion_text(response)

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        """Call the LiteLLM provider with ``stream=True`` and yield assistant text deltas.

        Same provider selection and error-translation boundary as ``complete`` — the
        request is identical except for ``stream=True``. Because this is a generator
        function, the try/except below covers the whole streamed response: LiteLLM's
        ``CustomStreamWrapper`` is lazy, so a connection failure raised while iterating
        chunks is caught here exactly like a failure raised by the initial call.
        """

        model, provider_kwargs = completion_model_and_kwargs(settings)
        try:
            response = self.completion_client(
                model=model,
                messages=list(messages),
                temperature=float(settings.llm_temperature),
                max_tokens=effective_max_tokens(settings, model),
                timeout=float(settings.llm_request_timeout_seconds),
                num_retries=int(settings.llm_num_retries),
                drop_params=True,
                stream=True,
                **provider_kwargs,
            )
            # completion_client's return type is `object` (it must also cover the
            # non-streaming response `complete` uses) — with stream=True it is
            # actually an iterable of chunks; cast narrows that for the loop below.
            for chunk in cast(Iterable[object], response):
                delta = extract_delta_text(chunk)
                if delta:
                    yield delta
        except (LiteLLMAPIConnectionError, LiteLLMTimeout) as exc:
            raise _connection_error(exc, settings) from exc
        except Exception as exc:
            if _never_connected(exc):
                raise _connection_error(exc, settings) from exc
            raise _provider_failure(exc) from exc


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
    # name says nothing LiteLLM's metadata could know about.
    if model.startswith((*_OLLAMA_MODEL_PREFIXES, "openai/")):
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
