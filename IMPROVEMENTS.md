# DocRAG Improvement Recommendations

Audit performed against the current codebase (backend RAG modules, frontend, tests,
Docker/config). Every finding below was verified against the actual source. One finding
(P1-1, raw 500 on bad OpenAI key) was proven live against a running local stack
(Qdrant + Ollama qwen2.5) during this session.

**Constraints for whoever implements these** (see `CLAUDE.md`):
- Hybrid RAG behavior is non-negotiable (RRF `k=60`, rerank fused top-N, citations with
  filename/page/section/chunk_id).
- The reranker default `BAAI/bge-reranker-v2-m3` is a spec'd default — do NOT change it
  unilaterally even though it's currently incompatible with installed FastEmbed 0.8.0
  (see P1-8 — this is flagged as an owner decision, not prescribed).
- Git is read-only by default; follow the repo's validation policy (backend-verify /
  frontend-verify workflows) for every change.
- Implement one item (or one tightly-related group) at a time, with a red-without/
  green-with test proof, then run `uv run pytest`, `ruff`, `mypy` (backend) and
  `npm run test`, `npm run build`, `oxlint` (frontend) before moving to the next.

---

## P1 — Correctness & robustness (user-visible failures)

### 1. `api/generation.py:77-89` — LiteLLM exceptions leak as raw 500 (proven live)

```python
def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
    model, provider_kwargs = completion_model_and_kwargs(settings)
    response = self.completion_client(
        model=model,
        messages=list(messages),
        temperature=float(settings.llm_temperature),
        max_tokens=int(settings.llm_max_tokens),
        timeout=float(settings.llm_request_timeout_seconds),
        **provider_kwargs,
    )
    return extract_completion_text(response)
```

No try/except around the completion call. Proven this session: selecting OpenAI with an
invalid key produced `litellm.exceptions.AuthenticationError` → an opaque HTTP 500 after
~40 seconds of LiteLLM's internal retries. The same class of raw 500 happens if Ollama is
unreachable.

**Fix:** wrap the completion call; map `litellm.AuthenticationError`,
`litellm.APIConnectionError`, and any other litellm exception into the existing
`GenerationError` (already mapped to 502 in `main.py`), preserving the provider's message
text. Pass `num_retries=0` (or add an `LLM_NUM_RETRIES` setting defaulting to 0) so auth
failures fail fast instead of burning ~40s.

**Test:** a fake `completion_client` that raises `litellm.AuthenticationError` → assert
502 with the provider message in `detail`.

### 2. `api/retrieval.py:81-111` — one malformed point 500s the entire query

```python
def payload_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RetrievalPayloadError(f"Retrieved point is missing payload field '{key}'.")
    return value
```

`scored_point_to_chunk` is called for every fused candidate (up to `fused_top_n=50`).
`RetrievalPayloadError` maps to a 500 in `main.py`. One legacy or whitespace-only point
among 50 good candidates aborts the whole answer instead of just being skipped.

**Fix:** in `retrieve_candidates`, convert points defensively — skip (log a warning for)
candidates that fail payload validation, and only raise `RetrievalPayloadError` if *zero*
candidates survive.

**Test:** repository returning one good + one malformed point → answer still succeeds
using the good one; all-malformed → still raises.

### 3. Lazy-loaded models bypass their own error-wrapping guards

`api/dependencies.py:41-48`:
```python
@lru_cache
def get_reranker() -> LocalCrossEncoderReranker:
    try:
        return LocalCrossEncoderReranker(get_app_settings())
    except ValueError as exc:
        raise RerankingError(str(exc)) from exc
```

`LocalCrossEncoderReranker.__init__` builds `TextCrossEncoder(..., lazy_load=True)`, so
construction almost never raises — the real model download/load happens inside the
threadpool on the first `.rerank()` call, **outside** this guard. A bad
`RERANKER_MODEL` or a failed download surfaces as a raw exception → 500, never the
intended `RerankingError` → 502. `get_embedding_provider` (same file, lines 34-38) has
the identical deferred-failure problem with no guard at all.

**Fix:** move the try/except into `LocalCrossEncoderReranker.score` (wrap → 502) and add
an equivalent wrap in `LocalEmbeddingProvider.embed_texts` (→ `EmbeddingError`).
Optionally add a FastAPI startup/lifespan warmup that touches both models once, so
misconfiguration fails loudly at boot instead of on the first user request.

**Test:** a model stub whose `.rerank()`/`.embed()` raises `ValueError` → assert the
call surfaces as `RerankingError`/`EmbeddingError`, not a bare exception.

### 4. `api/repository.py:40-44` — over-broad exception fallback masks real errors

```python
try:
    return self._server_side_hybrid_query(settings, query_embedding)
except ApiException:
    # Older Qdrant servers may not support server-side prefetch + RRF.
    return self._manual_hybrid_query(settings, query_embedding)
```

`ApiException` is the base class for nearly all Qdrant HTTP errors (bad collection,
malformed query, auth, 4xx/5xx) — not just "server doesn't support RRF." Any transient
server-side failure silently triggers two extra round-trips of manual fusion, which then
fails with a more confusing, unrelated error; the real cause is swallowed. Meanwhile,
non-`ApiException` infra failures (Qdrant down → connection refused,
`ResponseHandlingException`, timeouts) propagate completely unwrapped as raw 500s.

**Fix:** narrow the fallback to the specific signal that indicates "RRF/prefetch
unsupported" (inspect status code, e.g. 400/404/501) and re-raise everything else. Wrap
connection-level failures in a `RetrievalError` subclass mapped to a new 503 handler in
`main.py` with a clear "Qdrant unreachable" message.

**Test:** client raising a generic connection error → 503 with a clean message; client
raising `ApiException` for a genuinely bad query (400) → does NOT get retried via manual
fusion, surfaces as an error instead of a confusing second failure.

### 5. `api/qdrant_schema.py:101-135` — readiness-check race + payload indexes never validated

```python
cached = _readiness_cache.get(client, {}).get(cache_key)
if cached is not None:
    return cached
ready = _ensure_collection_uncached(client, settings)   # unsynchronized
```
and inside `_ensure_collection_uncached`:
```python
if not client.collection_exists(collection_name):
    client.create_collection(...)
    create_payload_indexes(client, collection_name)   # separate, non-atomic call
```

FastAPI runs sync dependencies in a threadpool, so two concurrent first-ever requests can
both see "not exists" and both call `create_collection`; the loser gets an unhandled
exception → 500. Separately, `create_collection` and `create_payload_indexes` are two
non-atomic calls, and `validate_collection_schema` (line 151) never checks that the
payload indexes actually exist — if index creation fails or the process dies between the
two calls, the collection passes validation forever without indexes, and filtered
deletes/queries silently run unindexed.

**Fix:** guard `_ensure_collection_uncached` with a module-level `threading.Lock`
(double-checked locking — re-check the cache once inside the lock). Treat an
"already exists" error from `create_collection` as success rather than a failure.
Extend `validate_collection_schema` to confirm the collection's payload schema contains
the four `PAYLOAD_INDEXES` fields, creating any that are missing (idempotent repair)
instead of only validating vectors/tags.

**Test:** two threads calling `ensure_collection` concurrently against a cold cache →
no exception raised, exactly one `create_collection` call.

### 6. `api/main.py:79-88` — unbounded upload buffered entirely into memory

```python
async def upload_document(...) -> DocumentIngestResponse:
    content = await file.read()
```

No size cap anywhere in the backend or frontend. A single multi-GB file (or a handful of
concurrent uploads) exhausts the container's memory. The frontend's
`accept=".pdf,.txt,.md,.markdown"` on the file input is advisory only and gives no early
feedback.

**Fix:** add `max_upload_bytes: PositiveInt` to `AppSettings` (e.g. default 50 MB). In
the upload handler, check `file.size` before/while reading and reject oversize requests
with 413 (add a payload-too-large exception handler alongside the existing ones in
`main.py`). Mirror the check client-side in `UploadPanel`/`App.handleUpload`: reject
`file.size > limit` with an inline error before any network call.

**Test:** backend — a file larger than the configured limit → 413, not ingested.
Frontend — selecting an oversize file → inline error shown, `uploadDocument` never
called.

### 7. `docker-compose.yml` — loads `.env.example` not `.env`; `host.docker.internal` breaks on Linux

```yaml
  api:
    env_file:
      - .env.example
```

A real operator `.env` (with a real `OPENAI_API_KEY`, etc.) is never read by Compose —
only the placeholder example file is. Separately, `.env.example` sets
`OLLAMA_BASE_URL=http://host.docker.internal:11434`, which resolves automatically only
on Docker Desktop (macOS/Windows); on native Linux Docker it fails to resolve unless the
container declares `extra_hosts`, so the default provider (Ollama) is broken out of the
box on Linux.

**Fix:** `env_file: [.env.example, .env]` with `.env` optional (Compose supports
`required: false` on an env_file entry) so it overrides the example when present. Add to
the `api` service:
```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
```
Document both in the README's Docker section.

### 8. Reranker default is incompatible with installed FastEmbed — **owner decision needed**

`api/settings.py` defaults `reranker_model` to `BAAI/bge-reranker-v2-m3`. FastEmbed
0.8.0 (currently installed) does not support this model — confirmed this session; a
working local stack required overriding to `RERANKER_MODEL=BAAI/bge-reranker-base`.
Out-of-the-box `docker compose up` followed by the first question will fail (500, or a
clean 502 once P1-3 lands).

This is a spec'd default per `CLAUDE.md` — not something to change unilaterally. Options
for the project owner: (a) bump the `fastembed` dependency when a supporting release
ships, (b) change the spec'd default to `BAAI/bge-reranker-base`, (c) keep the spec'd
default and clearly document the required override in the README/`.env.example`. This
item is a decision to make, not a fix to apply.

---

## P2 — Quality & efficiency

### 9. `api/documents.py:106,113` — chunks sized in whitespace words; embedder truncates at 512 subword tokens

```python
tokens = section.text.split()
...
chunk_text = " ".join(tokens[start:end]).strip()
```

`chunk_size_tokens=500` counts whitespace-delimited words, but the dense model
(`BAAI/bge-small-en-v1.5`) hard-truncates at 512 *subword* tokens. 500 English words
typically expand to ~650-900 subword tokens, so the tail of most full-size chunks is
silently dropped from the embedding — while the full untruncated text is still stored
and shown in citations, misrepresenting what was actually indexed. Additionally,
`" ".join(...)` collapses all newlines, losing paragraph structure in stored/cited text.

**Fix (cheap, immediate):** lower the default `chunk_size_tokens` to ~300 to stay safely
under the 512 subword limit after expansion, and rename/document the setting honestly
(it counts words, not model tokens). Note this invalidates existing indexes — document
that changing it requires re-ingest. **Fix (better, later):** size chunks using the
actual embedding model's tokenizer instead of `str.split()`. Consider preserving
original whitespace/line breaks for stored/displayed chunk text.

### 10. `api/documents.py:161-167` — UTF-8-only decode rejects common real-world files

```python
text = content.decode("utf-8-sig")
```

`utf-8-sig` only adds BOM tolerance, not encoding fallback. A `.txt`/`.md` file saved as
cp1252 or latin-1 (common from Windows tools) is rejected outright with
`DocumentParseError`.

**Fix:** try `utf-8-sig` first; on `UnicodeDecodeError`, fall back to `cp1252`, then
`latin-1` with `errors="replace"` as a last resort (or use a charset-detection library).

**Test:** a cp1252-encoded byte string (e.g. `"café".encode("cp1252")`) → parses
successfully instead of raising.

### 11. `frontend/src/api/client.ts:21-36` — no timeout or cancellation on any fetch

```ts
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
```

No `AbortSignal`, no timeout. Backend answers routinely take 70+ seconds; if the backend
hangs longer than that (Ollama stall, network blip), the UI is stuck on "Generating
answer..." for the browser's default timeout (~300s) with no way to cancel. In
`App.tsx`, `checkHealth`'s `cancelled` flag guards `setState` but never actually aborts
the underlying fetch, so a hung health check leaks silently.

**Fix:** give `request()` an `AbortController` + configurable timeout (e.g. 120s for
`/questions`, 15s for `/health` and `/config`). Have `useChat` hold the controller for
the in-flight question so unmounting (or a future Cancel button) aborts it; have `App`'s
effect cleanup abort the health-check fetch.

**Test:** mock `fetch` rejecting with an `AbortError` → the hook surfaces an error turn
instead of hanging.

### 12. `frontend/src/hooks/useChat.ts:26-38` — no reentrancy guard inside the hook

Re-entrancy is prevented only by the UI's `disabled` prop on the submit button. The hook
itself has no in-flight guard, so any path that calls `ask` twice in quick succession
(a future keyboard binding, a race before React re-renders `disabled`) fires two
concurrent ~70s LLM calls; since responses are unordered, whichever finishes last wins
the render regardless of which question it actually answered.

**Fix:** add an in-flight ref inside the hook itself:
```ts
const inflight = useRef(false);
const ask = async (...) => {
  if (inflight.current) return;
  inflight.current = true;
  try { ... } finally { inflight.current = false; setPending(false); }
};
```
Pairs naturally with the AbortController from item 11.

**Test:** call `ask()` twice synchronously → only one fetch is made.

### 13. `api/ingestion.py:65-66` — delete-then-upsert is not atomic

```python
repository.delete_by_filename(settings, [chunk.filename for chunk in chunks])
repository.upsert(settings, points)
```

This order is required (deterministic point IDs share the filename, so upserting first
would let a filename-wide delete wipe the new points too) — but there's no transaction.
If `upsert` fails after `delete` succeeds, the document is left with **zero** indexed
chunks: a failed re-ingest attempt leaves the user worse off than before they retried.

**Fix (pragmatic):** on upsert failure, raise a clear error explicitly stating the
document is now un-indexed and must be re-uploaded. **Fix (better):** upsert new points
first, then delete only the specific stale point IDs (set difference by filename minus
the new chunk IDs), avoiding the destructive filename-wide delete entirely.

### 14. Dead / inert configuration

- `settings.embedding_provider` (`api/settings.py`) is read only to echo into
  `/config`'s response — `get_embedding_provider` in `dependencies.py` always
  constructs `LocalEmbeddingProvider` regardless of its value. It looks like a working
  switch but isn't.
- `settings.upload_dir` (`api/settings.py`, default `"uploads"`) has zero references
  anywhere else in the codebase.

**Fix:** delete `upload_dir` entirely. For `embedding_provider`, either wire it into
`get_embedding_provider` to actually branch on the value, or remove it from settings and
`/config` — don't leave a setting that looks functional but silently does nothing.

---

## P3 — Tests, accessibility, polish

### 15. Backend test coverage gaps
- No test exercises the 502 error paths (`GenerationError`, `RerankingError` →
  `bad_gateway_handler`) or the 500 paths (`RetrievalPayloadError`, `EmbeddingError` →
  `internal_error_handler`) — the frontend's `extractErrorDetail` depends on this exact
  response shape, so it should be pinned by a test.
- `tests/test_api.py` sends `"   "` (whitespace) for the blank-question case, which
  passes the schema's `min_length=1` and is only caught later by the pipeline's strip
  check → 400. The `""` (truly empty) case, which should 422 at the schema boundary, is
  never asserted.
- No `tests/test_settings.py` — the validation bounds (`llm_temperature` between 0.0 and
  2.0, `llm_request_timeout_seconds > 0`, `chunk_overlap_tokens >= 0`) are never tested
  against an out-of-range env value.
- No dedicated test for `api/dependencies.py`'s `clear_dependency_caches` (every API
  test's fixture relies on it clearing state correctly between tests).

**Fix:** add `tests/test_settings.py`; extend `tests/test_api.py` with fake
generator/reranker doubles that raise the relevant domain errors, asserting 502/500
response shapes; add the empty-string 422 case.

### 16. Frontend test coverage gaps
- `src/hooks/useChat.ts` — the most logic-heavy, most bug-prone module (async state,
  pending toggling, error-turn mapping) has **zero** tests. This is exactly where items
  11 and 12 live.
- `src/components/UploadPanel.tsx` — no test for the upload flow: early-return on no
  file / while uploading, input reset after success, success/error rendering.
- `src/components/QuestionInput.tsx` — no test for Enter-to-submit vs. Shift+Enter
  (newline, not submit), trim/empty rejection, value clearing after submit.
- `src/components/ChatThread.tsx` — no test for empty state vs. pending indicator vs.
  rendered turns.
- `src/api/client.ts` — `uploadDocument` (FormData construction) and the `config`/
  `health` happy paths are untested; only `askQuestion` and `extractErrorDetail` are
  covered.

**Fix:** add `frontend/tests/hooks/useChat.test.ts` first (mock `api.askQuestion`,
assert turn sequence, `pending` lifecycle, and error-turn mapping for `ApiClientError`
vs. generic errors), then `UploadPanel` and `QuestionInput` interaction tests.

### 17. Accessibility gaps on dynamic/interactive elements
- `ChatThread.tsx` — the "Generating answer..." pending indicator has no
  `role="status" aria-live="polite"`; screen-reader users get no announcement that a
  70-second answer is in progress.
- `StatusBadge.tsx` — API-connectivity transitions (`checking → ok → error`) are silent
  to assistive tech; add `role="status" aria-live="polite"`.
- `App.tsx` — the "Cannot reach the DocRAG API" banner is a plain `<div>` with no
  `role="alert"`, so it won't be announced when it appears.
- `ChatMessage.tsx` — error turns are conveyed by color/class only; add `role="alert"`
  when `turn.role === 'error'`.
- `QuestionInput.tsx` — the textarea has only a `placeholder`, no `aria-label`;
  placeholders are not accessible names.

### 18. Minor polish
- `api/documents.py`'s overlap-config guard
  (`if settings.chunk_overlap_tokens >= settings.chunk_size_tokens: raise ValueError(...)`)
  raises a bare `ValueError`, surfacing as a 500 rather than a clean 400 config error —
  make it a `DocumentError` subclass instead.
- `api/repository.py`'s `delete_by_filename` loops per-filename with `wait=True`
  (O(n) blocking round-trips). Only caller today is single-file ingest (n=1), so this is
  low priority — collapse to one delete with `Filter(should=[...])` if/when
  multi-file ingest is added.
- The manual RRF fallback (`api/repository.py`, `reciprocal_rank_fusion`) uses 0-based
  ranks and is an independent reimplementation from the server-side `RrfQuery` path —
  document it explicitly as a best-effort fallback whose scores need not exactly match
  the server path, rather than implying strict parity.

---

## Verified NOT bugs (do not "fix" these)

- Chunk-overlap loop math in `api/documents.py` (`start`/`end`/`step` computation) is
  correct — no off-by-one, and the `overlap >= size` guard prevents an infinite loop.
- `create_collection(metadata=...)` is a real, supported parameter in the installed
  `qdrant-client` version.
- `ensure_ready` being called twice per query (once in `retrieve_candidates`, once
  inside `hybrid_search`) is cached per-client and negligible.
- `candidates[:fused_top_n]` in `api/reranking.py` is a redundant re-slice (retrieval
  already limits to `fused_top_n`) but harmless.

---

## Suggested implementation order

1. **P1-1, P1-3, P1-4 together** — one theme: wrap provider/infra failures into domain
   errors so they surface as correct 502/503 instead of raw 500. Touches
   `generation.py`, `reranking.py`, `embeddings.py`, `repository.py`, and `main.py`'s
   exception handlers, plus tests for each.
2. **P1-2** (defensive payload conversion) and **P1-5** (locking + index validation) —
   independent of each other and of step 1.
3. **P1-6** (upload size cap, both sides) and **P1-7** (compose env/host fix) —
   independent, low-risk quick wins.
4. **P2 items** individually; P2-9 (chunk sizing) needs an explicit re-ingest note and
   ideally the project owner's sign-off since it invalidates existing indexes.
5. **P3** test and accessibility batches last.

Each item should land with a red-without/green-with test proof, then the full backend
gate (`uv run pytest`, `ruff check`, `mypy`) and/or frontend gate
(`npm run test`, `npm run build`, `oxlint`) depending on what it touches.
