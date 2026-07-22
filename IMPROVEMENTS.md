# DocRAG Improvement Recommendations — Audit #2

> **Status banner (2026-07-14):** this audit predates the conversation-memory,
> document-management, chat-UX, and ops/quality milestone (per `CLAUDE.md`'s
> Current State). Several findings below are now stale/resolved — see the inline
> annotations on C3 and D2 — kept in place rather than deleted, since this file
> documents decision history `CLAUDE.md` still references (e.g. the reranker-swap
> approval in A2). Treat the rest of this document as a historical snapshot, not
> a current punch list.

Second full-codebase audit, done after all 12 actionable items from audit #1 (previous version of
this doc) were implemented and shipped, plus the `drop_params` gpt-5 fix and the frontend dark/light
redesign. This audit covers the current codebase, redesigned frontend included. Every finding below
was verified directly against source this session.

**Constraints for whoever implements these** (see `CLAUDE.md`):
- Hybrid RAG behavior is non-negotiable (RRF `k=60`, rerank fused top-N, citations with
  filename/page/section/chunk_id).
- ~~The reranker default `BAAI/bge-reranker-v2-m3` is a spec'd default — do NOT change it
  unilaterally~~ **Resolved:** owner approved swapping the default to
  `jinaai/jina-reranker-v2-base-multilingual` (see A2).
- Git is read-only by default; follow the repo's validation policy (backend-verify / frontend-verify
  workflows) for every change.
- Implement one item (or one tightly-related group) at a time, with a red-without/green-with test
  proof, then run `uv run pytest`, `ruff`, `mypy` (backend) and `npm run test`, `npm run build`,
  `oxlint`/`npm run lint` (frontend) before moving to the next.

---

## Tier A — Highest value (correctness / newcomer experience)

### A1. Test isolation: real `.env` leaks into unit tests (proven failing this session)

`api/settings.py:10` — `SettingsConfigDict(env_file=".env", ...)` means every
`AppSettings(**overrides)` constructed in tests still reads the developer's real `.env` for any
field not explicitly overridden.

Proven: `tests/test_api.py::test_config_endpoint_exposes_non_secret_settings` fails on any machine
with a real `OPENAI_API_KEY` in `.env` (asserts `openai_available is False`, gets `True`). Every
`make_settings()` helper across the test suite has the same latent exposure — model names, chunk
sizes, provider settings can all silently leak from the developer's local `.env` into "isolated"
unit tests.

**Fix:** add `tests/conftest.py` with a fixture, or pass `AppSettings(_env_file=None, **overrides)`
in each `make_settings()` helper (pydantic-settings supports this per-instantiation override).
Prefer the `_env_file=None` approach — it's explicit at the call site and doesn't rely on fixture
ordering.

**Test:** run the full suite with a real `OPENAI_API_KEY` (and other overrides) present in `.env` —
all green regardless.

### A2. Out-of-box `docker compose up` still breaks on first question — RESOLVED

`api/settings.py:20` defaulted `reranker_model` to `BAAI/bge-reranker-v2-m3`; confirmed against
fastembed's `main` branch (not just the pinned 0.8.0) that this model is not loadable by fastembed
at any version — `TextCrossEncoder.list_supported_models()` never lists it. A newcomer following
the README quickstart got a clean 502 on their very first question.

**Owner decision:** default changed to `jinaai/jina-reranker-v2-base-multilingual` — the strongest
cross-encoder fastembed does support (1K context, ~1.1 GB). Updated in `CLAUDE.md`, `api/settings.py`,
`.env.example`, and the local `.env`. Swapping the reranker does not invalidate the Qdrant index
(the embedding tag is derived only from the dense model). See D2 for the now-removed README callout.

### A3. No request length cap on questions

`api/schemas.py:49` — `question: str = Field(min_length=1)` has no `max_length`. A multi-hundred-KB
question embeds fine (FastEmbed truncates) but is then pasted verbatim into the generation prompt
(`api/generation.py build_grounded_messages`), which can blow the LLM's context window or the
provider's request-size limit, surfacing as an opaque provider error instead of a clean validation
error.

**Fix:** `question: str = Field(min_length=1, max_length=4000)` (tune the exact number), plus a
test asserting an oversized question gets a 422. Mirror a `maxLength` on the frontend's
`QuestionInput` textarea for early feedback.

### A4. Focus visibility regressions in the redesign (introduced this session)

- `frontend/src/components/ProviderSelector.css` — `.provider-selector__select:focus { outline:
  none; }` leaves keyboard users with no focus indicator on the provider dropdown.
- `frontend/src/components/QuestionInput.css` — `.question-input__textarea:focus { outline: none;
  }` — same problem on the main question input.
- `frontend/src/components/UploadPanel.tsx` / `.css` — the real `<input type="file">` sits
  `opacity: 0` over the drop-zone; when it receives keyboard focus there is currently no visible
  indication anywhere in the UI.

**Fix:** add `:focus-within` styling to the pill/chip containers (e.g. `border-color:
var(--accent)` on `.provider-selector__control` and `.question-input`), and a focus-visible style
on `.upload-panel__dropzone` when its hidden input is focused (`:has(.upload-panel__input:focus-visible)`
with a sibling-selector fallback for broader support). Verify by tabbing through the whole app in
both themes.

### A5. Remaining accessibility batch (carried from audit #1, still open)

- `ChatThread.tsx` pending indicator ("Generating answer...") — add `role="status"
  aria-live="polite"`.
- `StatusBadge.tsx` — add `role="status"` so API-connectivity transitions are announced.
- `App.tsx` unreachable banner — add `role="alert"`.
- `ChatMessage.tsx` error bubble — add `role="alert"`.
- `QuestionInput.tsx` textarea — add `aria-label="Ask a question about your documents"`
  (placeholder text is not an accessible name).

All one-attribute changes; add an assertion to each component's existing test file.

---

## Tier B — Robustness & quality

### B1. `localStorage` can throw (Safari private mode / storage-denied contexts)

`frontend/src/App.tsx toggleTheme` calls `localStorage.setItem('docrag-theme', next)` unguarded —
this throws in some private-browsing/embedded contexts, crashing the click handler after the theme
state has already flipped visually. `frontend/src/main.tsx`'s `localStorage.getItem(...)` at module
load has the same exposure, which would prevent the app from booting at all in a storage-denied
context.

**Fix:** wrap both calls in try/catch; theme persistence silently degrades to per-session-only
instead of crashing. **Test:** mock `localStorage` throwing on `getItem`/`setItem` → app still
renders, toggle still flips the visible theme.

### B2. No way to cancel an in-flight generation

`useChat` already builds an `AbortController` per request (for unmount-abort) but never exposes a
way to trigger it from the UI. Ollama generations routinely take 60-70+ seconds.

**Fix:** have `useChat` return a `cancel()` that aborts the current controller; add a small "Stop"
control in `ChatThread` while `pending` is true (or swap the "Ask" button to "Stop"). The abort path
already produces a clean error turn ("Request timed out or was cancelled") via the existing client
mapping — no new error-handling logic needed. **Test:** cancel mid-flight → pending clears, an error
turn appears, and a subsequent `ask()` still works normally.

### B3. Upload success/error state never resets

`App.tsx`'s `uploadState` keeps showing the success line indefinitely after a successful upload
(and a stale error persists) until another upload completes. **Fix:** clear `uploadState` back to
`{status: 'idle'}` in `UploadPanel.handleFileChange` when the user selects a new file, so switching
files clears the previous result immediately rather than waiting for the next upload to finish.

### B4. Ingestion atomicity — the "better" fix from audit #1 is still open

`api/ingestion.py ingest_chunks` deletes a filename's old points before upserting new ones; a
failure between the two leaves the document un-indexed. This is now clearly reported via
`IngestionError` (audit #1's fix), but still lossy on failure. The better fix sketched back then is
still unimplemented: upsert new points first (deterministic IDs overwrite matching old ones in
place), then delete only the point IDs that are genuinely stale (existing IDs for that filename
minus the new chunk IDs).

**Fix:** add a `point_ids_for_filename` method to `VectorRepository` (scroll with a filename
filter, `with_payload=False`), reorder `ingest_chunks` to upsert-then-delete-stale. **Test:** an
upsert failure now leaves the OLD version of the document fully queryable; a shrunk document
(fewer chunks on re-ingest) correctly removes the now-stale chunk IDs.

### B5. `delete_by_filename` per-filename loop (carried, trivial, bundle with B4)

`api/repository.py:69-87` loops one blocking delete per filename with `wait=True`. Collapse to a
single delete using `Filter(should=[FieldCondition(...) for each filename])` when
`len(filenames) > 1`. Only one filename is ever passed today, so this is low value on its own —
worth doing only as part of B4 since it touches the same method.

### B6. Startup model warmup (optional, carried from audit #1)

The first question after boot pays embedding + reranker model download/load cost (can be tens of
seconds), which stacks with a cold Ollama model load and can trip the frontend's 120s request
budget. **Fix:** add a FastAPI lifespan hook that calls `embedding_provider.embed_texts(["warmup"])`
and `reranker.score("warmup", ["warmup"])` once at startup (in a thread, logging duration) so a
misconfigured model fails loudly at boot instead of on a user's first request. Gate behind a
`WARMUP_MODELS` setting (default `true`) so tests/dev can disable it.

---

## Tier C — Tests & CI

### C1. Backend test gaps

- `""` (truly empty) question → should 422 at the schema boundary (`min_length=1`); only `"   "`
  (whitespace, caught later by the pipeline) → 400 is currently tested.
- 502 response-shape pinning: a fake generator raising `GenerationError` through the live API →
  assert 502 + the exact `detail` string shape (the frontend's `extractErrorDetail` depends on it).
  Same for a fake reranker raising `RerankingError`.
- 503 shape: a fake repository raising `VectorStoreUnavailableError` → assert 503.
- New `tests/test_settings.py`: validation bounds (`llm_temperature` 0.0–2.0,
  `chunk_overlap_tokens >= 0`, `llm_num_retries >= 0`, `llm_request_timeout_seconds > 0`) rejected
  when out of range; also covers the A1 `_env_file=None` isolation fix itself.

### C2. Frontend test gaps

- New, currently-untested redesign behavior: theme toggle (flips `data-theme`, persists to
  `localStorage`, restores correctly on reload), the `main.tsx` pre-paint init path, `UploadPanel`
  drag-and-drop (valid file drop, oversized file drop → size error, same as the existing
  click-to-browse test coverage).
- Carried from audit #1: `QuestionInput.test.tsx` (Enter submits, Shift+Enter inserts a newline,
  trims whitespace, disabled state blocks submit), `ChatThread.test.tsx` (empty state, pending
  indicator, rendered turn list).

### C3. No CI — RESOLVED

`.github/workflows/ci.yml` now exists with backend, frontend, and Docker-build jobs, matching the
shape recommended below. Left in place for context on the original ask.

No `.github/workflows/` directory exists. Add one workflow with two jobs mirroring the local
validation gates:
- **backend** — `uv sync --extra test`, `uv run pytest`, `uv run ruff check`,
  `uv run mypy api` (needs `--extra dev` for mypy/ruff).
- **frontend** — `npm ci`, `npm run test`, `npm run build`, `npm run lint`, run from `frontend/`.

Trigger on push and PR to `main`. Note: land A1 first — CI runners won't have a `.env` at all
(which is fine either way, but A1 makes that irrelevant rather than incidentally-working).

---

## Tier D — Docs & polish

### D1. This document was stale

12 of the 18 findings in the previous version of `IMPROVEMENTS.md` were implemented and shipped.
This audit replaces it wholesale — that replacement is the deliverable of the audit task itself.

### D2. README drift risk around the reranker default — RESOLVED

Was: the README's Docker quickstart implied `docker compose up` + ask a question works out of the
box, but with A2 unresolved the first question 502s. Now that A2 is resolved (new default is
fastembed-loadable), the "known issue" callout describing the 502 has been removed from the README
and replaced with a note about the new reranker's larger (~1.1 GB) first-download size.

~~Still open: document the `eval` extra (`uv sync --extra eval`) and a usage example for
`eval/ragas_runner.py` — the module is complete, tested, and functional but currently unreachable
from any documentation, so nobody would discover it exists.~~ **RESOLVED:** the README's
"Evaluation (optional)" section documents the extra, dataset format, and run command, and
`eval/datasets/sample_eval.jsonl` ships a runnable example.

### D3. Minor code-quality notes (fix opportunistically, not worth standalone PRs)

- ~~`api/generation.py cited_sources` — when the model declines to answer ("insufficient context")
  in its own words without citing any `[n]` marker, the fallback currently attaches ALL context
  sources to that non-answer.~~ **Resolved:** the fallback was removed — zero `[n]` markers now
  returns zero sources, since an answer citing nothing is exactly the case that shouldn't imply
  every candidate chunk is a source.
- `api/documents.py` `ChunkConfigError` currently maps to 400 (user-blame) though the root cause is
  server-side configuration (`CHUNK_OVERLAP_TOKENS`/`CHUNK_SIZE_TOKENS`), not user input — arguably
  should be a 500. Low impact either way; noting for awareness only.
- `frontend/src/components/UploadPanel.tsx` — dropping multiple files onto the drop-zone silently
  takes only the first (`event.dataTransfer.files?.[0]`). Consider a brief hint or explicit
  rejection message when more than one file is dropped.
- Verified: no dead `--bg-secondary` token references remain anywhere after the redesign.

---

## Verified NOT issues (checked this audit — do not "fix")

- Starlette's exception-handler dispatch is MRO-specific: `GenerationConfigError` (→ 400) is
  correctly matched over its parent `GenerationError` (→ 502) because Starlette walks
  `__mro__` and the most-specific registered handler wins.
- `completion_model_and_kwargs` in `api/generation.py` is called OUTSIDE the try/except in
  `LiteLLMGenerator.complete` — config errors (`GenerationConfigError`, e.g. missing API key)
  correctly bypass the 502 wrapper and hit the 400 handler directly, as intended.
- `Dockerfile`'s `uv sync --frozen` does not install the `test`/`dev`/`eval` optional-dependency
  groups (they're `[project.optional-dependencies]`, not installed by default) — the runtime image
  is not bloated with test/dev tooling.
- Root `.gitignore` doesn't list `node_modules/`, but `frontend/.gitignore` covers it (and `dist`
  is covered in both places) — no actual gap.
- The manual RRF fallback (`api/repository.py reciprocal_rank_fusion`) uses 0-based ranks and is an
  independent reimplementation from the server-side `RrfQuery` path — already documented in-code as
  a best-effort fallback, not claiming exact parity. No further action.
- The collection-readiness cache's double-checked locking (`api/qdrant_schema.py`) is correct as
  written — GIL-safe, re-checks the cache inside the lock before doing real work.

---

## Suggested implementation order

1. **A1** (test isolation) — unblocks everything else; the suite must be green with a real `.env`
   present before other work is easy to verify.
2. **A3 + A4 + A5** — small and independent: one backend schema field, one frontend CSS/a11y batch.
3. **B1 + B3** (frontend robustness quick wins), then **B2** (cancel button — small feature, touches
   `useChat` + `ChatThread`).
4. **C1 + C2** (test batches), then **C3** (CI — do this once the suites are fully hermetic per A1).
5. **B4 (+ B5)** — ingestion atomicity; the only structurally meaningful backend behavior change.
6. **B6** (startup warmup) and **D2** (README) any time; **A2** blocks on an explicit owner decision.

Each item should land with a red-without/green-with test proof, then the full backend gate
(`uv run pytest`, `ruff check`, `mypy`) and/or frontend gate (`npm run test`, `npm run build`,
`npm run lint`) depending on what it touches.
