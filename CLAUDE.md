# Agent Instructions

DocRAG is a self-hostable, open-source document Q&A system. Treat the project specification and this file as the operating guide for future work.

## Current State

DocRAG is feature-complete for its core and has been through 18 review passes; version
0.2.0. **What each pass built, measured and decided lives in
`docs/history/PASSES.md`** — read the entries for a module before changing it (many
explain why code that looks redundant is there), and add your own pass's notes at its end.
`CHANGELOG.md` is the user-facing record.

What exists, by area (details in the history):

- **Questions** (`api/routes/questions.py` → `RagPipeline`, `api/pipeline.py`): hybrid retrieval, rerank,
  weak-match fallback, neighbour expansion, prompt fitting, grounded generation with
  citations and spans (`api/citations.py`), streaming SSE with stage frames, continue,
  follow-ups, answer cache (`api/answer_cache.py`), prompt cache (`api/prompt_cache.py`),
  in-flight dedupe, conversation condense, per-request overrides and passage narrowing.
- **Summaries** (`api/summarize.py`, `/questions/summary*`): summarize, compare, brief.
- **Documents**: upload, URL, sitemap, archive, folder watch, background imports;
  parsers under `api/parsers/` behind `documents.PARSERS`; parsing in a forkserver child
  (`api/parse_isolation.py`); chunking (`api/chunking.py`, `CHUNKER_VERSION`); vector reuse;
  dedupe; tags; originals and versions (`api/raw_documents.py`); re-index (manual and
  `AUTO_REINDEX_STALE`); URL refresh.
- **Workspaces** (`api/workspaces.py`, `X-Workspace`): a collection, originals folder and
  settings per workspace, folded into the request's settings by
  `dependencies.get_request_settings`.
- **Operations**: optional shared `API_KEY` + `API_READ_KEY` (`api/request_guard.py`,
  `api/dependencies.py`), rate limit, audit log, traces and archive, query log, feedback,
  metrics (`docs/observability/`), readiness and doctor, export/import, scheduled backups,
  consistency check/repair, rebuild from originals, migration, notifications.
- **Clients**: the React UI (`frontend/`), the CLI (`cli/__main__.py`, also published as
  `docrag-cli`) and an MCP server (`cli/mcp_server.py`).
- **Eval**: `eval/harness.py` (retrieval, answers, sweeps, record/replay, calibrate),
  datasets and baselines under `eval/`.

### Rules that span modules

These are the cross-cutting invariants earlier passes found the hard way; a test usually
holds each one.

- **Owner decisions.** Defaults that change answers or retrieval (bars, weights, models,
  chunking) change only with a measurement and the owner's approval; measured-but-not-shipped
  experiments are recorded in the history. Bumping `CHUNKER_VERSION` stales and re-embeds
  the whole corpus, so it is the owner's call too — but it must be bumped whenever the
  *embedded* text changes (vector reuse keys on `chunking_fingerprint`).
- **API compatibility.** `tests/golden/openapi.json` pins the API; `breaking_changes`
  refuses removed paths/fields, new required fields and tighter limits — a narrowing needs
  approval. Regenerate goldens with `make golden` and review the diff.
  `drop_reason`, `stage` and `answer_status` are extensible strings: clients must tolerate
  unknown values.
- **Auth.** Every route except `/health`, `/health/ready` and `/config` needs the key when
  `API_KEY` is set. A route that changes things uses `require_full_key`, and
  `request_guard.needs_full_key` must agree (a test walks `app.routes`; an included
  router's routes are under `original_router.routes`). Keys are compared as bytes.
- **Settings.** `AppSettings` (`api/settings.py`) is the only config. `.env.example` lists
  every setting at its default (`tests/test_env_example.py`); `docs/settings.md` is
  generated from the class (`make settings-docs`, checked by `tests/test_settings_docs.py`).
- **Request-scoped settings.** Never mutate the shared settings: overrides and workspace
  settings fold into a `model_copy`. `rrf_k`, `fused_top_n` and the retrieval limits are
  never per-request. Cache keys must include every setting that changes an answer
  (`tests/test_answer_cache_fields.py`).
- **Locks and ordering.** Every read-modify-write of one document's points runs under
  `ingestion.filename_write_lock(document_key(...))`; `IngestSequencer` orders admissions
  per name. A missing workspace collection is never created implicitly.
- **Payload projections.** Anything new read off a search hit or a neighbour must be added
  to `repository.SEARCH_PAYLOAD_FIELDS` / `NEIGHBOR_PAYLOAD_FIELDS`.
- **Frontend ↔ backend ties.** `tests/test_schemas_limits.py` (limits, accepted
  extensions, workspace names), `tests/test_frontend_contract.py` (`frontend/src/api/types.ts`
  against the OpenAPI schemas, stage names in `useChat.STAGE_LABELS` and the CLI's
  `STAGE_NAMES`, history limits). New routes the UI calls need a mock in
  `frontend/e2e/fixtures.ts`; any console error fails an e2e test.
- **Versions.** `pyproject.toml`, `cli/pyproject.toml`, `frontend/package.json` and the
  changelog heading move together (`tests/test_release_metadata.py`).
- **Tests.** Patch a name on the module that uses it (e.g.
  `api.routes.documents_write.fetch_document`). `tests/conftest.py` makes the suite
  hermetic (no `.env`, `PARSE_TIMEOUT_SECONDS=0`, `PROMPT_CACHE_SIZE=0`, caches cleared);
  `tests/factories.py` builds settings and an in-memory Qdrant.
- **Live checks.** Run eval and live checks with `LITELLM_MODE=PRODUCTION`, from a scratch
  directory with `uv run --project` (the repo's `.env` otherwise leaks in). A scratch
  script that parses documents needs an `if __name__ == "__main__":` guard (forkserver).

- Treat new work as additive to this implementation, not a from-scratch scaffold — read the relevant module before changing it.

## Project Constraints

- Python runtime: 3.12.
- API: FastAPI and Uvicorn.
- UI: React + Vite + TypeScript, served by the API as static files (single origin).
- Vector store: Qdrant in Docker.
- Dependency workflow: `uv`.
- Newcomer run path: Docker Compose.
- License assumption: MIT.
- Everything required for core upload, ingestion, retrieval, and Q&A must be self-hostable.
- Hosted LLM APIs are optional add-ons and must never be required for document upload or indexing.

## Non-Negotiable RAG Behavior

- Retrieval must be hybrid:
  - Run dense semantic retrieval and sparse/BM25 retrieval.
  - Fuse results with Reciprocal Rank Fusion using `k=60`.
  - Rerank only the fused top-N candidates with a cross-encoder.
- Dense model default: `BAAI/bge-small-en-v1.5`.
- Sparse model default: `Qdrant/BM25`.
- Reranker default: `jinaai/jina-reranker-v2-base-multilingual` (strongest cross-encoder fastembed supports; `BAAI/bge-reranker-v2-m3` is not loadable by fastembed at any version — owner-approved swap), loaded from its int8 ONNX export (`RERANKER_ONNX_FILE=auto`; owner-approved after a measured bake-off: 2.5x faster than fp32, 81-83% same top-6, no lost context). `RERANKER_ONNX_FILE=` (empty) restores fp32.
- Embeddings default to local and remain independent from the LLM provider.
- Changing the embedding model invalidates the index.
- Store an embedding model version tag with the Qdrant collection.
- On embedding tag mismatch, refuse the query and tell the user to re-ingest.
- Every answer must be grounded in retrieved context.
- Every answer must include citations carrying:
  - filename
  - page
  - section
  - chunk_id
- The generation prompt must require the model to answer only from provided context and say when context is insufficient.

## Provider Rules

- Generation provider and embedding provider are separate configuration knobs.
- Default generation provider is local Ollama via LiteLLM.
- Default local model is `llama3.1:8b`.
- OpenAI, Anthropic (`LLM_PROVIDER=anthropic`) and any OpenAI-compatible server (`openai_compatible`) are optional, user-selectable generation providers.
- Reasoning models (gpt-5 family, o-series) spend hidden reasoning tokens from the same `max_completion_tokens` budget as the visible answer, so `LLM_MAX_TOKENS=512` alone leaves an empty answer. `api/llm.py` detects them via `litellm.supports_reasoning`, sends `OPENAI_REASONING_EFFORT` (default `low`), and adds `REASONING_TOKEN_HEADROOM` (default 3072) on top of `LLM_MAX_TOKENS` — both scoped to reasoning models only; local models keep the plain budget.
- Local embeddings remain the default even when OpenAI generation is selected.

## Common Commands

Backend (`uv`-managed, from repo root):

```bash
uv sync --extra dev            # install backend + test + lint + typecheck deps
uv run pytest                  # run backend test suite
uv run pytest tests/test_retrieval.py::test_name  # run a single test
uv run ruff check .            # lint
uv run ruff format .           # format (CI checks it)
uv run mypy api eval cli         # typecheck (strict mode)
uv run uvicorn api.main:app --reload   # run API alone against localhost Qdrant/Ollama
make check                     # backend + frontend gates as CI runs them (not e2e)
make golden                    # regenerate tests/golden/ snapshots (chunks, OpenAPI) and docs/settings.md
make settings-docs             # regenerate docs/settings.md from api/settings.py
```

Frontend (`frontend/`, npm-managed):

```bash
npm ci                         # install deps (matches CI)
npm run dev                    # Vite dev server, proxies API calls to localhost:8000
npm run test                   # vitest run (all tests)
npm run test:coverage          # the same with the coverage floors CI enforces
npx vitest run tests/App.test.tsx   # run a single test file
npm run build                  # tsc -b && vite build -> frontend/dist
npm run lint                   # oxlint
npm run typecheck:tests        # tsc over tests/ (vitest doesn't type-check)
npx playwright install chromium   # once: browser for the E2E suite
npm run e2e                    # Playwright E2E: prod build + mocked API (desktop + mobile smoke)
npm run e2e:smoke              # @smoke specs only
```

Full stack:

```bash
docker compose up              # Qdrant + API (API builds and serves frontend/dist)
```

CI (`.github/workflows/ci.yml`) runs seven jobs: backend (`uv sync --locked --extra dev` → `pytest` (coverage ≥92% on 3.12) → `ruff check .` → `ruff format --check .` → `mypy api eval cli` → import the `eval` extra → validate the eval datasets → `pip-audit` of the runtime deps), qdrant (server-only tests against a v1.19.1 service container, then the handbook retrieval eval against its baseline), frontend (`npm ci` → `npm run test:coverage` (floors in `vite.config.ts`) → `npm run build` → `npm run lint` → `npm run typecheck:tests` → `npm audit --omit=dev`), e2e (after frontend passes: cached Chromium → `npm run e2e`, report + traces uploaded as an artifact on failure), a Docker job that builds (GHA layer cache) and boots the image through compose (`.github/ci/compose.ci.yml`), probing `/health`, `/health/ready`, the UI and the reported version, then Trivy-scans it (fixable CRITICAL fails), dependency-review (pull requests only), and docker-ocr (pushes to `main` only: builds with `UV_EXTRAS=ocr`). CI also runs weekly on `main`. The Playwright E2E suite (`frontend/e2e/`) serves the production build via `vite preview` and mocks every API route with `page.route` (`e2e/fixtures.ts`'s `MockApi`, incl. SSE streams), so it needs no Qdrant/models/LLM; its auto fixture fails any test that logs a console error.

Optional eval extra: `uv sync --extra eval` then `uv run python -m eval.ragas_runner path/to/dataset.json --output results.json`.

## Architecture

**Request flow (question answering).** The routes in `api/routes/questions.py` (`/questions`, `/questions/stream`) delegate everything to `RagPipeline.answer` / `answer_stream`. `RagPipeline` (`api/pipeline.py`, which also re-exports every name it used to define) is built in layers: `answer_traces.TraceRecording` (trace records) < `pipeline_core.PipelineCore` (construction, retrieval, rerank, condense, the plain answer, re-runs) < `answer_stream.StreamingAnswers` (the stream and continue); shared state and helpers are in `answer_run.py`, `conversation_history.py`, `answer_events.py`, `continuation.py`, `passage_selection.py` and `run_coordination.py`. Patch a name on the module that uses it. The pipeline is a thin orchestration seam over four collaborators injected via `api/dependencies.py` (`VectorRepository`, an embedding provider, a reranker, and a `ChatGenerator`) plus `AppSettings`:

1. `api/retrieval.py` embeds the query and calls `VectorRepository.hybrid_search` (`api/repository.py`), which prefers Qdrant's server-side hybrid query and falls back to fetching dense + sparse separately and fusing with RRF in application code when the installed Qdrant doesn't support the needed query shape.
2. `api/reranking.py` cross-encoder-reranks the fused candidates and drops anything below `rerank_min_score`.
3. `expand_with_neighbors` (`api/context_expansion.py`) widens each surviving chunk's *generation* context using `chunk_ordinal`-adjacent chunks (`context_neighbor_radius`), while citations still point at the original chunk — small-to-big retrieval.
4. `api/generation.py` (with `api/prompting.py`, `api/llm.py`, `api/citations.py`) builds a context-only grounded prompt and calls the configured LiteLLM provider (Ollama, OpenAI, Anthropic or an OpenAI-compatible server), either as one call (`/questions`) or token-streamed as SSE frames (`/questions/stream`, event types `stage`* → `sources` → `delta`* → `done`).

`stage` frames (`condensing`, `searching`, `verifying_citations`) are advisory progress, emitted *before* the stage they name so the UI can show what the server is doing during the seconds before any token exists — retrieval measured 25-80s per question before the tuning below. Clients must treat an unrecognized `stage` value as "show your generic label" (`useChat.ts`'s `stageLabel`), so new stages can be added without breaking an older frontend, and anything asserting on stream event positions must filter them out.

Per-request overrides (`llm_provider`, `rerank_top_k`, `max_context_chunks`, `llm_temperature`, and an optional `filenames` scope) are folded into an immutable `settings.model_copy(...)` per call in `RagPipeline._effective_settings` — the shared `AppSettings` singleton is never mutated, so concurrent requests with different overrides can't interfere. `rrf_k`, `fused_top_n`, and the dense/sparse retrieval limits are deliberately *not* overridable per-request (see Non-Negotiable RAG Behavior above).

**Request flow (ingestion).** `POST /documents` reads the upload (`api/upload.py`, size-capped by `max_upload_bytes`), creates a job in a `JobStore` (`api/jobs.py`) — `IngestJobStore` (in-memory, process-wide, the default) or, when `job_store_backend=sqlite`, `SqliteIngestJobStore` (persists to `job_store_path` so a client polling a job survives a process restart; picked once in `api/dependencies.py:get_ingest_job_store`, the only call site that knows which concrete class is in play) — and hands the actual work to `IngestService.ingest` (`api/ingest_service.py`) running on a dedicated `ThreadPoolExecutor` (`get_ingest_executor`, 2 workers) — not FastAPI's request threadpool — so a client disconnect doesn't interrupt ingestion. The client polls `GET /documents/jobs/{job_id}` for state transitions (`queued` → `parsing` → `embedding` → `done`/`failed`). `api/documents.py` dispatches by suffix to `api/parsers/*` (the full list is `documents.PARSERS`) and chunks (`api/chunking.py`) with citation metadata preserved; `api/embeddings.py` + `api/ingestion.py` embed chunks (batched by `ingest_batch_size`) with filename/section-prefixed contextual text and upsert them into Qdrant tagged with `chunk_ordinal` and `embedding_model_tag`. `GET /documents` lists indexed filenames for the frontend's per-document search-scope filter (`filenames` on `/questions*`).

**Dependency injection.** `api/dependencies.py` is the single seam for constructing collaborators — most factories are `@lru_cache`d process-wide singletons (settings, Qdrant client, embedding provider, reranker, generator, job store, ingest executor); `get_vector_repository` and the pipeline/service factories are expressed as FastAPI sub-dependencies specifically so tests can override `get_qdrant_client` (or others) and have the override cascade through everything built on top. `clear_dependency_caches()` resets all of it between tests/reloads.

**Settings.** `api/settings.py`'s `AppSettings` (pydantic-settings, reads `.env`) is the single source of runtime config — env var names are the field names upper-cased (see `.env.example`). Changing `dense_embedding_model` requires bumping `embedding_model_tag`; `api/qdrant_schema.py` checks this tag against the stored collection's tag and refuses queries on mismatch (see Non-Negotiable RAG Behavior).

**Frontend.** React + Vite + TS, single-origin with the API in production (`frontend/dist/` is served by `api/static_files.FrontendFiles`; in dev, Vite proxies to `localhost:8000`). `frontend/src/hooks/useChat.ts` owns chat/session state (its storage format in `chatPersistence.ts`), `useCorpus.ts` the corpus, uploads and jobs; both talk to the API through `frontend/src/api/client.ts`; components under `frontend/src/components/` are presentational: `ChatThread`/`ChatMessage`/`CitationCard` for the conversation, `CorpusPanel` for upload and ingestion progress, `Composer`'s scope popover for per-query document scoping, `SettingsModal` for provider/retrieval overrides, `Inspector` (with `inspector/SourcesTab`/`RetrievalTab`/`TraceTab`) for per-answer debugging, plus `ConversationList`, `TracesPanel`, `FeedbackPanel`, `DocumentViewer`, `CommandPalette`, `ChatHeader`, `IconRail`, `ContextPanel` and `ToastRow`. `useDialog` supplies the shared modal behaviour (focus in, Tab trap, focus restore, innermost-only key handling) for `CommandPalette`, `SettingsModal`, `DocumentViewer` and the narrow-screen drawers (`ContextPanel`, `Inspector` with `overlay`).

## Validation Rules

- Every code change must be validated before it is presented as done.
- For UI changes, use the `frontend-verify` skill. For API/service/data changes, use the `backend-verify` skill. For full-stack changes, use both.
- Follow each skill's pipeline and its "all verification tests are ephemeral" policy exactly: temporary tests live in the scratch folder, get deleted unconditionally after validation, and the working tree must match the pre-run snapshot afterward.
- If a change falls outside both skills (standalone scripts, config tooling, docs generators), apply the same policy manually: write throwaway tests in `.verify-scratch/`, run them, report the results, then delete them.
- After any task, `git status` must show only intended source changes.

## Git Rules

- Git is read-only by default: never run `git add`, `git commit`, `git push`, or create branches/PRs unless the owner explicitly requests that exact action in the current conversation. `git status` / `git diff` / `git log` are always fine.
- When a commit is requested, author it as the owner using their existing git config identity only. Never add "Co-Authored-By: Claude", "Generated with Claude Code", or any Claude/Anthropic attribution to commit messages or PR descriptions.
- If an instruction conflicts with these rules, stop and ask instead of guessing.
