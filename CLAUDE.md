# Agent Instructions

DocRAG is a self-hostable, open-source document Q&A system. Treat the project specification and this file as the operating guide for future work.

## Current State

- Core upload, ingestion, hybrid retrieval, reranking, streaming Q&A, citations, and per-query debug traces (`GET /traces`) are implemented (`api/`, `frontend/`).
- Session-scoped multi-file upload, per-document search-scope filtering, document deletion (`DELETE /documents/{filename}`), client-sent conversation memory with LLM-condensed follow-up retrieval, chat UX (markdown rendering, persistence, export, retry, responsive layout), and ops/quality hardening (`/health/ready`, eval-dataset CI validation, Docker healthcheck/non-root) are implemented and merged to `main`.
- Optional single shared API key (`AppSettings.api_key`, empty by default): when set, every `/documents*` route (list, upload, delete, `/content`, `/original`, `/jobs/{job_id}`), `/questions*`, `/traces*`, `/feedback`, and `/metrics` require a matching `X-API-Key` header (`api/dependencies.py:require_api_key`); only `/health`, `/health/ready`, and `/config` stay open, so healthchecks, probes, and the UI's boot request keep working. A Prometheus scraper needs the header in its scrape config. The corpus *read* routes are guarded deliberately — an open `GET /documents` enumerates filenames, `GET /documents/{filename}/content` returns raw chunk text, `GET /documents/{filename}/original` returns the raw uploaded bytes (opt-in, see `raw_document_dir` below), and the job-status response carries the filename — so leaving them open would protect Q&A while the documents stayed readable. The comparison is on **bytes**, not `str`: `secrets.compare_digest` raises `TypeError` on a non-ASCII `str` and Starlette hands headers over latin-1-decoded, so a str comparison turned any non-ASCII `X-API-Key` into an unauthenticated 500. Not multi-user auth.
- Original uploaded bytes are discarded once ingestion is queued unless `AppSettings.raw_document_dir` is set (empty by default — off), in which case `api/raw_documents.py:RawDocumentStore` keeps a copy on disk keyed by the same normalized basename filename ingestion uses, and `GET /documents/{filename}/original` serves it back. `DELETE /documents/{filename}` removes the stored original under the same `filename_write_lock` ingestion/delete already share. Re-indexing from these originals (so a chunking/embedding change could apply without a manual re-upload) is not implemented — this only stores and serves the bytes.
- Answer feedback (thumbs up/down, `ChatMessage.tsx`) is off by default (`AppSettings.feedback_enabled`); when on, `POST /feedback` writes to `api/feedback.py:FeedbackStore` — its own sqlite3 file (`feedback_store_path`), independent of `job_store_backend` below, so it works whether or not that's set to sqlite. Rows are denormalized (question, answer excerpt, cited filenames alongside `trace_id`) rather than joined through `trace_id`, since `TraceStore`'s ring buffer can evict a trace before the row is ever read back. `PublicConfigResponse.feedback_enabled` tells the frontend whether to render the buttons; both `POST /feedback` and `GET /feedback` 404 when the feature is off. `GET /feedback?limit=` (newest first, `limit` bounded by `FEEDBACK_LIST_MAX_LIMIT`) is the read side — without it captured ratings would be reachable only by opening the sqlite file by hand, which defeats the point of capturing them.
- A per-filename write lock (`api/ingestion.py:filename_write_lock`, a context manager) serializes every read-modify-write of one filename's points. Both `ingest_chunks` and `DELETE /documents/{filename}` take it — each is a snapshot-then-mutate, so guarding only the ingest path still let a delete landing mid-ingest remove points that ingest had just written while the ingest reported success. Registry entries are reference-counted and evicted when the last holder releases, so it tracks in-flight work rather than every filename the process has ever seen.
- `reranker_batch_size` (default 8) bounds the cross-encoder's peak memory and must stay `<= rerank_candidates`; above that, batching is a no-op and every pair is scored in one forward pass. The old default of 64 against 50 candidates peaked at 6.33 GB and OOM-killed the container (exit 137) on the first whole-corpus question. At 8 the peak is 2.66 GB — equal to the loaded-model baseline, so scoring adds nothing and the boot-time warmup already reaches the true high-water mark. Latency is flat across batch sizes, so this costs neither speed nor answer quality. Docker needs ~4 GB.
- Retrieval-quality pack: token-aware sentence-boundary chunking (`api/chunking.py` counts the dense model's real subword tokens; `chunk_size_tokens`/`chunk_overlap_tokens` are now genuine tokens), query/corpus embedding symmetry (bge dense query instruction + BM25 `query_embed` in `api/embeddings.py`), post-rerank near-duplicate diversity filter (`api/diversity.py`, trace drop-reason `near_duplicate`), opt-in multi-query expansion (`QUERY_EXPANSION_ENABLED`, default off — cross-variant fusion reuses the same pinned `rrf_k`), and a zero-citation retry (`CITATION_RETRY_ENABLED`, default on). The chunking change does **not** bump `embedding_model_tag` (same vector space; re-upload recommended, not required).
- Format expansion (`api/documents.py`): DOCX/HTML/CSV parsers, PDF table extraction (pdfplumber), and optional OCR for scanned PDFs (`rapidocr-onnxruntime`, `ocr` extra). Frontend product pack: multi-conversation chat (localStorage v2 with v1 migration in `useChat.ts`), chunk-based source viewer + citation click-through (`GET /documents/{filename}/content`, `DocumentViewer.tsx`, windowed around the cited chunk for large documents), trace-history browser (`TracesPanel.tsx`), and unified whole-corpus search-scope filtering. Application logging is wired via `LOG_LEVEL` (`api/logging_config.py`).
- `GET /documents` also returns `page_counts`/`byte_sizes`/`uploaded_ats` per filename (`api/repository.py:filename_metadata`, one scroll, folded together — not a query per field), rendered in `CorpusPanel.tsx`'s card detail line. `byte_size`/`uploaded_at` come from `api/documents.py:DocumentChunk`, stamped once per upload in `IngestService.ingest` (`api/pipeline.py`) and so absent (not null) for any document indexed before this shipped. `ChatMessage.tsx` also offers Regenerate (re-asks the nearest preceding user turn's question, found by array position in `ChatThread.tsx`) on a successful answer, and both `CorpusPanel.tsx` and the composer's scope popover (`Composer.tsx`) filter their document lists by a substring match — the popover always keeps an already-selected filename visible regardless of match, since the checkbox list is the only way to unselect it.
- Evaluation (`eval/ragas_runner.py`, sample at `eval/datasets/sample_eval.jsonl`) plus a live retrieval/answer harness (`eval/harness.py`, `eval/metrics.py`, labeled sample `eval/datasets/retrieval_eval.sample.jsonl`, baselines under `eval/baselines/`). The harness needs a live corpus/models so it runs locally; only the metric math is unit-tested in CI. Both test suites (`tests/`, `frontend/tests/`) run in CI (`.github/workflows/ci.yml`), which also builds the Docker image.
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
- Reranker default: `jinaai/jina-reranker-v2-base-multilingual` (strongest cross-encoder fastembed supports; `BAAI/bge-reranker-v2-m3` is not loadable by fastembed at any version — owner-approved swap).
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
- OpenAI generation support is optional and user-selectable.
- Reasoning models (gpt-5 family, o-series) spend hidden reasoning tokens from the same `max_completion_tokens` budget as the visible answer, so `LLM_MAX_TOKENS=512` alone leaves an empty answer. `api/generation.py` detects them via `litellm.supports_reasoning`, sends `OPENAI_REASONING_EFFORT` (default `low`), and adds `REASONING_TOKEN_HEADROOM` (default 3072) on top of `LLM_MAX_TOKENS` — both scoped to reasoning models only; local models keep the plain budget.
- Local embeddings remain the default even when OpenAI generation is selected.

## Common Commands

Backend (`uv`-managed, from repo root):

```bash
uv sync --extra dev            # install backend + test + lint + typecheck deps
uv run pytest                  # run backend test suite
uv run pytest tests/test_retrieval.py::test_name  # run a single test
uv run ruff check .            # lint
uv run mypy api                # typecheck (strict mode, api/ only)
uv run uvicorn api.main:app --reload   # run API alone against localhost Qdrant/Ollama
```

Frontend (`frontend/`, npm-managed):

```bash
npm ci                         # install deps (matches CI)
npm run dev                    # Vite dev server, proxies API calls to localhost:8000
npm run test                   # vitest run (all tests)
npx vitest run tests/App.test.tsx   # run a single test file
npm run build                  # tsc -b && vite build -> frontend/dist
npm run lint                   # oxlint
```

Full stack:

```bash
docker compose up              # Qdrant + API (API builds and serves frontend/dist)
```

CI (`.github/workflows/ci.yml`) runs three jobs: backend (`uv sync --extra dev` → `pytest` → `ruff check .` → `mypy api` → import the `eval` extra → validate the sample eval dataset), frontend (`npm ci` → `npm run test` → `npm run build` → `npm run lint`), and a build-only Docker image job.

Optional eval extra: `uv sync --extra eval` then `uv run python -m eval.ragas_runner path/to/dataset.json --output results.json`.

## Architecture

**Request flow (question answering).** `api/main.py` routes (`/questions`, `/questions/stream`) delegate everything to `RagPipeline.answer` / `answer_stream` in `api/pipeline.py`. The pipeline is a thin orchestration seam over four collaborators injected via `api/dependencies.py` (`VectorRepository`, an embedding provider, a reranker, and a `ChatGenerator`) plus `AppSettings`:

1. `api/retrieval.py` embeds the query and calls `VectorRepository.hybrid_search` (`api/repository.py`), which prefers Qdrant's server-side hybrid query and falls back to fetching dense + sparse separately and fusing with RRF in application code when the installed Qdrant doesn't support the needed query shape.
2. `api/reranking.py` cross-encoder-reranks the fused candidates and drops anything below `rerank_min_score`.
3. `expand_with_neighbors` (in `pipeline.py`) widens each surviving chunk's *generation* context using `chunk_ordinal`-adjacent chunks (`context_neighbor_radius`), while citations still point at the original chunk — small-to-big retrieval.
4. `api/generation.py` builds a context-only grounded prompt and calls the configured LiteLLM provider (Ollama or OpenAI), either as one call (`/questions`) or token-streamed as SSE frames (`/questions/stream`, event types `stage`* → `sources` → `delta`* → `done`).

`stage` frames (`condensing`, `searching`) are advisory progress, emitted *before* the stage they name so the UI can show what the server is doing during the seconds before any token exists — retrieval measured 25-80s per question before the tuning below. Clients must treat an unrecognized `stage` value as "show your generic label" (`useChat.ts`'s `stageLabel`), so new stages can be added without breaking an older frontend, and anything asserting on stream event positions must filter them out.

Per-request overrides (`llm_provider`, `rerank_top_k`, `max_context_chunks`, `llm_temperature`, and an optional `filenames` scope) are folded into an immutable `settings.model_copy(...)` per call in `RagPipeline._effective_settings` — the shared `AppSettings` singleton is never mutated, so concurrent requests with different overrides can't interfere. `rrf_k`, `fused_top_n`, and the dense/sparse retrieval limits are deliberately *not* overridable per-request (see Non-Negotiable RAG Behavior above).

**Request flow (ingestion).** `POST /documents` reads the upload (`api/upload.py`, size-capped by `max_upload_bytes`), creates a job in a `JobStore` (`api/jobs.py`) — `IngestJobStore` (in-memory, process-wide, the default) or, when `job_store_backend=sqlite`, `SqliteIngestJobStore` (persists to `job_store_path` so a client polling a job survives a process restart; picked once in `api/dependencies.py:get_ingest_job_store`, the only call site that knows which concrete class is in play) — and hands the actual work to `IngestService.ingest` (`api/pipeline.py`) running on a dedicated `ThreadPoolExecutor` (`get_ingest_executor`, 2 workers) — not FastAPI's request threadpool — so a client disconnect doesn't interrupt ingestion. The client polls `GET /documents/jobs/{job_id}` for state transitions (`queued` → `parsing` → `embedding` → `done`/`failed`). `api/documents.py` parses (PDF/text/Markdown) and chunks with citation metadata preserved; `api/embeddings.py` + `api/ingestion.py` embed chunks (batched by `ingest_batch_size`) with filename/section-prefixed contextual text and upsert them into Qdrant tagged with `chunk_ordinal` and `embedding_model_tag`. `GET /documents` lists indexed filenames for the frontend's per-document search-scope filter (`filenames` on `/questions*`).

**Dependency injection.** `api/dependencies.py` is the single seam for constructing collaborators — most factories are `@lru_cache`d process-wide singletons (settings, Qdrant client, embedding provider, reranker, generator, job store, ingest executor); `get_vector_repository` and the pipeline/service factories are expressed as FastAPI sub-dependencies specifically so tests can override `get_qdrant_client` (or others) and have the override cascade through everything built on top. `clear_dependency_caches()` resets all of it between tests/reloads.

**Settings.** `api/settings.py`'s `AppSettings` (pydantic-settings, reads `.env`) is the single source of runtime config — env var names are the field names upper-cased (see `.env.example`). Changing `dense_embedding_model` requires bumping `embedding_model_tag`; `api/qdrant_schema.py` checks this tag against the stored collection's tag and refuses queries on mismatch (see Non-Negotiable RAG Behavior).

**Frontend.** React + Vite + TS, single-origin with the API in production (`frontend/dist/` is mounted by `api/main.py`'s `StaticFiles`; in dev, Vite proxies to `localhost:8000`). `frontend/src/hooks/useChat.ts` owns chat/session state and talks to the API through `frontend/src/api/client.ts`; components under `frontend/src/components/` are presentational: `ChatThread`/`ChatMessage`/`CitationCard` for the conversation, `CorpusPanel` for upload and ingestion progress, `Composer`'s scope popover for per-query document scoping, `SettingsModal` for provider/retrieval overrides, `Inspector` (with `inspector/SourcesTab`/`RetrievalTab`/`TraceTab`) for per-answer debugging, plus `ConversationList`, `TracesPanel`, `DocumentViewer`, `CommandPalette`, `ChatHeader`, `IconRail`, `ContextPanel` and `ToastRow`. `useDialog` supplies the shared modal behaviour (focus in, Tab trap, focus restore) for `CommandPalette`, `SettingsModal` and `DocumentViewer`.

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
