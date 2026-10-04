# Changelog

Every entry says whether existing documents need attention after upgrading:

- **Re-index**: documents must be re-uploaded or re-indexed to get the change.
- **Recommended**: it applies to new uploads; re-indexing brings old ones in line.
- **None**: it applies to the existing corpus as is.

The version is `version` in `pyproject.toml` (served as the API's OpenAPI version).

## Unreleased

Re-index: **None**. The BM25 IDF modifier is enabled on an existing collection in place
the first time the API touches it — no re-embedding. Tags start empty.

### Retrieval
- BM25 is scored with Qdrant's IDF modifier, so rare terms outweigh common ones (it was
  term frequency only). Existing collections are migrated automatically.
- The reranker scores `filename › section` + text, the same string that is embedded.
- With `OLLAMA_NUM_CTX` or `LLM_CONTEXT_WINDOW` set, the prompt is fitted to the model's
  window (lowest-ranked sources first, then the oldest history); traces mark what was dropped.

### Features
- Answer cache for repeat questions (`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL_SECONDS`).
- Document tags: `PATCH /documents/{filename}/tags`, `tags` on `/questions*`, tag chips in
  the corpus panel and the scope popover.
- `LLM_PROVIDER=openai_compatible` for vLLM, llama.cpp, LM Studio and similar servers.
- Chat: edit a question into a forked conversation, JSON import, conversation search,
  per-conversation scope, copy with citation footnotes, `/` and Esc shortcuts, OS theme
  by default, stopped and cached answers labelled, trace status in the traces panel.

### Fixes
- The app no longer crashes on first load over plain HTTP on a LAN address.
- Browser history: old answers' passage text is no longer saved (it filled the quota),
  conversations are pruned by last use, two tabs merge instead of overwriting, and a
  question in flight survives a reload.
- Bulk re-index and large drops no longer hand back job ids that already 404.
- A queued re-index, or a slower older upload, can no longer overwrite a newer upload.
- `DELETE`/`/content`/`/original` refuse names that were never stored as given.
- Password-protected PDFs are a 400; one broken PDF page no longer fails the document.
- CSV cells with line breaks, and HTML comments, are handled correctly.
- Streaming answers survive proxies (keepalives, no buffering); `/health` answers under load.

### Security and operations
- Unauthorized or oversized requests are rejected before their body is read.
- Provider error messages no longer reach clients verbatim; URLs are shown without credentials.
- Qdrant is published on 127.0.0.1 only; `QDRANT_API_KEY` is supported.
- Feedback rows are capped (`FEEDBACK_MAX_ROWS`).
- Compose: restart policy, API memory cap, a real Qdrant readiness check, `UV_EXTRAS=ocr`.
- CI: `uv sync --locked`, the image is booted, server-only Qdrant tests, eval typechecked.

### Also in this release
- Read-only API key (`API_READ_KEY`); `MAX_CONCURRENT_QUESTIONS` (429 past it); uploads
  retry by themselves when the indexing queue is full.
- `GET /export` backup zip; paged `/documents/{filename}/content` (`around`, `start`/`end`).
- Chat history in IndexedDB; the prompt fitter drops sources before history.
- CJK sentence splitting and cutting of over-long unspaced "words" (re-upload CJK or
  base64-heavy documents to benefit); PDF pages with no drawn lines skip the table finder.
- A Qdrant collection deleted under a running API is recreated instead of failing.
- `GET /documents` is served from a cache between corpus changes.
- The API container is read-only with no capabilities; CI covers Python 3.13 and
  `ruff format`; `drop_reason` is declared an extensible enum.

### Evaluation
- The harness measures the retrieval path the API runs, per stage (`--arm`), against a
  committed fixture corpus and baseline; mismatched baselines are refused.
- RAGAS can judge with a local model (`--llm`, `--embeddings fastembed`); all-NaN runs fail.

## Second review pass (`feat/review-pass-2`)

Re-index: **None**. No chunker or embedding change. New settings default to today's
behaviour except the PDF caps (`MAX_PDF_PAGES=2000`, `MAX_OCR_PAGES=100`).

### Fixes
- A client that disconnects mid-question no longer frees its question slot while the
  retrieval and rerank it started keep running; an abandoned stream stops before the
  rerank or the LLM call.
- An upload queued before a `DELETE` of the same document no longer indexes it back.
- A delete whose stored original can't be removed still invalidates cached answers.
- A filename over 255 bytes is refused at upload; one already indexed no longer makes
  `GET /documents` fail with `RAW_DOCUMENT_DIR` set (Python 3.12).
- Citations written `[1, 2]` or `[1-3]` count as citations (no needless retry call).
- PDFs over `MAX_PDF_PAGES` are refused; OCR stops after `MAX_OCR_PAGES` scanned pages.
- `/content` without paging returns at most 2000 chunks (`total_chunks` stays exact).
- `/export` skips an original deleted while it runs instead of failing.
- Model and Qdrant errors no longer put internal exception text in responses, job
  errors or traces.
- UTF-16/UTF-32 text files (with a BOM) are decoded correctly.
- Frontend: a partly streamed answer whose connection fails is marked incomplete and
  kept out of history; dropping a file outside the corpus panel no longer navigates
  away; no duplicate empty "New chat" after a reload; iOS layout uses `100dvh`; the
  composer no longer pops the keyboard after each answer on touch screens; the
  conversation delete confirm's Cancel button is visible in the light theme.

### Features
- `POST /search`: ranked passages without an answer (works with the LLM down).
- `DELETE /documents/jobs/{id}`: cancel an ingest job before it writes anything; the
  corpus panel's indexing cards have a Cancel button (also stops an upload still waiting
  for room in the queue).
- `POST /import`: restore a `/export` backup (originals, tags, upload times, feedback).
- `GET /access`: which key the caller holds; the UI hides what the read-only key
  can't do.
- `ALLOWED_REQUEST_PROVIDERS`: restrict per-question provider switching.
- `X-Request-ID` on every response, in every log line and in traces; `LOG_FORMAT=json`.
- UI: clickable `[n]` citations, download the original document, restore from the
  palette, a Feedback panel, a tab-title badge (and optional notification) when an
  answer finishes in the background, the answering model recorded per answer, the
  chosen provider remembered.

### Performance
- While streaming, only the markdown block still growing is re-parsed per token.
- The corpus panel and conversation list no longer re-render on every token.
- History writes to IndexedDB are coalesced; uploads go two at a time.
- The markdown renderer is loaded after the first paint, not with the app: the first
  load fetches 280 KB of JavaScript instead of 435 KB (88 KB gzipped instead of 135 KB).
- The chat header no longer re-renders on every token.
- Screen readers hear answer progress from one status line for the thread instead of
  one region per message (and "Answer ready." only for an answer they watched arrive).

### Operations
- Docker: the runtime stage is plain `python:3.12-slim` (no uv); base images pinned
  by digest; compose rotates container logs (10 MB x 3).
- Release workflow: GHCR image with provenance and SBOM, cosign signature, Trivy scan,
  GitHub release from this file, on a `v*` tag.
- CI runs the handbook retrieval eval against its baseline; tests time out after 120 s
  and report coverage.

### Code layout
- `api/main.py` split into `api/routes/*`, `api/ingest_jobs.py`, `api/sse.py` and
  `api/error_handlers.py`; `api/generation.py` into `api/llm.py`, `api/prompting.py`
  and `api/citations.py`; format parsers into `api/parsers/*`. No API change.
- Removed `retrieve_candidates`, `rerank_candidates` and `filename_chunk_counts`, used
  only by tests (which now exercise the production paths); test settings are built by
  `tests/factories.py`.
- Frontend: `useCommands` (palette list and shortcuts), `ConfirmInline` (the three
  inline delete/clear confirms), one metadata record per document in `useCorpus`.

### Evaluation
- `relevant_passages` labels (resolved to chunk ids at run time) and nDCG@k.
- The handbook corpus gained three distractor documents and is scored with small
  chunks, so it is no longer saturated (reranked chunk hit@1 0.95, fused 0.74).

## PR #12 — review follow-ups

Re-index: **Recommended** to use re-index at all: every chunk now records its
`chunker_version`, so documents indexed earlier show as stale.

- Re-index from stored originals (`POST /documents/{filename}/reindex`, bulk variant).
- Narrow-screen drawers, lazy-loaded dialogs, accessibility pass, upload retry/dismiss.
- LLM timeout messages, OCR failure handling, DOCX zip-bomb guard, shutdown handling.

## PR #10 / #11 — latency pass

Re-index: **Recommended** (`CHUNKER_VERSION` 3: token counts exclude special tokens).

- Ollama via `ollama_chat/` (keep-alive now works), int8 jina reranker by default,
  prompt and context trimming, rerank score cache, Playwright E2E in CI.
