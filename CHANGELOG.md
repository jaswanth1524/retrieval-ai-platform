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
