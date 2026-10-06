# Changelog

Every entry says whether existing documents need attention after upgrading:

- **Re-index**: documents must be re-uploaded or re-indexed to get the change.
- **Recommended**: it applies to new uploads; re-indexing brings old ones in line.
- **None**: it applies to the existing corpus as is.

The version is `version` in `pyproject.toml` (served as the API's OpenAPI version).

## Unreleased

## 0.1.0 — 2026-10-05

The first release. It collects everything below: three review passes and the work before
them. If you are upgrading from a development build:

- **Re-index: Recommended.** `CHUNKER_VERSION` is now 4 (section labels are capped,
  Markdown code fences no longer split sections, CSV sections name spreadsheet rows, and
  a long filename can no longer shrink the chunks). Every existing document is listed
  as stale once; with `RAW_DOCUMENT_DIR` set, the corpus panel's "Re-index all" applies it.
  Documents now also record which chunk settings made them, so changing
  `CHUNK_SIZE_TOKENS`, the overlap or the dense model lists them as stale too.
- **Changed the embedding model or `EMBEDDING_MODEL_TAG`?** Listing, exporting, tagging
  and deleting documents keep working, and `/health/ready` reports `index_compatible:
  false`. Questions and uploads are refused until the index matches again: download a
  backup (`GET /export`), delete every document, then `POST /import` it (an emptied
  collection is rebuilt for the new model), or delete everything and upload again.

### Fixes
- Deleting a document no longer removes a newer upload of it that won the race for the
  document's lock (409 instead); a failure removing the stored original after the
  points are gone is logged, not a 500.
- Embedding a document no longer holds the document's lock: tag changes and deletes of
  it are not blocked for minutes behind a large re-index, and a cancel now works
  throughout embedding, not just before its first batch.
- A Markdown heading of any length no longer multiplies a document's chunks (a
  2000-character heading made 4351 chunks); `# comments` inside fenced code blocks are
  not headings; a window overlap larger than the budget no longer advances one sentence
  at a time; CSV sections are labelled with spreadsheet row numbers.
- `/documents` listing: concurrent requests share one scan of the collection, and a
  scan slower than the 30 s TTL still caches.
- A wrong `QDRANT_DENSE_VECTOR_SIZE` is named at boot and at the first upload, not
  reported as "retry the upload"; Qdrant's own error statuses (a rejected key, a 5xx)
  are 503s with a fixed message, and one failed hybrid query no longer switches
  server-side hybrid search off for the process.
- A client that leaves mid-answer now closes the HTTP stream to the model server, so
  Ollama stops generating for nobody.
- The tokenizer is retried after a failed fetch (it stayed on the word heuristic for the
  life of the process); documents record which counter sized them.
- With stored originals, `Report.pdf` and `report.pdf` (or two Unicode forms of one
  name) can no longer share one original: the second upload is a 409.
- Question tags match stored tags regardless of case; request ids reach the worker
  threads of query expansion and embedding; restored feedback keeps its original
  order; a busy `/search` is not counted as a question; two `X-API-Key` headers are a
  400; leftover `.upload-*` temp files are swept; a reranker that can't load names
  `RERANKER_MODEL` instead of the library's error text.
- Frontend: a clicked trace shows that trace (not the latest answer's); Cancel aborts an
  upload still being sent and large uploads get a time budget that grows with their
  size; a drop on the corpus panel during an upload is kept; other tabs' renames,
  deletes and long histories no longer revert, resurrect or truncate each other; a bad
  imported conversation can't crash the app on every load (and there is a recovery
  page if anything does); light-theme text meets 4.5:1 contrast; ⌘ shortcuts no longer
  take Ctrl+K from text fields on a Mac; IME Enter no longer submits; sizes show as
  KB/MB/GB; a rating that failed to send can be sent again; downloads time out
  instead of leaving "Download a backup" disabled; many smaller accessibility and layout
  fixes.
- The Docker image reported version 0.0.0; it now reports the release's.

### Features
- Passage search in the UI (rail panel and palette): the passages a question would be
  answered from, with scores, scoped like questions and working with no LLM reachable.
- Stale documents are detected from the chunk settings they were made with, not only
  the chunker version.

### Operations
- Release: a fixable CRITICAL vulnerability stops the release before anything is
  pushed; images are published for linux/amd64 and linux/arm64; the CHANGELOG must have
  this version's section. Compose can run a released image (`DOCRAG_VERSION`,
  `DOCRAG_PULL_POLICY=always`).
- CI: pushes to main no longer cancel each other; actions are pinned by commit; the
  image is booted through compose with its hardening and a real Qdrant, and must report
  the pyproject version; the OCR image build runs on main; the model cache is keyed on
  what is downloaded; the eval baseline refuses a run made with different ranking
  settings and its results are uploaded.
- Compose waits 60 s for running ingests when stopping.
- Dependabot version-update config removed; dependencies are bumped by hand.

## History before 0.1.0

What the development builds each added, newest first.

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

## Full review pass (after PR #12)
Re-index: **None**. The BM25 IDF modifier is enabled on an existing collection in place
the first time the API touches it — no re-embedding. Tags start empty.

#### Retrieval
- BM25 is scored with Qdrant's IDF modifier, so rare terms outweigh common ones (it was
  term frequency only). Existing collections are migrated automatically.
- The reranker scores `filename › section` + text, the same string that is embedded.
- With `OLLAMA_NUM_CTX` or `LLM_CONTEXT_WINDOW` set, the prompt is fitted to the model's
  window (lowest-ranked sources first, then the oldest history); traces mark what was dropped.

#### Features
- Answer cache for repeat questions (`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL_SECONDS`).
- Document tags: `PATCH /documents/{filename}/tags`, `tags` on `/questions*`, tag chips in
  the corpus panel and the scope popover.
- `LLM_PROVIDER=openai_compatible` for vLLM, llama.cpp, LM Studio and similar servers.
- Chat: edit a question into a forked conversation, JSON import, conversation search,
  per-conversation scope, copy with citation footnotes, `/` and Esc shortcuts, OS theme
  by default, stopped and cached answers labelled, trace status in the traces panel.

#### Fixes
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

#### Security and operations
- Unauthorized or oversized requests are rejected before their body is read.
- Provider error messages no longer reach clients verbatim; URLs are shown without credentials.
- Qdrant is published on 127.0.0.1 only; `QDRANT_API_KEY` is supported.
- Feedback rows are capped (`FEEDBACK_MAX_ROWS`).
- Compose: restart policy, API memory cap, a real Qdrant readiness check, `UV_EXTRAS=ocr`.
- CI: `uv sync --locked`, the image is booted, server-only Qdrant tests, eval typechecked.

#### Also in this release
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

#### Evaluation
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
