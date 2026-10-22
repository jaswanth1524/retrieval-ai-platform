# DocRAG

DocRAG is a self-hostable, open-source document Q&A system. Clone the repository, run it with one command, upload documents in a browser, and ask questions answered only from those documents with citations back to the source.

## Features

- **Hybrid retrieval**: dense semantic search (`BAAI/bge-small-en-v1.5`) and sparse BM25 search (`Qdrant/BM25`, scored with Qdrant's IDF modifier so rare terms outweigh common ones), fused with Reciprocal Rank Fusion (`k=60`), then reranked with a cross-encoder (`jinaai/jina-reranker-v2-base-multilingual`). Queries are embedded symmetrically to the corpus (bge query instruction on the dense side, BM25's query-side encoding on the sparse side), and a post-rerank diversity filter drops near-duplicate chunks so context slots go to distinct passages.
- **Token-aware chunking**: chunks are sized by the dense model's real subword tokens (not word counts) and split on sentence boundaries — table rows are never split, and a table chunk names the rows it holds — so an embedded chunk never silently overflows the model's 512-token limit. A passage that runs onto the next page is cited with both pages (`p.3–4`), and chunk ids depend on the content, not the position, so citations survive a re-index of an edited document. Each upload is parsed in its own process, stopped after `PARSE_TIMEOUT_SECONDS`.
- **Broad format support**: PDF (with table extraction, optional OCR for scanned pages, and running headers, footers and page numbers left out), DOCX, HTML (only the page's content — its `<main>` or article, without navigation, header, footer and sidebars; tables as `column: value` rows; the charset its `<meta>` or the server declares; text the page hides — `hidden`, `aria-hidden`, inline `display:none` — is not indexed), CSV, TXT, Markdown, EPUB (chapters in reading order; the chapter number is the cited page), email `.eml` files (headers and body; attachments are not indexed), Excel `.xlsx` (each sheet is a page, its rows cited by spreadsheet row number; cells hold their last computed values) and PowerPoint `.pptx` (each slide is a page titled by its slide title, with tables and speaker notes). OCR is an optional extra (`uv sync --extra ocr`). `GET /config` lists the accepted types (`supported_extensions`), and the UI follows it.
- **Grounded, cited answers**: every answer is generated only from retrieved context and cites filename, page, section, and chunk ID — streamed token-by-token over SSE or returned in full. An answer that comes back without citations is retried once with a stricter reminder. Each answer also reports `answer_status` — `answered`, `uncited` (it answered but cited none of its sources, even after the retry), `insufficient_context` (the model said the documents don't cover it and cited nothing) or `no_context` (nothing cleared the rerank bar) — the best `top_rerank_score`, and `truncated` when the model stopped at its token limit; the UI marks those answers ("not in your documents", "no sources cited", "cut off"). Citation markers (`[2]`, `[1, 2]`, `[1-3]`) in the answer text are buttons that open the cited passage. Each cited source also says which part of its chunk the answer rests on (`spans`: the quoted sentence or two, matched from the answer without any model call), and the viewer marks it.
- **Whole-document summaries and comparisons**: a question sees the top few retrieved passages, so "summarize this document" got whatever ranked highest. `POST /questions/summary` (and `/stream`) reads every passage of one document (`"mode": "summarize"`) or two (`"mode": "compare"`), or writes one brief across three to `SUMMARY_MAX_DOCUMENTS` (5) of them, organised by theme (`"mode": "brief"`), in order, optionally with a focus (`"instruction": "deadlines"`; the UI asks for one before it starts): a document that fits one prompt is answered in one call, a longer one is turned into cited notes part by part — side by side with a hosted provider (`SUMMARY_PARALLEL_PARTS`), one at a time with Ollama — and the notes combined. Citations are the usual filename/page/section/chunk_id, checked and retried like an answer's. At most `SUMMARY_MAX_BATCHES` parts are read (each is one LLM call); a longer document is read at evenly spaced passages and the answer says how many it read (`chunks_read` of `chunks_total`). In the UI: ∑ on a document's card, Compare when two are selected, Brief for three or more, or the ⌘K palette for a scoped document; Retry and Regenerate repeat the summary.
- **Suggested follow-ups** (opt-in): with `FOLLOWUPS_ENABLED`, an answer that cited its sources gets up to `FOLLOWUP_COUNT` short questions those passages can answer, shown as one-click questions under the latest answer (`followups` on `/questions`, a `followups` frame after `done` on the stream). One extra LLM call per answer, so off by default.
- **Workspaces** (opt-in, `WORKSPACES_ENABLED`): separate corpora on one server — each its own Qdrant collection and originals folder — picked per request with the `X-Workspace` header (the corpus panel's workspace menu, `--workspace` on the CLI). Without the header nothing changes. In the UI a conversation stays with the workspace it was asked in: opening it switches back there. Answer feedback, backups and restores are per workspace too. Only an empty workspace can be deleted. Not access control: every key can use every workspace.
- **Index from a URL** (opt-in, `URL_INGEST_ENABLED`): `POST /documents/from-url` (the corpus panel's URL field, `add-url` on the CLI) has the server fetch a page or file and index it like an upload. Only public addresses are fetched unless `URL_INGEST_ALLOW_PRIVATE` is set; the connection goes to the exact address that was checked (so DNS can't swap in an internal one), redirects are re-checked hop by hop, and `MAX_UPLOAD_BYTES` caps the download.
- **Watched folder** (opt-in, `WATCH_DIR`): new and changed files in a folder are indexed on their own every `WATCH_INTERVAL_SECONDS`, into the default workspace or `WATCH_WORKSPACE`; `WATCH_DELETE_MISSING` also removes the documents of files deleted from it (only ones the watcher saw there). A file whose indexing fails is tried again with backoff; `GET /watch` (and the corpus panel) shows what is retrying, and `POST /watch/scan` runs a pass now.
- **Identical uploads are skipped**: a document re-sent with the same bytes (a re-upload, a `touch`ed watched file, a URL that hasn't changed) is left as indexed — the job finishes `done` with `unchanged: true`, nothing re-embedded — unless it was chunked under older settings. A URL-ingested document keeps its source (`source_urls` on `GET /documents`, a link on its card) and `POST /documents/{filename}/refresh` fetches it again, conditionally: a 304 costs no download and no job.
- **Indexing in progress, from anywhere**: `GET /documents/jobs` lists the jobs queued or running in a workspace — the watcher's, a restore's, another tab's — and the corpus panel shows the ones it didn't start.
- **Command line and MCP**: `docrag` (`uv tool install ./cli`, or `python -m cli` in the repository) ingests folders, asks, searches, summarizes, exports and restores over the REST API; `docrag-mcp` (`./cli[mcp]`, or `python -m cli.mcp_server`) offers search, cited answers, summaries and document reading as tools to MCP clients such as Claude Desktop. See [Command line and MCP](#command-line-and-mcp).
- **Multilingual documents**: an opt-in profile swaps in a multilingual embedding model and BM25 stemming for the corpus's language; the reranker is multilingual already. See [Multilingual documents](#multilingual-documents).
- **Search without an answer**: `POST /search` returns the ranked passages for a query — the same hybrid search, RRF and rerank a question uses, with no LLM call — so finding things in your documents works even when the LLM is down. It also narrows by page range (`page_min`, `page_max`), section label (`section`) and documents left out (`exclude_filenames`). The UI has it as the Search panel in the left rail (and in the ⌘K palette), scoped like a question.
- **Answer language**: `ANSWER_LANGUAGE=match` makes answers and summaries use the language of the question (or name one: `German`); follow-up rewriting keeps the question's language, and recognizes Spanish, German, French, Portuguese and Italian follow-ups.
- **Multi-query expansion** (opt-in): rewrite the query into several phrasings, retrieve each, and RRF-fuse before reranking — off by default (`QUERY_EXPANSION_ENABLED`).
- **Conversation memory**: the client sends recent chat turns with each question; when history is present, one extra LLM call rewrites a follow-up ("what about the second one?") into a standalone retrieval query before hybrid search runs. The server itself stores no session state.
- **Multi-conversation chat**: named conversations with new/switch/rename/delete, persisted locally across reloads (older single-thread history migrates automatically). Up to 10 conversations can be pinned: they are listed first and never dropped by the 100-conversation cap (the UI says when the history fills, and "Export all conversations" in the ⌘K palette saves every one; importing that file restores them). The conversation search matches what was asked and answered, not just titles, and opens a match at the turn it was found in. Other conversations stay usable while an answer streams (it keeps writing to the one it was asked in), and an unsent question is kept per conversation.
- **Document management**: upload documents (or drop files anywhere on the page), track background ingestion progress (failed uploads can be retried or dismissed; `DELETE /documents/jobs/{id}` cancels a job before it writes anything), list the full indexed corpus (sorted, filtered by name or tag), read a document from its card, and delete, tag or re-index several at once. With originals kept, a document's uploaded file can be downloaded again from its card or the source viewer.
- **Re-index from stored originals** (opt-in): with `RAW_DOCUMENT_DIR` set, the original upload is kept and a document can be re-chunked and re-embedded without re-uploading it — `POST /documents/{filename}/reindex`, or `POST /documents/reindex` for every document chunked by an older chunker version (the corpus panel flags those and offers a ↻ button). With `RAW_DOCUMENT_VERSIONS` set too, a re-upload keeps up to that many earlier originals: `GET /documents/{filename}/versions` lists them and `POST /documents/{filename}/versions/{version}/restore` puts one back (`docrag versions`).
- **Query log** (opt-in, `QUERY_LOG_ENABLED`): every question and summary — its text, workspace, outcome and best passage score — kept in a small sqlite file (`QUERY_LOG_PATH`, capped at `QUERY_LOG_MAX_ROWS`), long after the trace buffer has moved on. `GET /query-log?uncovered=true` (full key; `docrag queries --uncovered`) lists what people asked that the documents didn't answer — the gaps worth filling. It holds what users typed, so it is off by default and never part of a backup.
- **Vector quantization** (opt-in, `QDRANT_QUANTIZATION=scalar`): Qdrant keeps int8 copies of the dense vectors in RAM — about a quarter of the memory — and rescores the best `QDRANT_QUANTIZATION_OVERSAMPLING` × limit candidates with the full vectors. Switched on (or off) in place at startup, with no re-index.
- **Source viewer**: click a citation to open the source document reconstructed from its chunks, scrolled to and highlighting the cited passage. It fetches a page around the citation (`GET /documents/{filename}/content?around=<chunk_id>&radius=N`, or `start`/`end` ordinals) rather than the whole document, and says so when a re-index has replaced the cited passage. With originals kept, a PDF can also be shown as the original file, opened at the cited page in the browser's own PDF viewer.
- **Per-document search scope and tags**: restrict a question to a chosen subset of the indexed corpus, or tag documents (`PATCH /documents/{filename}/tags`, edited inline in the corpus panel — no re-embedding) and scope by tag. Each conversation keeps its own scope.
- **Answer cache**: a repeated question (same scope, settings and corpus) is answered instantly from an in-process cache instead of re-running retrieval and the LLM; any upload, delete or tag change invalidates it, and Regenerate always asks again (`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL_SECONDS`).
- **Context-window fitting**: with `OLLAMA_NUM_CTX` or `LLM_CONTEXT_WINDOW` set — or for an OpenAI or Anthropic model LiteLLM knows the window of — the prompt drops the lowest-ranked sources and then the oldest history until it fits with room for the answer, instead of letting the model server silently cut the grounding rules off its front. Chinese, Japanese and Korean text is counted at a token per character. The trace records what was left out.
- **Per-query debug traces**: every question's retrieval candidates, rerank keep/drop decisions, the exact LLM prompt, and stage timings are recorded and browsable via `GET /traces`, a per-message drawer, and a trace-history browser in the UI.
- **Chat UX**: markdown-rendered answers (images in answers are never loaded — only their alt text shows, so a poisoned document can't make the browser fetch a URL; links open in a new tab), copy-to-clipboard with the citations as footnotes, retry on failure (and "Ask again" on a question left without an answer; a question refused because the server is busy is asked again after a countdown), a "weakly supported" flag when the best passage only just cleared the bar, "not relevant" feedback on a single cited source, edit a question and ask again in a forked conversation, chat export (Markdown/JSON) and JSON import, conversation search (list and ⌘K palette), keyboard shortcuts (⌘K palette, ⌘⇧O new chat, ⌘U upload, ⌘, settings, ⌘J theme, ⌘⇧F search passages, ⌘\\ inspector, `/` focus the question box, `]`/`[` next/previous conversation, Esc stop the answer; `?` lists them all; on touch screens Enter starts a new line and the Ask button sends), "Ask about these" on documents to scope the chat to them, and a responsive layout with a light/dark theme that follows the OS until you pick one — on narrow screens the side panel and the inspector open as drawers. Works over plain HTTP on a LAN address (Compose serves loopback only unless `API_BIND=0.0.0.0`; set `API_KEY` too), and several tabs share one history. History lives in IndexedDB (localStorage keeps a compact copy for first paint), so it isn't capped by localStorage's ~5 MB.
- **Evaluation harness**: retrieval-quality metrics (hit@k, recall@k, MRR, nDCG@k) at file and chunk level over a labeled dataset — measured on the same retrieval path the API runs, per stage (`--arm reranked|fused|dense|sparse`) — with a fixture corpus, a committed baseline and a regression gate that refuses to compare mismatched runs, plus an answer-generation mode feeding the RAGAS runner (`eval/harness.py`; RAGAS can judge with a local model via `--llm`). CI runs the handbook set against its baseline on every push.
- **Access and limits**: an optional shared `API_KEY`, plus an optional read-only `API_READ_KEY` for people who should ask questions but not change the corpus (403 on uploads, deletes, re-index, tags, export, import, feedback and metrics; the UI asks `GET /access` and hides those controls). `ALLOWED_REQUEST_PROVIDERS` limits which LLM providers a question may switch to, so a caller can't pick a paid one. Unauthorized or oversized requests are refused before their body is read. At most `MAX_CONCURRENT_QUESTIONS` questions run at once; past that a question gets 429 + `Retry-After`, and uploads past the indexing queue's limit wait and retry on their own. `RATE_LIMIT_PER_MINUTE` (off by default) caps each client's changes and wrong-key attempts per minute.
- **Backup and restore**: `GET /export` (full key; "Download a backup" in the ⌘K palette) returns a zip with a manifest of every document (tags, upload time, chunker version), the stored originals and the feedback rows. `POST /import` ("Restore documents from a backup") re-indexes each stored original with its tags and upload time and adds the feedback back; documents already indexed are kept as they are.
- **Feedback review**: with `FEEDBACK_ENABLED`, a Feedback panel lists answer ratings (filter to thumbs-down) and opens a rated answer's trace.
- **Security headers**: the UI is served with a strict Content-Security-Policy and may not be framed; every response carries `nosniff` and `no-referrer`. Text in a document imitating a prompt source header (`[2] filename=…`) is rewritten, without its number, before it reaches the model. See [SECURITY.md](SECURITY.md).
- **Observability**: configurable application logging (`LOG_LEVEL`; `LOG_FORMAT=json` for one JSON object per line, with one access line per request carrying its id), an `X-Request-ID` on every response that also appears in every log line and in the request's trace, `GET /metrics` (Prometheus: stage latencies, answers by grounding status, truncated answers, citation retries, feedback ratings, LLM tokens reported by the provider, questions running, the indexing backlog, document count, build version), `GET /health` (liveness, with the running version — the UI offers a reload when it changes under an open page), `GET /health/ready` (readiness — probes Qdrant and the generation provider, and reports an index built for another embedding setup, which the UI shows as a banner with the way out). Answers carry the token usage the provider reported (`usage` on the response, the `done` frame and the trace).
- **Provider flexibility**: local Ollama by default; any self-hosted OpenAI-compatible server (vLLM, llama.cpp, LM Studio — `LLM_PROVIDER=openai_compatible`), OpenAI, or Anthropic's Claude (`LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`; `ANTHROPIC_MODEL`, default `claude-sonnet-5-5`) are user-selectable alternatives for generation only — embeddings always stay local, and uploads and indexing never call a hosted API. Provider errors reach clients as a short reason, never the provider's raw message.

## Stack

- Runtime: Python 3.12 (the Docker image; 3.13 is tested in CI too)
- API: FastAPI and Uvicorn
- UI: React + Vite + TypeScript (served by the API as static files, single origin)
- Vector store: Qdrant in Docker
- Dependency workflow: `uv`
- Embeddings, sparse retrieval, and reranking: FastEmbed
- LLM access: LiteLLM
- Evaluation: RAGAS (optional)
- Testing: Pytest + FastAPI TestClient (backend), Vitest + Testing Library (frontend)

## Architecture

The question-answering pipeline (`api/pipeline.py`):

1. Optionally condense a follow-up question using client-sent conversation history into a standalone retrieval query (one extra LLM call). Skipped when there's no history, and also when the follow-up is already standalone — it names no anaphora (`it`, `that`, `those`, …), no continuation word (`more`, `details`, `explain`, `second`, `steps`, …), doesn't open like "and …" or "what about …", and still has enough substantive words to retrieve on. That call is fully serial ahead of retrieval and measured 2.5-7.3s against a local model, so skipping it for "What is the referral bonus amount?" is pure latency saved; anything ambiguous still condenses. The rewrite is cleaned of echoed labels and preambles and dropped when it isn't a question; exchanges answered "not enough information" are left out of the history it reads. The trace's `condense_note` says why a follow-up wasn't rewritten.
2. Embed the (condensed) query locally — and, for a rewritten follow-up, the question as typed too: both are searched and RRF-fused before one rerank against the rewrite.
3. Hybrid search: dense + sparse retrieval, fused server-side in Qdrant with RRF `k=60` where supported, else fused in application code.
4. Cross-encoder rerank the fused top-N candidates — scored as `filename › section` + text, the same string that was embedded — and drop anything below `RERANK_MIN_SCORE`. When nothing clears it, the `RERANK_FALLBACK_COUNT` (3) best-ranked passages are used anyway as *weak matches*: the prompt tells the model to answer from them only if they state the answer, and the answer is flagged `weak_matches` ("weak match" in the UI). Every reranker scores on its own scale, so a fixed floor alone refused questions the documents answer.
5. Expand each surviving chunk's *generation* context with up to `CONTEXT_NEIGHBOR_RADIUS` neighboring chunks (small-to-big retrieval) — citations still point at the original chunk.
6. Generate an answer through LiteLLM (Ollama or OpenAI) from a context-only, citation-required prompt — streamed over `/questions/stream` (which also emits `stage` progress frames ahead of each phase, since retrieval can run for tens of seconds before the first token: `searching`, `waiting_for_model` when another question has the local model, `reading` with the prompt's size and, once measured, how long the model takes to read it) or returned whole from `/questions`. An answer that cites nothing gets its citations added: on a local model by matching its sentences to the passages (`CITATION_RETRY_MODE=auto`, "auto-cited" in the UI), on a hosted one by one more LLM call.
7. Return the answer, citations, per-stage latency timings, and a trace ID.

**Latency.** Two stages dominate: cross-encoder reranking (linear in `RERANK_CANDIDATES`) and, on a local model, LLM prefill (linear in prompt tokens). Measured on a 7-document, 158-chunk corpus, Apple M1 (Docker 8 CPUs) with `qwen2.5` via Ollama on Metal (~80 tok/s prefill, ~9 tok/s decode):

| Reranker (`RERANKER_MODEL` / `RERANKER_ONNX_FILE`) | 20 candidates | 50 candidates | Peak RSS | Top-6 context vs jina fp32 | Questions left with no context (of 7) |
|---|---|---|---|---|---|
| jina-v2 fp32 (`RERANKER_ONNX_FILE=`) | 11.7s | 31.2s | 2.3 GB | — | 0 |
| jina-v2 int8 (default, `RERANKER_ONNX_FILE=auto`) | 4.6s | 12.7s | 1.4 GB | 81-83% overlap | 0 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 1.7s | 4.3s | 0.8 GB | 43-45% overlap | 2 (incl. the Spanish one) |

Faster rerankers change *which* chunks reach the model, and MiniLM is English-only — pick with `GET /traces/{id}` on your own corpus, not these figures. On the generation side, `CONTEXT_NEIGHBOR_RADIUS=0` cut the median question from 40s to 22s on that M1 by more than halving the prompt, at the cost of narrower context per chunk (it also changes what the diversity filter drops). `GET /traces/{id}` and `docrag_question_stage_seconds{stage=...}` give the per-stage split for your hardware — including `first_token` (request to the first streamed token) and `rerank_wait` (time a question queued behind another one's rerank).

**Local models: timeouts and one question at a time.** Ollama sends nothing while it reads the prompt, and LiteLLM gives httpx a single timeout that limits exactly that silence — so with `LLM_REQUEST_TIMEOUT_SECONDS=60` an 8.5k-token prompt (about 80 s to read on an M1) failed before its first token. Each call to a local provider now gets `LLM_REQUEST_TIMEOUT_SECONDS` plus its estimated prompt tokens divided by `LLM_PREFILL_TOKENS_PER_SECOND` (40; or the slower rate measured on this server), and Ollama's non-streamed calls (condense, citation retry, summaries, `/questions`) are streamed and joined, so the timeout bounds a silence rather than the whole answer. The same prompt now answers (first token at 79 s). Ollama also answers one request at a time and queues the rest, with their timeouts running: a second question asked during a long one timed out unread. `OLLAMA_MAX_PARALLEL` (1) keeps the queue in DocRAG instead — the UI says "waiting for the model" — and a question's timeout starts when it is sent; the same two questions both answered (the second at 78 s). Raise it with Ollama's own `OLLAMA_NUM_PARALLEL`. Closing the connection does not stop Ollama reading a prompt (measured: an abandoned 30k-character prompt still held the model a minute later), so a stopped question keeps its place until Ollama lets go of it.

Three opt-in settings make the prompt smaller, all off by default: `PROMPT_TOKEN_BUDGET` (neighbour text goes first, then the lowest-ranked passages, then the oldest history), `RERANK_RELATIVE_FLOOR` (drops passages far below the best one, on any reranker's scale; never causes a refusal) and `GENERATION_HISTORY_ASSISTANT_CHARS` (earlier answers clipped for the answering model). Measure them on your corpus with `make eval-answer` before turning them on.

Measured again for the latency pass, on a different corpus: the 15 handbook documents plus this repository's README, CLAUDE.md and CHANGELOG as distractors (`CHUNK_SIZE_TOKENS=96`, about 450 chunks, 50 candidates reranked), jina int8, `qwen2.5` via Ollama on the same M1, answer and rerank caches off, ten questions each asked once after the corpus was indexed. Single runs on a laptop vary by tens of percent, so read the medians as a direction, not a benchmark:

| Build | First token p50 | Answer done p50 | Rerank p50 | UI bytes on a cold load |
|---|---|---|---|---|
| 0.1.0 | 7.7s | 12.4s | 3.5s | 326 KB |
| This release (default) | 8.1s | 13.9s | 2.9s | 87 KB |
| This release, `CONTEXT_NEIGHBOR_MAX_CHARS=300` | 5.0s | 9.5s | 1.7s | 87 KB |

In isolation, scoring the shortest passages first made a 50-candidate rerank about 22% faster (2.5s vs 3.2s median); the rest of the rerank difference between rows is run-to-run noise (the neighbour cap can't affect reranking). The cap's own effect is on prefill, about 4.3s → 3.2s median, with the same answers and citations on all 11 questions — measured only at these 96-token chunks, so it stays off by default. The UI's JavaScript and CSS are served precompressed.

Document ingestion (`POST /documents`) runs as a background job: the request returns a `job_id` immediately (HTTP 202), and `GET /documents/jobs/{job_id}` reports progress (`queued` → `parsing` → `embedding` → `done`/`failed`) as the document is parsed, chunked, embedded (with filename/section-prefixed contextual text), and indexed in batches, tagged with an embedding-model version so a later model change is detected and queries are refused until re-ingestion. `GET /documents` lists the indexed corpus; `DELETE /documents/{filename}` removes a document's chunks (and its stored original, when kept).

Qdrant server-side hybrid query is used where available (with `QDRANT_PREFER_GRPC` to skip REST JSON overhead). If the installed Qdrant server or client doesn't support the needed hybrid query shape, DocRAG fetches dense and sparse results separately and performs RRF in application code.

## Repository Structure

```text
.
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── SECURITY.md
├── CHANGELOG.md
├── Makefile        # `make check`: the gates CI runs
├── pyproject.toml
├── api/            # FastAPI backend
├── frontend/       # React + Vite + TypeScript UI
├── cli/            # `python -m cli` and the MCP server, over the REST API
├── eval/           # retrieval harness, RAGAS runner, fixture corpus, datasets, baselines
├── tests/          # backend test suite (golden chunking and OpenAPI snapshots in tests/golden/)
├── docs/           # design notes
└── .github/        # CI and release workflows
```

## Run it

```bash
docker compose up
```

Needs Docker Compose v2.24+ (Docker Engine 25+): the file uses an optional `env_file` entry and `start_interval` healthchecks. Once a release is tagged, `DOCRAG_VERSION=<version> DOCRAG_PULL_POLICY=always docker compose up` runs it instead of building (`ghcr.io/jaswanth1524/retrieval-ai-platform:<version>`, linux/amd64 and linux/arm64); until then, `docker compose up --build` builds the image from the checkout.

The UI and API are published on `127.0.0.1:8000` only. To open DocRAG from another device, set `API_BIND=0.0.0.0` in `.env`, and set `API_KEY` too: with no key, anyone who can reach the port can upload, delete and download the whole corpus.

The Compose file runs two services: Qdrant, and the API. The API's Docker image builds the `frontend/` React app and serves the built static files itself, so the browser UI and the JSON API share a single origin/port — there is no separate UI container. It loads non-secret defaults from `.env.example`; copy it to `.env` (`cp .env.example .env`) to set real secrets such as `OPENAI_API_KEY` — values in `.env` override `.env.example` and the file is git-ignored. The hosts in `.env.example` are the compose-network ones (`QDRANT_URL=http://qdrant:6333`, Ollama at `host.docker.internal`); when running the API on the host with `uv run uvicorn`, set them to `localhost` in your `.env`.

The default `OLLAMA_BASE_URL` targets `host.docker.internal`, which Docker Desktop (macOS/Windows) resolves automatically. On native Linux, Compose maps this via `extra_hosts: host-gateway`, so no extra setup is required there either. Ollama itself must be running on the host (`ollama serve`) with the configured model pulled (default `llama3.1:8b`).

For local iteration without a full container rebuild:

```bash
uv sync --extra dev
uv run uvicorn api.main:app --reload      # API alone, against localhost Qdrant/Ollama
npm --prefix frontend run dev             # Vite dev server, proxies API calls to :8000
```

> **Note:** the default reranker (`jinaai/jina-reranker-v2-base-multilingual`, int8
> export) is ~0.28 GB (the fp32 export, `RERANKER_ONNX_FILE=`, is ~1.1 GB) and downloads
> once on first use (or at boot if `WARMUP_MODELS=true`) into the `model_cache` volume,
> so container recreates don't download it again.
>
> It is also the memory floor: reranking peaked around **1.4 GB** with the int8 default
> and **2.3-2.7 GB** with fp32 (depending on the corpus), so give Docker at least **4 GB**
> (Docker Desktop → Settings → Resources). Compose caps the API at `API_MEM_LIMIT`
> (default `4g`) and restarts it if it is ever killed.
>
> Qdrant's ports are published on `127.0.0.1` only. Set `QDRANT_API_KEY` (and uncomment
> `QDRANT__SERVICE__API_KEY` in `docker-compose.yml`) if anything else can reach them.
> For scanned PDFs, build the image with OCR: `UV_EXTRAS=ocr docker compose up --build`.
>
> The API container runs with a read-only root filesystem, no Linux capabilities and
> `no-new-privileges`. It writes only to the model cache volume, a tmpfs `/tmp` and the
> `/app/data` volume (`docrag_data`), where `JOB_STORE_BACKEND=sqlite`,
> `RAW_DOCUMENT_DIR=./data/originals` and `FEEDBACK_ENABLED` keep their files. A store
> that is turned on but can't write its directory stops the API at startup with a
> message naming the setting.
> That peak is reached when the model loads, so a container that boots and answers one
> question has already hit its high-water mark. If you raise `RERANKER_BATCH_SIZE` above
> `RERANK_CANDIDATES`, the reranker scores every candidate in one forward pass instead
> and the peak jumps to ~6.3 GB, which OOM-kills the container (exit 137) — see the
> comment on that setting in `.env.example`.

## Development commands

Backend (`uv`-managed, from repo root):

```bash
uv sync --extra dev            # install backend + test + lint + typecheck deps
uv run pytest                  # run backend test suite
uv run ruff check .            # lint
uv run ruff format .           # format (CI checks it)
uv run mypy api eval cli         # typecheck (strict mode)
make check                     # the backend and frontend gates CI runs, coverage floor and audits included (not e2e)
make golden                    # regenerate the chunking and OpenAPI snapshots, then review the diff
```

Frontend (`frontend/`, npm-managed):

```bash
npm ci                         # install deps (matches CI)
npm run test                   # vitest run (all tests)
npm run build                  # tsc -b && vite build -> frontend/dist
npm run lint                   # oxlint
npm run typecheck:tests        # type-check the vitest suite (vitest itself doesn't)
npx playwright install chromium   # once: browser for the E2E suite
npm run e2e                    # Playwright E2E: prod build + mocked API (desktop + mobile smoke)
npm run e2e:smoke              # @smoke specs only
```

CI (`.github/workflows/ci.yml`) runs seven jobs on every push/PR (dependency-review on pull requests only, docker-ocr everywhere but pull requests), with a read-only token and superseded runs cancelled. It also runs weekly on `main` (Monday 05:23 UTC), so a new advisory against a pinned dependency or base image is noticed without a push, and can be started by hand from the Actions tab.

- **backend** (Python 3.12 and 3.13): `uv sync --locked` (a lockfile that drifted from `pyproject.toml` fails) → pytest (with coverage on 3.12, which must stay at or above 92%; each test times out after 120 s; `tests/test_openapi_snapshot.py` names any breaking API change) → import the eval extra; then, on 3.12, ruff check → ruff format --check → mypy → validate the eval datasets → `pip-audit` of the runtime dependencies (core, the `ocr` extra and the CLI's `mcp` extra).
- **qdrant**: the server-only tests (`tests/test_qdrant_server.py`: server-side hybrid query, IDF, payload indexes, tags) against a `qdrant/qdrant:v1.19.1` service container — they skip locally unless `DOCRAG_TEST_QDRANT_URL` is set — then the retrieval eval on the handbook set against its committed baseline (models cached between runs).
- **frontend**: vitest → build → lint → type-check the tests → `npm audit --omit=dev`.
- **dependency-review** (pull requests): fails on a newly added dependency with a known high-severity vulnerability. It needs the repository's Dependency graph (Settings → Code security); while that is off the job is skipped with a notice.
- **e2e**: Playwright against the production build with a mocked API; a test that passes only on its retry fails the run.
- **docker**: builds the image (layer cache in GitHub Actions), boots it through Compose with the hardened settings and a real Qdrant, checks `/health`, `/health/ready`, the served UI and the reported version, and scans the image with Trivy (a fixable CRITICAL vulnerability fails).

- **docker-ocr**: builds the image with `UV_EXTRAS=ocr`, imports its OCR engine inside it and scans it with Trivy like the default image.

Releases (`.github/workflows/release.yml`): pushing a tag `vX.Y.Z` that matches `version` in `pyproject.toml`, on a commit that is on `main` and whose CI passed (the release waits up to an hour for that run), publishes `ghcr.io/<owner>/<repo>:X.Y.Z` (linux/amd64 and linux/arm64) with build provenance and an SBOM, signs it with cosign (keyless), scans it with Trivy (a fixable CRITICAL vulnerability in either the amd64 or the arm64 image stops the release before the push; HIGH findings go to the Security tab) and creates the GitHub release from that version's `CHANGELOG.md` section, which must exist and be non-empty (`tests/test_release_metadata.py` checks it, and that `frontend/package.json` and the Qdrant pins agree). The first push of a package to GHCR may start private: set it to public in the package settings if you want anonymous pulls.

## Configuration

All runtime configuration lives in `api/settings.py` (`AppSettings`, pydantic-settings) and is read from environment variables — see `.env.example` for the full list with defaults and explanatory comments, including hybrid-retrieval tuning (`RRF_K`, `RERANK_TOP_K`, `RERANK_MIN_SCORE`, ...), chunking (`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, ...), conversation memory (`CONVERSATION_CONDENSE_ENABLED`, `CONVERSATION_MAX_HISTORY_MESSAGES`), and provider selection (`LLM_PROVIDER`, `OPENAI_API_KEY`, ...).

## API reference

The API documents itself: Swagger UI at `/docs`, ReDoc at `/redoc`, the schema at `/openapi.json` (all behind `API_KEY` when one is set). `tests/golden/openapi.json` is the committed copy; a change to the API shows up as a diff there.

From a script (drop the `X-API-Key` header when no key is set):

```bash
KEY="X-API-Key: $API_KEY"
# Upload: 202 with a job id; indexing runs in the background.
job=$(curl -fsS -H "$KEY" -F file=@handbook.pdf http://127.0.0.1:8000/documents | jq -r .job_id)
# Poll until the state is "done" (or "failed", with an error).
curl -fsS -H "$KEY" "http://127.0.0.1:8000/documents/jobs/$job" | jq '{state, error}'
# Ask: the whole answer at once, with its sources and answer_status.
curl -fsS -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"question": "How many days of leave do I get?"}' \
  http://127.0.0.1:8000/questions | jq '{answer, answer_status, sources: [.sources[] | {filename, page}]}'
# Or streamed as Server-Sent Events: stage*, sources, delta*, done.
curl -N -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"question": "How many days of leave do I get?"}' http://127.0.0.1:8000/questions/stream
# Summarize a whole document (or "mode": "compare" with two filenames); /stream also exists.
curl -fsS -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"mode": "summarize", "filenames": ["handbook.pdf"], "instruction": "leave rules"}' \
  http://127.0.0.1:8000/questions/summary | jq '{answer, chunks_read, chunks_total}'
# Passages only, no LLM:
curl -fsS -H "$KEY" -H 'Content-Type: application/json' -d '{"query": "leave policy"}' \
  http://127.0.0.1:8000/search | jq '.results[] | {filename, page, rerank_score}'
```

To scrape the metrics with Prometheus when `API_KEY` is set (`/metrics` needs the full key; `http_headers` needs a recent Prometheus — on an older one, put the header on a proxy in front):

```yaml
scrape_configs:
  - job_name: docrag
    static_configs:
      - targets: ["docrag-host:8000"]
    http_headers:
      X-API-Key:
        secrets: ["<your API_KEY>"]
```

## Command line and MCP

The `docrag` command talks to a running DocRAG over its REST API, so the key, read-only
key, rate limits and question slots apply as they do to the web UI. It is its own small
package (httpx only — none of the server's models or parsers), installed from the
repository on any machine that can reach the server:

```bash
uv tool install ./cli            # or: pipx install ./cli
uv tool install './cli[mcp]'     # also the docrag-mcp server
```

Inside the repository, `uv run python -m cli` is the same command without installing it.
`--url`/`DOCRAG_URL` (default `http://127.0.0.1:8000`), `--api-key`/`DOCRAG_API_KEY` and
`--workspace`/`DOCRAG_WORKSPACE` select the server:

```bash
docrag status
docrag ingest ./handbook ./notes.pdf     # folders recursively; waits for indexing
docrag add-url https://example.com/policy.html
docrag ask "How many days of leave do I get?" --file handbook.pdf [--strict]
docrag chat                              # questions from stdin, with the conversation
docrag search "leave policy"
docrag summarize handbook.pdf [other.pdf ...] --focus "deadlines"   # 2 = compare, 3+ = brief
docrag read handbook.pdf [--around CHUNK_ID]   # a document's passages, or around a citation
docrag tags add handbook.pdf hr          # set / add / remove; --tag on ask and search
docrag delete old.pdf
docrag reindex --stale                   # or one document; needs RAW_DOCUMENT_DIR
docrag refresh policy.html               # a URL-ingested document, fetched again
docrag jobs [--cancel JOB_ID]            # indexing in progress, from anywhere
docrag workspaces list|create NAME|delete NAME
docrag watch [--scan]                    # the watched folder's state
docrag versions handbook.pdf [--restore VERSION]   # needs RAW_DOCUMENT_VERSIONS
docrag queries [--uncovered]             # the query log (QUERY_LOG_ENABLED)
docrag export backup.zip                 # and: import backup.zip
```

`--json` gives machine-readable output; the exit code is 1 when a request (or any upload)
failed, and with `ask --strict` 3 when the documents don't answer the question. `ask`
and `chat` say how an answer was grounded (not found, uncited, auto-cited, weak match)
and, while a local model works, print the stage and the time waited every 15 s. `chat`
asks each question with the conversation so far, leaving out answers that found nothing
(`/new` starts over). Uploads that meet a full indexing queue wait and retry, as the UI does. `ingest`,
`add-url` and `import` wait for indexing for up to `--timeout` seconds (default 3600, 0 =
no limit); anything still indexing then is reported and goes on on the server.

The MCP server offers `list_documents`, `search_passages`, `ask`, `summarize`,
`read_document` and `list_workspaces` to an MCP client, all read-only (the read-only key is
enough). Answers keep their citations and list the exact sentences each one rests on, so
the client can quote them with filename, page and section; `ask` and `search_passages`
take `tags`, `ask` takes the conversation so far as `history` for a follow-up, and
`read_document` reads around a cited chunk (`around_chunk_id`).
`DOCRAG_WORKSPACE` in its environment points it at a workspace. For Claude Desktop,
add to its MCP configuration (with `docrag-mcp` installed as above):

```json
{
  "mcpServers": {
    "docrag": {
      "command": "docrag-mcp",
      "env": {"DOCRAG_URL": "http://127.0.0.1:8000", "DOCRAG_API_KEY": "<your key>"}
    }
  }
}
```

or, without installing anything, from a checkout of the repository:

```json
{
  "mcpServers": {
    "docrag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/retrieval-ai-platform", "--extra", "mcp",
               "python", "-m", "cli.mcp_server"],
      "env": {"DOCRAG_URL": "http://127.0.0.1:8000", "DOCRAG_API_KEY": "<your key>"}
    }
  }
}
```

## Multilingual documents

The defaults are English-first: `BAAI/bge-small-en-v1.5` for dense retrieval and English
stemming for BM25 (the jina reranker is multilingual already). For a corpus in other
languages, switch both — a new vector space, so set a new `EMBEDDING_MODEL_TAG` and
re-index (download a backup, delete every document, restore it):

| Setting | Light (~50 languages) | Stronger (~100 languages, 2.2 GB) |
|---|---|---|
| `DENSE_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `intfloat/multilingual-e5-large` |
| `QDRANT_DENSE_VECTOR_SIZE` | `384` | `1024` |
| `DENSE_QUERY_INSTRUCTION` | empty | `query: ` |
| `DENSE_PASSAGE_PREFIX` | empty | `passage: ` |
| `EMBEDDING_MODEL_TAG` | e.g. `fastembed:paraphrase-multilingual-MiniLM-L12-v2` | e.g. `fastembed:multilingual-e5-large` |

Set `SPARSE_LANGUAGE` to the corpus's main language (`german`, `french`, `spanish`,
`italian`, `dutch`, …) for BM25's stemmer. The collection records the dense model, the
sparse language and the passage prefix, and refuses questions and uploads that would mix
them with another setup. The evaluation baselines are English. To measure a multilingual
setup, `eval/corpus_multilingual/` and `eval/datasets/multilingual_eval.jsonl` are a
small Spanish/German/French starting point (with an archived distractor and English
questions about the non-English documents): `python -m eval.harness ingest
eval/corpus_multilingual`, then `python -m eval.harness retrieval
eval/datasets/multilingual_eval.jsonl`. It has no baseline; measure on your own
documents too before relying on it.

## Watched folder

With `WATCH_DIR` set, the API indexes that folder into the default workspace (or into
`WATCH_WORKSPACE`, with workspaces on; created at startup if it doesn't exist): on start and
then every `WATCH_INTERVAL_SECONDS` (30), new files and files modified since their document
was indexed go through the same queue as uploads (a full queue waits for the next pass).
Subfolders are included, dotfiles and unsupported types skipped; documents are stored by
file name, so two files with the same name are one document (the second is skipped and
logged). `WATCH_DELETE_MISSING=true` also deletes the document of a file removed from the
folder — only documents the watcher saw there, never other uploads. A file the server
refuses (a bad name, a conflict) waits until it changes; one whose read or indexing job
fails is tried again after `WATCH_INTERVAL_SECONDS` × 2ⁿ, up to `WATCH_MAX_RETRIES` times
(`WATCH_RETRY_MAX_SECONDS` at most between tries). `GET /watch` reports the last pass
(files, problems, what is retrying or gave up), `POST /watch/scan` (full key) runs one now,
and the corpus panel shows both. A file only touched is not re-embedded. With Compose, mount
the folder read-only and point `WATCH_DIR` at it (see the commented line in
`docker-compose.yml`).

## Troubleshooting

- **"The generation provider sent nothing for N s" / "did not finish within LLM_REQUEST_TIMEOUT_SECONDS".** A local model reading a long prompt on modest hardware. Raise `LLM_REQUEST_TIMEOUT_SECONDS` or lower `LLM_PREFILL_TOKENS_PER_SECOND` (the error names both); `PROMPT_TOKEN_BUDGET` makes the prompt smaller instead.
- **Follow-ups or reworded questions get "not enough information" (a "weak match").** Retrieval found no close passage. A faster, non-default `RERANKER_MODEL` is the usual cause: on the handbook eval sets `Xenova/ms-marco-MiniLM-L-6-v2` left 3-4 of 10 follow-ups unanswered where the default jina reranker (with `RERANK_CANDIDATES=50`) answered all 10, with no unanswerable question answered by either. `GET /traces/{id}` shows the scores.
- **"Cannot reach Ollama" / answers fail, uploads work.** Generation needs Ollama running on the host (`ollama serve`) with the model pulled (`ollama pull llama3.1:8b`); `OLLAMA_BASE_URL` points the container at `host.docker.internal`. `/health/ready` says which dependency is down. Upload, indexing and passage search never need the LLM.
- **409 "Questions and uploads are refused until the index matches the configuration".** The collection was indexed with a different embedding model or tag, and questions are refused rather than compared across vector spaces. Restore the previous embedding settings, or rebuild: download a backup (⌘K → Download a backup), delete every document, then restore the backup; it re-indexes from the stored originals.
- **The API restarts, exit code 137.** Out of memory: give Docker at least 4 GB, and keep `RERANKER_BATCH_SIZE` at or below `RERANK_CANDIDATES` (see the note under Run it).
- **"API unreachable" in the UI.** The page retries on its own and has a Retry now button; check `docker compose ps` and the API logs. A blank page after an upgrade is a stale tab: reload it.
- **The UI is blank or blocked when served from another origin.** The bundled Content-Security-Policy allows requests to the page's own origin only. A frontend hosted elsewhere (`VITE_API_BASE_URL`) needs its own policy and `CORS_ALLOW_ORIGINS` on the API.
- **The API won't start: "… needs a writable directory".** A store turned on in `.env` (`RAW_DOCUMENT_DIR`, `JOB_STORE_BACKEND=sqlite`, `FEEDBACK_ENABLED`) points outside the `/app/data` volume, where the container can't write. Keep their paths under `./data/` (the defaults).
- **Uploads fail with 413 behind nginx, or answers stop after a minute.** The proxy's limits, not DocRAG's: see [Behind a reverse proxy](#behind-a-reverse-proxy).
- **`docker compose pull` of a release is denied.** A package pushed to GHCR for the first time can be private; make it public in its package settings, or log in to GHCR.

## Evaluation (optional)

`eval/ragas_runner.py` scores a set of question/answer/context examples with
[RAGAS](https://github.com/explodinggradients/ragas) metrics (faithfulness, answer
relevancy, context precision, context recall by default). It's independent of the
running API — you supply the question, the answer DocRAG produced, and the context
chunks it was grounded in, as a dataset file. A sample dataset ships at
`eval/datasets/sample_eval.jsonl` (questions about DocRAG's own architecture) so the
command below is runnable out of the box.

Install the optional extra first:

```bash
uv sync --extra eval
```

Dataset format — a JSON list, a JSON object with an `examples` key, or JSONL (one
object per line), each example shaped like:

```json
{
  "question": "What is the refund window?",
  "answer": "Refunds must be requested within 30 days of purchase.",
  "contexts": ["Refund requests must be submitted within 30 days of purchase..."]
}
```

Run it:

```bash
uv run python -m eval.ragas_runner eval/datasets/sample_eval.jsonl --output results.json
```

RAGAS metrics are themselves LLM-judged. By default RAGAS uses OpenAI and needs
`OPENAI_API_KEY`; to keep scoring local, pass a LiteLLM model and local embeddings:
`--llm ollama_chat/qwen2.5 --embeddings fastembed`. A run in which every judge call
failed exits 1 instead of reporting all-NaN scores as a result.

`--metrics` accepts a comma-separated list to override the default four metrics.
`--validate-only` loads and shape-checks a dataset without running RAGAS (no LLM
calls, no `ragas`/`datasets` import) — this is what CI runs on every push to keep
the shipped sample dataset honest.

### Live retrieval harness

`eval/harness.py` drives the real pipeline against a labeled dataset. Unlike the
RAGAS runner it needs a running Qdrant with an ingested corpus (and, for `answer`
mode, an LLM). A labeled dataset row adds relevance labels to the RAGAS shape:

```json
{"question": "How does retrieval work?", "reference": "...", "relevant_filenames": ["hybrid.md"], "relevant_passages": ["fused with Reciprocal Rank Fusion"]}
```

`relevant_passages` is text the right chunk contains (whitespace- and case-insensitive),
resolved to chunk ids against the indexed corpus when the run starts — so chunk-level
labels survive chunking changes that would change every chunk id. A passage that is in
no chunk fails the run. (`relevant_chunk_ids` still works for fixed ids.)

`eval/corpus/` holds the documents the sample dataset is labeled against. Ingest them
into an empty collection, then score retrieval (hit@k, recall@k, MRR, nDCG@k — no LLM unless
query expansion is on) exactly as the API retrieves (`RagPipeline.retrieve`: rerank,
diversity filter, neighbour expansion), optionally gating against the committed baseline:

```bash
export QDRANT_COLLECTION=docrag_eval LITELLM_MODE=PRODUCTION   # see the note below
uv run python -m eval.harness ingest eval/corpus
uv run python -m eval.harness retrieval eval/datasets/retrieval_eval.sample.jsonl \
  --baseline eval/baselines/sample_reranked.json --max-regression 0.05   # gate
uv run python -m eval.harness retrieval eval/datasets/retrieval_eval.sample.jsonl \
  --arm sparse                                   # score one stage: fused, dense, sparse
```

A baseline records what it was measured with (arm, models, chunker version); the gate
refuses a baseline with other cutoffs, another question count or another arm instead of
passing on no evidence, and notes any setting that changed. A second, harder set lives in
`eval/corpus_handbook/` (15 policy documents sharing boilerplate, five of them
distractors — including an archived policy and a contractor handbook with conflicting
numbers) with `eval/datasets/handbook_eval.jsonl` (19 questions, passage-labeled) and
`eval/baselines/handbook_reranked.json`. It is scored with small chunks so each section
is its own chunk, which is also what CI runs:

```bash
export QDRANT_COLLECTION=eval_handbook LITELLM_MODE=PRODUCTION \
  CHUNK_SIZE_TOKENS=48 CHUNK_OVERLAP_TOKENS=8 MIN_SECTION_WORDS=0
uv run python -m eval.harness ingest eval/corpus_handbook
uv run python -m eval.harness retrieval eval/datasets/handbook_eval.jsonl \
  --baseline eval/baselines/handbook_reranked.json
```

Measured with the default models on macOS arm64, chunk hit@1 is 0.95 after rerank
against 0.74 for the fused list, 0.68 dense-only and 0.79 sparse-only. The committed
baseline is from linux/amd64 (CI), where the int8 reranker orders one borderline question
differently (0.89); CI allows a drop of 0.06, i.e. one question. The small sample set still scores 1.0,
so it only catches breakage. `LITELLM_MODE=PRODUCTION` stops LiteLLM loading the repo's
`.env` on import (and `LITELLM_LOCAL_MODEL_COST_MAP=True`, set in the Docker image and the tests, stops it fetching its model price list from GitHub); run from a directory without a `.env` (or unset what it sets) so the
run uses exactly the settings you exported.

`answer` mode answers each question the way the API does (`RagPipeline.answer`: a
follow-up's `history` is condensed, retrieval, prompt fitting, generation, citations and
status) and writes rows the RAGAS runner consumes. Each row is saved as it finishes —
at a minute per answer on a local model a run is long — so `--resume` picks an
interrupted run up, and `--limit N` answers only the first N:

```bash
uv run python -m eval.harness answer eval/datasets/retrieval_eval.sample.jsonl --output answers.json
uv run python -m eval.ragas_runner answers.json
make eval-answer EVAL_DATASET=eval/datasets/followup_eval.jsonl EVAL_ARGS=--resume
make eval-retrieval      # handbook corpus vs its baseline, as CI runs it
make eval-ragas LLM=ollama_chat/qwen2.5
```

The Makefile targets index the handbook corpus into its own `eval_handbook` collection
with the settings its baseline was measured with (`EVAL_QDRANT_URL`, default
`http://127.0.0.1:6333`). A row's `answer_contains` (any of a few ways of stating the fact:
`["sixty euros", "60 euros"]`) scores `fact_hit_rate` without a judge model.

`answer` mode also prints how the answers went: `cited_relevant_rate` (an answer cited a
labeled document), `answered_rate`, `uncited_rate`, `fact_hit_rate`, `error_rate` (a
question the pipeline failed on is kept as an error row, not the end of the run) and, for rows marked
`expect_no_context`, `refused_unanswerable_rate` (the answer said the documents don't
cover it). A row can carry a `history` (the earlier turns, as the UI sends them); it is
scored on the follow-up as the API rewrites it. `eval/datasets/followup_eval.jsonl` and
`unanswerable_eval.jsonl` are such sets for the handbook corpus.

With `FEEDBACK_ENABLED`, the answer ratings people give can seed a dataset of your own
questions. `from-feedback` reads the feedback store (or a backup zip) and writes each
thumbs-up answer's question labeled with the documents it cited; thumbs-down questions go
to a review file, to label by hand, since a bad rating says the answer was wrong but not
which documents were right. Only the latest rating of a question counts, and ratings of a
single citation are skipped. The output is validated like `validate` before it is kept:

```bash
uv run python -m eval.harness from-feedback --store data/feedback.db \
  --output eval/datasets/from_feedback.jsonl --review review.jsonl
# or from a backup: --export docrag-backup.zip
```

The metric math and the harness logic (ranking per arm, passage resolution, the
baseline checks) are unit-tested with fakes; the handbook eval runs for real in CI.

> **Upgrading?** The token-aware chunker changes chunk contents versus older word-based
> chunking. The vector space is unchanged, so old and new chunks stay mutually
> queryable and no re-ingest is *required* — but re-uploading documents is recommended
> so a corpus is chunked consistently. Re-uploading replaces a document's chunks
> automatically.

## Behind a reverse proxy

DocRAG speaks plain HTTP; for anything beyond your own machine put it behind a proxy that
terminates TLS (the API key travels in a header). What the proxy has to allow:

- **Upload size**: `MAX_UPLOAD_BYTES` (50 MB) per document and `MAX_IMPORT_BYTES` (256 MB)
  for a backup restore. nginx's default `client_max_body_size` is 1 MB, which turns
  every larger upload into a 413.
- **Streaming**: answers stream as Server-Sent Events, with a `: ping` every 15 s while
  the server works. The proxy must not buffer the response (the API sends
  `X-Accel-Buffering: no` for nginx) and must allow a read timeout above 15 s.
- **The real client address**: set `FORWARDED_ALLOW_IPS` in `.env` to the proxy's address
  as the API container sees it (uvicorn reads it). Without it every request appears to
  come from the proxy: all users share one `RATE_LIMIT_PER_MINUTE` bucket and logs show
  the proxy.
- **A sub-path** (`https://host/docrag/`): strip the prefix in the proxy, or run uvicorn
  with `--root-path /docrag` (the API's checks apply to the path without it).

nginx:

```nginx
server {
    listen 443 ssl;
    server_name docs.example.com;
    # ssl_certificate / ssl_certificate_key ...
    client_max_body_size 300m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

Caddy (TLS certificates are automatic; streaming and large bodies work by default):

```caddy
docs.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

## Backup and upgrade

The simplest backup is the app's own: `GET /export` (or "Download a backup" in the ⌘K
palette) and, to restore, `POST /import` with that zip ("Restore documents from a
backup"). It re-indexes from the stored originals, so it needs `RAW_DOCUMENT_DIR` on
when the backup was made. A restore skips
documents already indexed, and one deferred because the indexing queue was full is
finished by importing the same file again. For a byte-level backup, everything DocRAG
knows lives in two places:

- **Qdrant** (the `qdrant_storage` volume): the chunks, vectors and tags. Take a snapshot
  with `curl -X POST http://127.0.0.1:6333/collections/docrag_documents/snapshots` (add
  `-H "api-key: $QDRANT_API_KEY"` when set); it is written under `/qdrant/snapshots` in
  the container — copy it out with `docker compose cp qdrant:/qdrant/snapshots ./backup`.
  Restore by uploading it to `/collections/docrag_documents/snapshots/upload`.
- **`/app/data`** (the `docrag_data` volume): the sqlite job store, the feedback
  database, and stored originals (`RAW_DOCUMENT_DIR`). Back it up with the volume, e.g.
  `docker run --rm -v <project>_docrag_data:/data -v "$PWD":/backup busybox tar czf
  /backup/docrag-data.tgz -C /data .`.

**After changing the embedding model** (`DENSE_EMBEDDING_MODEL`/`EMBEDDING_MODEL_TAG`, or the
sparse model) the existing index no longer matches: questions, search and uploads answer
409 with the way out, while listing, exporting, tagging and deleting documents keep
working and `/health/ready` reports `index_compatible: false`. Export a backup, delete
every document, then restore it — the emptied collection is rebuilt for the new model on
the first restore or upload. Without stored originals, delete everything and upload the
files again.

To upgrade: back up both, then pull the new source and `docker compose up -d --build`
— or, running a published image, set `DOCRAG_VERSION` to the new release and
`docker compose pull && docker compose up -d`. Bump `QDRANT_IMAGE`
on its own, after a snapshot — Qdrant reads older storage but not newer. Check
`CHANGELOG.md` for releases that need a re-index (with originals kept, the corpus panel
offers ↻ on documents chunked by an older chunker). The IDF modifier is applied to an
existing collection automatically, with no re-ingest.

## Current Files

- `CLAUDE.md`: operating instructions for future agent work on this repo.
- `.env.example`: non-secret configuration defaults, documented inline.
- `pyproject.toml`: Python 3.12 dependency and tool configuration.
- `docker-compose.yml`: Qdrant + API services (the API also serves the built frontend).
- `Dockerfile`: multi-stage build — Node builds `frontend/`, uv builds the Python venv, and a plain `python:3.12-slim` stage runs the API as a non-root user with a container healthcheck (base images pinned by digest).
- `api/settings.py`, `api/qdrant_schema.py`: backend configuration and Qdrant collection schema/versioning helpers.
- `api/documents.py`, `api/parsers/`, `api/chunking.py`: the chunk model and a suffix → parser table; one parser module per format (PDF, DOCX, HTML, CSV, text/Markdown, EPUB, email, XLSX, PPTX); token-aware sentence windowing.
- `api/parse_isolation.py`: parses each upload in a child process with a time limit.
- `api/summarize.py`: whole-document summaries and comparisons.
- `api/url_fetch.py`: the SSRF-guarded URL fetcher behind `POST /documents/from-url` and refresh.
- `api/watch.py`: the watched-folder poller (`WATCH_DIR`), its retries and status.
- `api/workspaces.py`: workspace names and the per-request settings they select.
- `api/admission.py`: the question (and summary) concurrency caps.
- `api/query_log.py`: the opt-in query log (`QUERY_LOG_ENABLED`).
- `api/corpus.py`: per-collection change counters that invalidate cached answers and listings.
- `api/model_files.py`: keeps a downloaded model's files beside it for ONNX Runtime.
- `cli/`: the `docrag` command and the `docrag-mcp` server (their own package, `cli/pyproject.toml`).
- `api/static_files.py`: serves the built UI — the precompressed `.br`/`.gz` copies the build writes, a year of caching for content-hashed assets, revalidation for `index.html`, and the UI's Content-Security-Policy.
- `api/security_headers.py`: `nosniff` and `no-referrer` on every response, and the policy string the UI is served with.
- `api/rate_limit.py`: the per-client token bucket behind `RATE_LIMIT_PER_MINUTE`.
- `api/upload.py`, `api/raw_documents.py`: size-capped upload reads, and the opt-in store of original uploads.
- `api/request_guard.py`: rejects unauthorized or oversized requests before their body is read.
- `api/request_id.py`: the `X-Request-ID` middleware and the log filter that stamps it.
- `api/answer_cache.py`: in-process cache of finished answers, invalidated by corpus changes.
- `api/diversity.py`: the post-rerank near-duplicate filter.
- `api/feedback.py`, `api/sqlite_store.py`: optional answer-feedback store and shared sqlite helpers.
- `api/dependencies.py`, `api/logging_config.py`: dependency-injection seam and logging setup.
- `api/embeddings.py`, `api/ingestion.py`: local embedding adapters and Qdrant upsert helpers.
- `api/retrieval.py`, `api/repository.py`: hybrid dense+sparse retrieval, RRF, and the Qdrant repository seam (search, upsert, delete).
- `api/reranking.py`: cross-encoder reranking for fused retrieval candidates.
- `api/generation.py`, `api/llm.py`, `api/prompting.py`, `api/citations.py`: grounded answers; the LiteLLM client and provider errors; the prompt, context-window fitting, condensing and query expansion; citation parsing and the citation retry.
- `api/pipeline.py`: orchestrates retrieve → rerank → generate and parse → chunk → embed → index behind single calls.
- `api/tracing.py`: in-memory per-query debug trace store.
- `api/jobs.py`: background job tracking for document ingestion — in-memory by default, or sqlite-backed (surviving a restart) when `JOB_STORE_BACKEND=sqlite`.
- `api/provider_health.py`: reachability probes for Qdrant and Ollama, used by `/health/ready` and `/config`.
- `api/metrics.py`: Prometheus counters/histograms for question and ingest latency (`GET /metrics`).
- `api/main.py`: the FastAPI app (lifespan, warmup, middleware); the routes live in `api/routes/` (`ops`, `documents`, `feedback`, `traces`, `questions`, `search`, `workspaces`, `watch`), with `api/ingest_jobs.py` (background ingest jobs and cancellation), `api/sse.py` (stream keepalive and question-slot hand-off), `api/error_handlers.py` / `api/errors.py` (status codes and client-safe messages), `api/export.py` / `api/restore.py` (backup and restore).
- `frontend/`: React + Vite + TypeScript browser UI, built and served by the API in production.
- `eval/harness.py`, `eval/metrics.py`: live retrieval/answer harness and its metric math.
- `eval/ragas_runner.py`, `eval/datasets/`, `eval/corpus*/`, `eval/baselines/`: optional RAGAS runner, the datasets (sample, handbook, multilingual, unanswerable questions scored on keeping nothing, and follow-ups scored after rewriting), the fixture corpora they are labeled against, and the committed baselines (sample and handbook).
- `eval/feedback_dataset.py`: turns recorded answer ratings into a labeled dataset (`harness from-feedback`).
- `CHANGELOG.md`: what changed per release, and whether it needs a re-index.
- `SECURITY.md`: how to report a vulnerability, and what DocRAG does and leaves to you.
- `Makefile`: `make check` (the backend and frontend gates CI runs), `make golden`, `make e2e`.
- `docs/`: design notes; `docs/history/` keeps the earlier audit and task plan for reference.

## License

MIT.
