# DocRAG

DocRAG is a self-hostable, open-source document Q&A system. Clone the repository, run it with one command, upload documents in a browser, and ask questions answered only from those documents with citations back to the source.

## Features

- **Hybrid retrieval**: dense semantic search (`BAAI/bge-small-en-v1.5`) and sparse BM25 search (`Qdrant/BM25`, scored with Qdrant's IDF modifier so rare terms outweigh common ones), fused with Reciprocal Rank Fusion (`k=60`), then reranked with a cross-encoder (`jinaai/jina-reranker-v2-base-multilingual`). Queries are embedded symmetrically to the corpus (bge query instruction on the dense side, BM25's query-side encoding on the sparse side), and a post-rerank diversity filter drops near-duplicate chunks so context slots go to distinct passages.
- **Token-aware chunking**: chunks are sized by the dense model's real subword tokens (not word counts) and split on sentence boundaries, so an embedded chunk never silently overflows the model's 512-token limit.
- **Broad format support**: PDF (with table extraction and optional OCR for scanned pages), DOCX, HTML, CSV, TXT, and Markdown. OCR is an optional extra (`uv sync --extra ocr`).
- **Grounded, cited answers**: every answer is generated only from retrieved context and cites filename, page, section, and chunk ID — streamed token-by-token over SSE or returned in full. An answer that comes back without citations is retried once with a stricter reminder. Citation markers (`[2]`, `[1, 2]`, `[1-3]`) in the answer text are buttons that open the cited passage.
- **Search without an answer**: `POST /search` returns the ranked passages for a query — the same hybrid search, RRF and rerank a question uses, with no LLM call — so finding things in your documents works even when the LLM is down. The UI has it as the Search panel in the left rail (and in the ⌘K palette), scoped like a question.
- **Multi-query expansion** (opt-in): rewrite the query into several phrasings, retrieve each, and RRF-fuse before reranking — off by default (`QUERY_EXPANSION_ENABLED`).
- **Conversation memory**: the client sends recent chat turns with each question; when history is present, one extra LLM call rewrites a follow-up ("what about the second one?") into a standalone retrieval query before hybrid search runs. The server itself stores no session state.
- **Multi-conversation chat**: named conversations with new/switch/rename/delete, persisted locally across reloads (older single-thread history migrates automatically).
- **Document management**: upload documents (or drop files anywhere on the page), track background ingestion progress (failed uploads can be retried or dismissed; `DELETE /documents/jobs/{id}` cancels a job before it writes anything), list the full indexed corpus, and delete a document's chunks. With originals kept, a document's uploaded file can be downloaded again from its card or the source viewer.
- **Re-index from stored originals** (opt-in): with `RAW_DOCUMENT_DIR` set, the original upload is kept and a document can be re-chunked and re-embedded without re-uploading it — `POST /documents/{filename}/reindex`, or `POST /documents/reindex` for every document chunked by an older chunker version (the corpus panel flags those and offers a ↻ button).
- **Source viewer**: click a citation to open the source document reconstructed from its chunks, scrolled to and highlighting the cited passage. It fetches a page around the citation (`GET /documents/{filename}/content?around=<chunk_id>&radius=N`, or `start`/`end` ordinals) rather than the whole document, and says so when a re-index has replaced the cited passage.
- **Per-document search scope and tags**: restrict a question to a chosen subset of the indexed corpus, or tag documents (`PATCH /documents/{filename}/tags`, edited inline in the corpus panel — no re-embedding) and scope by tag. Each conversation keeps its own scope.
- **Answer cache**: a repeated question (same scope, settings and corpus) is answered instantly from an in-process cache instead of re-running retrieval and the LLM; any upload, delete or tag change invalidates it, and Regenerate always asks again (`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL_SECONDS`).
- **Context-window fitting**: with `OLLAMA_NUM_CTX` or `LLM_CONTEXT_WINDOW` set, the prompt drops the lowest-ranked sources and then the oldest history until it fits with room for the answer, instead of letting the model server silently cut the grounding rules off its front. The trace records what was left out.
- **Per-query debug traces**: every question's retrieval candidates, rerank keep/drop decisions, the exact LLM prompt, and stage timings are recorded and browsable via `GET /traces`, a per-message drawer, and a trace-history browser in the UI.
- **Chat UX**: markdown-rendered answers (images in answers are never loaded — only their alt text shows, so a poisoned document can't make the browser fetch a URL; links open in a new tab), copy-to-clipboard with the citations as footnotes, retry on failure, edit a question and ask again in a forked conversation, chat export (Markdown/JSON) and JSON import, conversation search (list and ⌘K palette), keyboard shortcuts (⌘K palette, ⌘⇧O new chat, ⌘U upload, ⌘, settings, ⌘J theme, `/` focus the question box, Esc stop the answer), and a responsive layout with a light/dark theme that follows the OS until you pick one — on narrow screens the side panel and the inspector open as drawers. Works over plain HTTP on a LAN address, and several tabs share one history. History lives in IndexedDB (localStorage keeps a compact copy for first paint), so it isn't capped by localStorage's ~5 MB.
- **Evaluation harness**: retrieval-quality metrics (hit@k, recall@k, MRR, nDCG@k) at file and chunk level over a labeled dataset — measured on the same retrieval path the API runs, per stage (`--arm reranked|fused|dense|sparse`) — with a fixture corpus, a committed baseline and a regression gate that refuses to compare mismatched runs, plus an answer-generation mode feeding the RAGAS runner (`eval/harness.py`; RAGAS can judge with a local model via `--llm`). CI runs the handbook set against its baseline on every push.
- **Access and limits**: an optional shared `API_KEY`, plus an optional read-only `API_READ_KEY` for people who should ask questions but not change the corpus (403 on uploads, deletes, re-index, tags, export, import, feedback and metrics; the UI asks `GET /access` and hides those controls). `ALLOWED_REQUEST_PROVIDERS` limits which LLM providers a question may switch to, so a caller can't pick a paid one. Unauthorized or oversized requests are refused before their body is read. At most `MAX_CONCURRENT_QUESTIONS` questions run at once; past that a question gets 429 + `Retry-After`, and uploads past the indexing queue's limit wait and retry on their own.
- **Backup and restore**: `GET /export` (full key; "Download a backup" in the ⌘K palette) returns a zip with a manifest of every document (tags, upload time, chunker version), the stored originals and the feedback rows. `POST /import` ("Restore documents from a backup") re-indexes each stored original with its tags and upload time and adds the feedback back; documents already indexed are kept as they are.
- **Feedback review**: with `FEEDBACK_ENABLED`, a Feedback panel lists answer ratings (filter to thumbs-down) and opens a rated answer's trace.
- **Observability**: configurable application logging (`LOG_LEVEL`; `LOG_FORMAT=json` for one JSON object per line), an `X-Request-ID` on every response that also appears in every log line and in the request's trace, `GET /metrics` (Prometheus), `GET /health` (liveness), `GET /health/ready` (readiness — probes Qdrant and the generation provider).
- **Provider flexibility**: local Ollama by default; any self-hosted OpenAI-compatible server (vLLM, llama.cpp, LM Studio — `LLM_PROVIDER=openai_compatible`) or OpenAI itself are user-selectable alternatives for generation only — embeddings always stay local. Provider errors reach clients as a short reason, never the provider's raw message.

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

1. Optionally condense a follow-up question using client-sent conversation history into a standalone retrieval query (one extra LLM call). Skipped when there's no history, and also when the follow-up is already standalone — it names no anaphora (`it`, `that`, `those`, …) and still has enough substantive words to retrieve on. That call is fully serial ahead of retrieval and measured 2.5-7.3s against a local model, so skipping it for "What is the referral bonus amount?" is pure latency saved; anything ambiguous still condenses.
2. Embed the (condensed) query locally.
3. Hybrid search: dense + sparse retrieval, fused server-side in Qdrant with RRF `k=60` where supported, else fused in application code.
4. Cross-encoder rerank the fused top-N candidates — scored as `filename › section` + text, the same string that was embedded — and drop anything below `RERANK_MIN_SCORE`.
5. Expand each surviving chunk's *generation* context with up to `CONTEXT_NEIGHBOR_RADIUS` neighboring chunks (small-to-big retrieval) — citations still point at the original chunk.
6. Generate an answer through LiteLLM (Ollama or OpenAI) from a context-only, citation-required prompt — streamed over `/questions/stream` (which also emits `stage` progress frames ahead of each phase, since retrieval can run for tens of seconds before the first token) or returned whole from `/questions`.
7. Return the answer, citations, per-stage latency timings, and a trace ID.

**Latency.** Two stages dominate: cross-encoder reranking (linear in `RERANK_CANDIDATES`) and, on a local model, LLM prefill (linear in prompt tokens). Measured on a 7-document, 158-chunk corpus, Apple M1 (Docker 8 CPUs) with `qwen2.5` via Ollama on Metal (~80 tok/s prefill, ~9 tok/s decode):

| Reranker (`RERANKER_MODEL` / `RERANKER_ONNX_FILE`) | 20 candidates | 50 candidates | Peak RSS | Top-6 context vs jina fp32 | Questions left with no context (of 7) |
|---|---|---|---|---|---|
| jina-v2 fp32 (`RERANKER_ONNX_FILE=`) | 11.7s | 31.2s | 2.3 GB | — | 0 |
| jina-v2 int8 (default, `RERANKER_ONNX_FILE=auto`) | 4.6s | 12.7s | 1.4 GB | 81-83% overlap | 0 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 1.7s | 4.3s | 0.8 GB | 43-45% overlap | 2 (incl. the Spanish one) |

Faster rerankers change *which* chunks reach the model, and MiniLM is English-only — pick with `GET /traces/{id}` on your own corpus, not these figures. On the generation side, `CONTEXT_NEIGHBOR_RADIUS=0` cut the median question from 40s to 22s on that M1 by more than halving the prompt, at the cost of narrower context per chunk (it also changes what the diversity filter drops). `GET /traces/{id}` and `docrag_question_stage_seconds{stage=...}` give the per-stage split for your hardware.

Document ingestion (`POST /documents`) runs as a background job: the request returns a `job_id` immediately (HTTP 202), and `GET /documents/jobs/{job_id}` reports progress (`queued` → `parsing` → `embedding` → `done`/`failed`) as the document is parsed, chunked, embedded (with filename/section-prefixed contextual text), and indexed in batches, tagged with an embedding-model version so a later model change is detected and queries are refused until re-ingestion. `GET /documents` lists the indexed corpus; `DELETE /documents/{filename}` removes a document's chunks (and its stored original, when kept).

Qdrant server-side hybrid query is used where available (with `QDRANT_PREFER_GRPC` to skip REST JSON overhead). If the installed Qdrant server or client doesn't support the needed hybrid query shape, DocRAG fetches dense and sparse results separately and performs RRF in application code.

## Repository Structure

```text
.
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
├── pyproject.toml
├── api/            # FastAPI backend
├── frontend/       # React + Vite + TypeScript UI
├── eval/           # retrieval harness, RAGAS runner, fixture corpus, datasets, baselines
└── tests/          # backend test suite
```

## Run it

```bash
docker compose up
```

Needs Docker Compose v2.24+ (Docker Engine 25+): the file uses an optional `env_file` entry and `start_interval` healthchecks. To run a published release instead of building, `DOCRAG_VERSION=0.1.0 DOCRAG_PULL_POLICY=always docker compose up` pulls `ghcr.io/jaswanth1524/retrieval-ai-platform:0.1.0` (linux/amd64 and linux/arm64).

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
> `no-new-privileges`. It writes only to the model cache volume, a tmpfs `/tmp` and, when
> mounted, the `/app/data` volume — mount that volume before turning on
> `JOB_STORE_BACKEND=sqlite`, `RAW_DOCUMENT_DIR` or `FEEDBACK_ENABLED`.
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
uv run mypy api eval           # typecheck (strict mode)
```

Frontend (`frontend/`, npm-managed):

```bash
npm ci                         # install deps (matches CI)
npm run test                   # vitest run (all tests)
npm run build                  # tsc -b && vite build -> frontend/dist
npm run lint                   # oxlint
npx playwright install chromium   # once: browser for the E2E suite
npm run e2e                    # Playwright E2E: prod build + mocked API (desktop + mobile smoke)
npm run e2e:smoke              # @smoke specs only
```

CI (`.github/workflows/ci.yml`) runs five jobs on every push/PR, with a read-only token and superseded runs cancelled:

- **backend** (Python 3.12 and 3.13): `uv sync --locked` (a lockfile that drifted from `pyproject.toml` fails) → pytest (with coverage on 3.12; each test times out after 120 s) → ruff check → ruff format --check → mypy → import the eval extra → validate the eval datasets → `pip-audit` of the runtime dependencies.
- **qdrant**: the server-only tests (`tests/test_qdrant_server.py`: server-side hybrid query, IDF, payload indexes, tags) against a `qdrant/qdrant:v1.19.1` service container — they skip locally unless `DOCRAG_TEST_QDRANT_URL` is set — then the retrieval eval on the handbook set against its committed baseline (models cached between runs).
- **frontend**: vitest → build → lint → `npm audit --omit=dev`.
- **e2e**: Playwright against the production build with a mocked API.
- **docker**: builds the image (layer cache in GitHub Actions) and boots it, probing `/health` and the served UI.

Releases (`.github/workflows/release.yml`): pushing a tag `vX.Y.Z` that matches `version` in `pyproject.toml` publishes `ghcr.io/<owner>/<repo>:X.Y.Z` (linux/amd64 and linux/arm64) with build provenance and an SBOM, signs it with cosign (keyless), scans it with Trivy (a fixable CRITICAL vulnerability stops the release before the push; HIGH findings go to the Security tab) and creates the GitHub release from that version's `CHANGELOG.md` section, which must exist and be non-empty (`tests/test_release_metadata.py` checks it, and that `frontend/package.json` and the Qdrant pins agree). The first push of a package to GHCR may start private: set it to public in the package settings if you want anonymous pulls.

## Configuration

All runtime configuration lives in `api/settings.py` (`AppSettings`, pydantic-settings) and is read from environment variables — see `.env.example` for the full list with defaults and explanatory comments, including hybrid-retrieval tuning (`RRF_K`, `RERANK_TOP_K`, `RERANK_MIN_SCORE`, ...), chunking (`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, ...), conversation memory (`CONVERSATION_CONDENSE_ENABLED`, `CONVERSATION_MAX_HISTORY_MESSAGES`), and provider selection (`LLM_PROVIDER`, `OPENAI_API_KEY`, ...).

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
`.env` on import; run from a directory without a `.env` (or unset what it sets) so the
run uses exactly the settings you exported.

`answer` mode runs the full pipeline and writes rows the RAGAS runner consumes:

```bash
uv run python -m eval.harness answer eval/datasets/retrieval_eval.sample.jsonl --output answers.json
uv run python -m eval.ragas_runner answers.json
```

The metric math and the harness logic (ranking per arm, passage resolution, the
baseline checks) are unit-tested with fakes; the handbook eval runs for real in CI.

> **Upgrading?** The token-aware chunker changes chunk contents versus older word-based
> chunking. The vector space is unchanged, so old and new chunks stay mutually
> queryable and no re-ingest is *required* — but re-uploading documents is recommended
> so a corpus is chunked consistently. Re-uploading replaces a document's chunks
> automatically.

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
- **`/app/data`** (the `docrag_job_store` volume, when enabled): the sqlite job store,
  the feedback database, and stored originals (`RAW_DOCUMENT_DIR`). Back it up with the
  volume, e.g. `docker run --rm -v <project>_docrag_job_store:/data -v "$PWD":/backup
  busybox tar czf /backup/docrag-data.tgz -C /data .`.

**After changing the embedding model** (`DENSE_EMBEDDING_MODEL`/`EMBEDDING_MODEL_TAG`, or the
sparse model) the existing index no longer matches: questions, search and uploads answer
409 with the way out, while listing, exporting, tagging and deleting documents keep
working and `/health/ready` reports `index_compatible: false`. Export a backup, delete
every document, then restore it — the emptied collection is rebuilt for the new model on
the first restore or upload. Without stored originals, delete everything and upload the
files again.

To upgrade: back up both, pull, then `docker compose up -d --build`. Bump `QDRANT_IMAGE`
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
- `api/documents.py`, `api/parsers/`, `api/chunking.py`: the chunk model and a suffix → parser table; one parser module per format (PDF, DOCX, HTML, CSV, text/Markdown); token-aware sentence windowing.
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
- `api/main.py`: the FastAPI app (lifespan, warmup, middleware); the routes live in `api/routes/` (`ops`, `documents`, `feedback`, `traces`, `questions`, `search`), with `api/ingest_jobs.py` (background ingest jobs and cancellation), `api/sse.py` (stream keepalive and question-slot hand-off), `api/error_handlers.py` / `api/errors.py` (status codes and client-safe messages), `api/export.py` / `api/restore.py` (backup and restore).
- `frontend/`: React + Vite + TypeScript browser UI, built and served by the API in production.
- `eval/harness.py`, `eval/metrics.py`: live retrieval/answer harness and its metric math.
- `eval/ragas_runner.py`, `eval/datasets/`, `eval/corpus/`, `eval/baselines/`: optional RAGAS runner, sample datasets, the fixture corpus they are labeled against, and a committed baseline.
- `CHANGELOG.md`: what changed per release, and whether it needs a re-index.
- `docs/`: design notes; `docs/history/` keeps the earlier audit and task plan for reference.

## License

MIT.
