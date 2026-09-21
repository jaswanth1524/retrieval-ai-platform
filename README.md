# DocRAG

DocRAG is a self-hostable, open-source document Q&A system. Clone the repository, run it with one command, upload documents in a browser, and ask questions answered only from those documents with citations back to the source.

## Features

- **Hybrid retrieval**: dense semantic search (`BAAI/bge-small-en-v1.5`) and sparse BM25 search (`Qdrant/BM25`), fused with Reciprocal Rank Fusion (`k=60`), then reranked with a cross-encoder (`jinaai/jina-reranker-v2-base-multilingual`). Queries are embedded symmetrically to the corpus (bge query instruction on the dense side, BM25's query-side encoding on the sparse side), and a post-rerank diversity filter drops near-duplicate chunks so context slots go to distinct passages.
- **Token-aware chunking**: chunks are sized by the dense model's real subword tokens (not word counts) and split on sentence boundaries, so an embedded chunk never silently overflows the model's 512-token limit.
- **Broad format support**: PDF (with table extraction and optional OCR for scanned pages), DOCX, HTML, CSV, TXT, and Markdown. OCR is an optional extra (`uv sync --extra ocr`).
- **Grounded, cited answers**: every answer is generated only from retrieved context and cites filename, page, section, and chunk ID — streamed token-by-token over SSE or returned in full. An answer that comes back without citations is retried once with a stricter reminder.
- **Multi-query expansion** (opt-in): rewrite the query into several phrasings, retrieve each, and RRF-fuse before reranking — off by default (`QUERY_EXPANSION_ENABLED`).
- **Conversation memory**: the client sends recent chat turns with each question; when history is present, one extra LLM call rewrites a follow-up ("what about the second one?") into a standalone retrieval query before hybrid search runs. The server itself stores no session state.
- **Multi-conversation chat**: named conversations with new/switch/rename/delete, persisted locally across reloads (older single-thread history migrates automatically).
- **Document management**: upload documents, track background ingestion progress, list the full indexed corpus, and delete a document's chunks.
- **Source viewer**: click a citation to open the source document reconstructed from its chunks, scrolled to and highlighting the cited passage.
- **Per-document search scope**: restrict a question to a chosen subset of the indexed corpus.
- **Per-query debug traces**: every question's retrieval candidates, rerank keep/drop decisions, the exact LLM prompt, and stage timings are recorded and browsable via `GET /traces`, a per-message drawer, and a trace-history browser in the UI.
- **Chat UX**: markdown-rendered answers, copy-to-clipboard, retry on failure, chat export (Markdown/JSON), and a responsive layout with a light/dark theme.
- **Evaluation harness**: retrieval-quality metrics (hit@k, recall@k, MRR) over a labeled dataset with a baseline regression gate, plus an answer-generation mode feeding the RAGAS runner (`eval/harness.py`, local-only).
- **Observability**: configurable application logging (`LOG_LEVEL`), `GET /metrics` (Prometheus), `GET /health` (liveness), `GET /health/ready` (readiness — probes Qdrant and the generation provider).
- **Provider flexibility**: local Ollama by default; OpenAI is an optional, user-selectable alternative for generation only — embeddings always stay local.

## Stack

- Runtime: Python 3.12
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
4. Cross-encoder rerank the fused top-N candidates; drop anything below `RERANK_MIN_SCORE`.
5. Expand each surviving chunk's *generation* context with up to `CONTEXT_NEIGHBOR_RADIUS` neighboring chunks (small-to-big retrieval) — citations still point at the original chunk.
6. Generate an answer through LiteLLM (Ollama or OpenAI) from a context-only, citation-required prompt — streamed over `/questions/stream` (which also emits `stage` progress frames ahead of each phase, since retrieval can run for tens of seconds before the first token) or returned whole from `/questions`.
7. Return the answer, citations, per-stage latency timings, and a trace ID.

**Latency.** Cross-encoder reranking dominates: it is linear in `RERANK_CANDIDATES` and on 8 CPU cores in Docker cost ~1.1s per candidate with the default reranker — ~52s of a ~66s question. `RERANK_CANDIDATES` and `RERANKER_MODEL` are the two knobs that matter, and `.env.example` documents both with measured numbers. `GET /traces/{id}` and `docrag_question_stage_seconds{stage=...}` give you the per-stage split for your own corpus; tune from that rather than from these figures.

Document ingestion (`POST /documents`) runs as a background job: the request returns a `job_id` immediately (HTTP 202), and `GET /documents/jobs/{job_id}` reports progress (`queued` → `parsing` → `embedding` → `done`/`failed`) as the document is parsed, chunked, embedded (with filename/section-prefixed contextual text), and indexed in batches, tagged with an embedding-model version so a later model change is detected and queries are refused until re-ingestion. `GET /documents` lists the indexed corpus; `DELETE /documents/{filename}` removes a document's chunks.

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
├── eval/           # optional RAGAS evaluation runner + sample dataset
└── tests/          # backend test suite
```

## Run it

```bash
docker compose up
```

The Compose file runs two services: Qdrant, and the API. The API's Docker image builds the `frontend/` React app and serves the built static files itself, so the browser UI and the JSON API share a single origin/port — there is no separate UI container. It loads non-secret defaults from `.env.example`; copy it to `.env` (`cp .env.example .env`) to set real secrets such as `OPENAI_API_KEY` — values in `.env` override `.env.example` and the file is git-ignored.

The default `OLLAMA_BASE_URL` targets `host.docker.internal`, which Docker Desktop (macOS/Windows) resolves automatically. On native Linux, Compose maps this via `extra_hosts: host-gateway`, so no extra setup is required there either. Ollama itself must be running on the host (`ollama serve`) with the configured model pulled (default `llama3.1:8b`).

For local iteration without a full container rebuild:

```bash
uv sync --extra dev
uv run uvicorn api.main:app --reload      # API alone, against localhost Qdrant/Ollama
npm --prefix frontend run dev             # Vite dev server, proxies API calls to :8000
```

> **Note:** the default reranker (`jinaai/jina-reranker-v2-base-multilingual`) is
> ~1.1 GB and downloads once on first use (or at boot if `WARMUP_MODELS=true`) —
> the first question after a fresh install pays that one-time download cost.
>
> It is also the memory floor: answering a question peaks around **2.7 GB** in the API
> container, so give Docker at least **4 GB** (Docker Desktop → Settings → Resources).
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
uv run mypy api                # typecheck (strict mode, api/ only)
```

Frontend (`frontend/`, npm-managed):

```bash
npm ci                         # install deps (matches CI)
npm run test                   # vitest run (all tests)
npm run build                  # tsc -b && vite build -> frontend/dist
npm run lint                   # oxlint
```

CI (`.github/workflows/ci.yml`) runs three independent jobs on every push/PR: backend (pytest → ruff → mypy → eval-dataset validation), frontend (vitest → build → lint), and a Docker build smoke check.

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

`--metrics` accepts a comma-separated list to override the default four metrics.
`--validate-only` loads and shape-checks a dataset without running RAGAS (no LLM
calls, no `ragas`/`datasets` import) — this is what CI runs on every push to keep
the shipped sample dataset honest.

### Live retrieval harness (local only)

`eval/harness.py` drives the real pipeline against a labeled dataset. Unlike the
RAGAS runner it needs a running Qdrant with an ingested corpus (and, for `answer`
mode, an LLM), so it runs locally, not in CI. A labeled dataset row adds relevance
labels to the RAGAS shape:

```json
{"question": "How does retrieval work?", "reference": "...", "relevant_filenames": ["hybrid.md"], "relevant_chunk_ids": []}
```

Score retrieval (hit@k, recall@k, MRR — embeddings + rerank only, no LLM), optionally
gating against a committed baseline:

```bash
uv run python -m eval.harness retrieval eval/datasets/retrieval_eval.sample.jsonl \
  --output eval/baselines/retrieval_baseline.json          # write a baseline
uv run python -m eval.harness retrieval eval/datasets/retrieval_eval.sample.jsonl \
  --baseline eval/baselines/retrieval_baseline.json --max-regression 0.05   # gate
```

`answer` mode runs the full pipeline and writes rows the RAGAS runner consumes:

```bash
uv run python -m eval.harness answer eval/datasets/retrieval_eval.sample.jsonl --output answers.json
uv run python -m eval.ragas_runner answers.json
```

The pure metric math (`eval/metrics.py`) is unit-tested in CI; the harness itself is
local because it needs models, a corpus, and (for `answer`) an LLM.

> **Upgrading?** The token-aware chunker changes chunk contents versus older word-based
> chunking. The vector space is unchanged, so old and new chunks stay mutually
> queryable and no re-ingest is *required* — but re-uploading documents is recommended
> so a corpus is chunked consistently. Re-uploading replaces a document's chunks
> automatically.

## Current Files

- `CLAUDE.md`: operating instructions for future agent work on this repo.
- `.env.example`: non-secret configuration defaults, documented inline.
- `pyproject.toml`: Python 3.12 dependency and tool configuration.
- `docker-compose.yml`: Qdrant + API services (the API also serves the built frontend).
- `Dockerfile`: multi-stage build — Node stage builds `frontend/`, Python stage runs the API as a non-root user with a container healthcheck.
- `api/settings.py`, `api/qdrant_schema.py`: backend configuration and Qdrant collection schema/versioning helpers.
- `api/documents.py`: document parsing and chunking for PDF, DOCX, HTML, CSV, text, and Markdown.
- `api/embeddings.py`, `api/ingestion.py`: local embedding adapters and Qdrant upsert helpers.
- `api/retrieval.py`, `api/repository.py`: hybrid dense+sparse retrieval, RRF, and the Qdrant repository seam (search, upsert, delete).
- `api/reranking.py`: cross-encoder reranking for fused retrieval candidates.
- `api/generation.py`: grounded prompt construction, conversation-history condensing, and LiteLLM generation/streaming adapter for Ollama/OpenAI.
- `api/pipeline.py`: orchestrates retrieve → rerank → generate and parse → chunk → embed → index behind single calls.
- `api/tracing.py`: in-memory per-query debug trace store.
- `api/jobs.py`: background job tracking for document ingestion — in-memory by default, or sqlite-backed (surviving a restart) when `JOB_STORE_BACKEND=sqlite`.
- `api/provider_health.py`: reachability probes for Qdrant and Ollama, used by `/health/ready` and `/config`.
- `api/metrics.py`: Prometheus counters/histograms for question and ingest latency (`GET /metrics`).
- `api/main.py`: the FastAPI app — health/readiness, config, document upload/listing/deletion/job-status, trace, and question-answering (including streaming) endpoints.
- `frontend/`: React + Vite + TypeScript browser UI, built and served by the API in production.
- `eval/ragas_runner.py`, `eval/datasets/`: optional RAGAS evaluation runner and a runnable sample dataset.

## License

MIT.
