# DocRAG

DocRAG is a planned self-hostable, open-source document Q&A system. A newcomer should be able to clone the repository, run the app with one command, upload documents in a browser, and ask questions answered only from those documents with citations back to the source.

This repository is currently in the scaffold phase. Application code has not been implemented yet.

## Goals

- Run locally and self-host without requiring hosted LLM APIs.
- Keep hosted providers optional and user-selectable.
- Upload documents through a browser.
- Ask questions over uploaded documents.
- Answer only from retrieved document context.
- Cite every answer with source metadata.

## Planned Stack

- Runtime: Python 3.12
- API: FastAPI and Uvicorn
- UI: React + Vite + TypeScript (served by the API as static files, single origin)
- Vector store: Qdrant in Docker
- Dependency workflow: `uv`
- Embeddings, sparse retrieval, and reranking: FastEmbed
- LLM access: LiteLLM
- Evaluation: RAGAS
- Testing: Pytest and FastAPI TestClient

## Planned Architecture

DocRAG will use a hybrid RAG pipeline:

1. Parse and chunk uploaded documents.
2. Preserve citation metadata on every chunk:
   - filename
   - page
   - section
   - chunk_id
3. Generate local dense embeddings with `BAAI/bge-small-en-v1.5`.
4. Generate local sparse vectors with `Qdrant/BM25`.
5. Store named dense and sparse vectors in Qdrant.
6. Retrieve with both semantic and keyword search.
7. Fuse retrieval results with Reciprocal Rank Fusion using `k=60`.
8. Rerank only the fused top-N candidates with `jinaai/jina-reranker-v2-base-multilingual`, then drop any candidate scoring below `RERANK_MIN_SCORE` (default 0.30) before it ever reaches the LLM context or citations.
9. Generate an answer through LiteLLM using only the provided context.
10. Return the answer with citations.

Qdrant server-side hybrid query will be used where available. If the installed Qdrant server or client does not support the needed hybrid query shape, DocRAG will fetch dense and sparse results separately and perform RRF in application code.

## Provider Model

Generation and embeddings are independent configuration choices.

- Default generation provider: Ollama
- Default generation model: `llama3.1:8b`
- Optional generation provider: OpenAI
- Default embedding provider: local FastEmbed
- Default dense embedding model: `BAAI/bge-small-en-v1.5`
- Default sparse model: `Qdrant/BM25`

Embeddings stay local by default regardless of the selected LLM provider. No hosted API key should be required to upload or index documents.

The embedding model defines the vector space. DocRAG will store an embedding model version tag with the collection. If the configured embedding tag does not match the collection tag, queries must be refused and the user must be told to re-ingest documents.

## Planned Repository Structure

```text
.
├── docker-compose.yml
├── .env.example
├── README.md
├── pyproject.toml
├── api/
├── frontend/
├── eval/
└── tests/
```

The current milestone creates the project scaffold only. The application directories are placeholders until the API, UI, evaluation, and test milestones are approved and implemented.

## Planned One-Command Run

The intended newcomer workflow is:

```bash
docker compose up
```

The Compose file runs two services: Qdrant, and the API. The API's Docker image builds the `frontend/` React app and serves the built static files itself, so the browser UI and the JSON API share a single origin/port — there is no separate UI container. It loads non-secret defaults from `.env.example`; copy it to `.env` (`cp .env.example .env`) to set real secrets such as `OPENAI_API_KEY` — values in `.env` override `.env.example` and the file is git-ignored.

The default `OLLAMA_BASE_URL` targets `host.docker.internal`, which Docker Desktop (macOS/Windows) resolves automatically. On native Linux, Compose maps this via `extra_hosts: host-gateway`, so no extra setup is required there either.

For local frontend iteration without a full rebuild, run `uv run uvicorn api.main:app --reload` and `npm --prefix frontend run dev` separately — Vite's dev server proxies API calls to `localhost:8000`.

> **Note:** the default reranker (`jinaai/jina-reranker-v2-base-multilingual`) is
> ~1.1 GB and downloads once on first use (or at boot if `WARMUP_MODELS=true`) —
> larger than the previous default, so the first question after a fresh install or a
> reranker config change pays that one-time download cost.

> **Known issue:** generation defaults to local Ollama. If `ollama serve` isn't
> running (or `OLLAMA_BASE_URL` doesn't resolve from wherever the API process runs —
> `host.docker.internal` only resolves *inside* a Docker container; use
> `http://localhost:11434` when running the API directly on your host), questions
> will fail with a 502. `/config`'s `ollama_available` field and the UI's provider
> selector both reflect current reachability.

> **Chunking improved:** `.txt` uploads are now split into paragraph sections (and,
> when the content contains two or more Markdown-style headings, parsed as Markdown)
> instead of becoming a single section for the whole file; Markdown parsing also now
> recognizes setext-style H1 headings (`Title` underlined with `===`). Documents
> ingested before this change keep their old, coarser chunk boundaries — re-upload
> any previously ingested `.txt` files (and `.md` files using setext headings) to
> pick up the improved splitting; re-uploading the same filename automatically
> replaces the old chunks.

> **Chunking improved further:** chunk windows now cross PDF page breaks and
> plain-text paragraph breaks (Markdown heading boundaries are still respected, so
> chunks never mix content from two different headings), and tiny Markdown
> subsections are folded into a neighboring section before chunking. As above,
> re-upload a previously ingested file to replace its chunks with the new,
> better-bounded ones.

## Current Files

- `AGENTS.md`: operating instructions for future agents.
- `TASKS.md`: milestone breakdown and known risks.
- `README.md`: project overview and planned architecture.
- `.gitignore`: ignored local, Python, cache, data, and verification artifacts.
- `.env.example`: non-secret configuration defaults.
- `pyproject.toml`: Python 3.12 dependency and tool configuration.
- `docker-compose.yml`: local services for Qdrant and the API (which also serves the built frontend).
- `Dockerfile`: multi-stage build — Node stage builds `frontend/`, Python stage runs the API and serves the built assets.
- `api/`, `frontend/`, `eval/`, `tests/`: application, browser UI, evaluation, and test directories.
- `api/settings.py` and `api/qdrant_schema.py`: backend configuration and Qdrant collection schema helpers.
- `api/documents.py`: document parsing and chunking helpers for PDF, text, and Markdown source files.
- `api/embeddings.py` and `api/ingestion.py`: local embedding adapters and Qdrant upsert helpers for parsed chunks.
- `api/retrieval.py`: hybrid dense+sparse retrieval with Qdrant RRF and manual RRF fallback.
- `api/reranking.py`: cross-encoder reranking for fused retrieval candidates.
- `api/generation.py`: grounded prompt construction and LiteLLM generation adapter for Ollama/OpenAI.
- `api/main.py`: FastAPI app with health, config, document upload, and question-answering endpoints.
- `frontend/`: React + Vite + TypeScript browser UI (chat-style Q&A with citation cards), built and served by the API in production.
- `eval/ragas_runner.py`: optional RAGAS evaluation runner for JSON/JSONL answer datasets.

## Evaluation (optional)

`eval/ragas_runner.py` scores a set of question/answer/context examples with
[RAGAS](https://github.com/explodinggradients/ragas) metrics (faithfulness, answer
relevancy, context precision, context recall by default). It's independent of the
running API — you supply the question, the answer DocRAG produced, and the context
chunks it was grounded in, as a dataset file.

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
uv run python -m eval.ragas_runner path/to/dataset.json --output results.json
```

`--metrics` accepts a comma-separated list to override the default four metrics.

## License

This project is planned as open source under the MIT license.
