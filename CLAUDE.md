# Agent Instructions

DocRAG is a self-hostable, open-source document Q&A system. Treat the project specification and this file as the operating guide for future work.

## Current State

- The repository is in the scaffold phase.
- Project configuration and placeholder directories may exist.
- Do not implement application feature code until the project owner explicitly approves the relevant implementation milestone.

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
- Local embeddings remain the default even when OpenAI generation is selected.

## Validation Rules

- Follow the repository owner's validation policy before presenting changes as complete.
- For backend/API/service/data changes, use the installed backend verification workflow when available.
- For UI changes, use the installed frontend verification workflow when available.
- For documentation, config, or standalone tooling changes, use ephemeral tests in `.verify-scratch/`, run them, report results, and delete the scratch directory unconditionally.
- After validation, `git status --short` should show only intended source changes.

## Git Rules

- Git is read-only by default.
- Do not run `git add`, `git commit`, `git push`, create branches, or open PRs unless the owner explicitly asks for that exact action in the current conversation.
- Do not add AI attribution, co-author trailers, or generated-by notices to commit messages or PR descriptions.
