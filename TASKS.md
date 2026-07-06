# DocRAG Task Plan

This task plan follows the requested build order and the architecture implied by the specification. Application feature code starts only after owner approval for the relevant milestone.

## Milestone 1: Planning Artifacts And Spec Review

- Create `AGENTS.md`, `TASKS.md`, `README.md`, `.gitignore`, and `.env.example`.
- Record architecture decisions, repository expectations, validation rules, and known risks.
- Confirm no application code is created in this milestone.

## Milestone 2: Project Scaffold

- Add `pyproject.toml` for Python 3.12 and `uv`.
- Create the planned directories: `api/`, `ui/`, `eval/`, and `tests/`.
- Add `docker-compose.yml` for the one-command local run path.
- Keep services configurable through `.env`.
- Status: scaffold created without application feature code.

## Milestone 3: Qdrant Schema And Configuration

- Configure Qdrant as the vector store.
- Use named dense and sparse vectors per point.
- Store collection metadata including the embedding model version tag.
- Add startup checks for collection existence and embedding tag compatibility.
- Status: schema helpers, settings, and tests added.

## Milestone 4: Document Upload, Parsing, And Chunking

- Accept document uploads through the browser and API.
- Extract text and metadata from supported document formats.
- Chunk text while carrying source metadata: filename, page, section, and chunk_id.
- Preserve metadata into Qdrant point payloads.
- Status: parsing and chunking helpers added for PDF, text, and Markdown; API/UI upload surfaces remain in later milestones.

## Milestone 5: Local Embedding And Ingestion

- Generate local dense embeddings with FastEmbed model `BAAI/bge-small-en-v1.5`.
- Generate local sparse vectors with FastEmbed model `Qdrant/BM25`.
- Upsert chunks into Qdrant with named dense and sparse vectors.
- Ensure no hosted key is required for ingestion.
- Status: local FastEmbed provider and Qdrant chunk ingestion added; tests use fake embedders to avoid model downloads.

## Milestone 6: Hybrid Retrieval

- Query dense and sparse indexes for each user question.
- Use Qdrant server-side hybrid query where supported.
- Fall back to separate dense and sparse result lists plus manual Reciprocal Rank Fusion with `k=60`.
- Limit fused candidates before reranking.
- Status: hybrid retrieval with Qdrant RRF and manual RRF fallback added.

## Milestone 7: Cross-Encoder Reranking

- Rerank only the fused top-N candidates.
- Use FastEmbed reranker `BAAI/bge-reranker-v2-m3`.
- Return the highest-ranked chunks with their source metadata intact.
- Status: reranking adapter and top-N/top-K ordering added; tests use fake cross-encoder scores to avoid model downloads.

## Milestone 8: Grounded Generation

- Add LiteLLM as the single completion interface.
- Support Ollama and OpenAI generation providers.
- Default to Ollama model `llama3.1:8b`.
- Keep embedding provider selection independent from generation provider selection.
- Prompt the LLM to answer only from retrieved context and state when context is insufficient.
- Status: grounded prompt builder, LiteLLM adapter, provider config, and citation packaging added.

## Milestone 9: FastAPI Service

- Expose health, configuration, upload/ingest, and question-answering endpoints.
- Return answers with citations.
- Return clear errors for embedding model mismatches, missing indexes, and insufficient context.
- Status: FastAPI app, public config, document upload/ingest, question-answering endpoint, and domain error mapping added.

## Milestone 10: Streamlit UI

- Provide document upload from the browser.
- Provide a question input and answer display.
- Show citations with filename, page, section, and chunk_id.
- Surface configuration and readiness errors clearly.
- Status: Streamlit UI and typed API client added.

## Milestone 11: Evaluation

- Add RAGAS-based evaluation under `eval/`.
- Keep evaluation optional and configurable.
- Avoid requiring hosted keys for the core app path.
- Status: optional lazy-loaded RAGAS evaluation runner and dataset validation added.

## Milestone 12: Tests And Documentation

- Add Pytest coverage for ingestion, retrieval fusion, reranking boundaries, provider configuration, and API behavior.
- Use FastAPI TestClient for endpoint tests.
- Document setup, one-command run, provider configuration, re-ingestion requirements, and troubleshooting.

## Specification Risks And Inconsistencies

- The prompt references a "Build Order section," but no explicit section with that heading is present; this plan infers build order from the requested artifact order and dependency flow.
- The requested repository structure is shown under `docrag/`, while the requested planning files are named without that prefix; this repository now treats the current root as the DocRAG project root.
- Qdrant server-side hybrid query support depends on the installed Qdrant server/client feature set, so manual RRF fallback is required.
- First-run FastEmbed model downloads may make the initial one-command run slower.
- `BAAI/bge-reranker-v2-m3` may require more CPU and memory than the dense embedding model.
- Installed FastEmbed exposes `TextCrossEncoder`, but version `0.8.0` does not list `BAAI/bge-reranker-v2-m3` as a supported model; runtime reranker setup may require a newer FastEmbed release, custom model registration, or an owner-approved supported-model fallback.
- RAGAS evaluation can require an evaluator LLM for some metrics; evaluation remains optional and separate from the core local workflow.
- The currently resolved RAGAS dependency set fails to import because `ragas 0.4.3` expects `langchain_community.chat_models.vertexai`, which is absent from `langchain-community 0.4.2`; the evaluation runner lazy-loads RAGAS and reports this setup error without breaking the core app.
