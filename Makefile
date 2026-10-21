# Local shortcuts for the checks CI runs (.github/workflows/ci.yml).
# `make check` runs the backend and frontend gates as CI does (coverage floor and
# dependency audits included); e2e and Docker stay separate.

.PHONY: check backend frontend e2e golden openapi eval-ingest eval-retrieval eval-answer eval-ragas

check: backend frontend

backend:
	uv run pytest --cov=api --cov=eval --cov=cli --cov-report=term-missing:skip-covered --cov-fail-under=92
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy api eval cli
	out=$$(mktemp -d) && uv build ./cli --wheel --out-dir "$$out/dist" \
		&& uv venv -q "$$out/venv" \
		&& VIRTUAL_ENV="$$out/venv" uv pip install -q "$$(ls "$$out"/dist/*.whl)[mcp]" \
		&& cd "$$out" && "$$out/venv/bin/docrag" --help > /dev/null \
		&& test -x "$$out/venv/bin/docrag-mcp" \
		&& "$$out/venv/bin/python" -I -c 'import sys, cli; from importlib.metadata import entry_points; [entry_points(group="console_scripts")[name].load() for name in ("docrag", "docrag-mcp")]; assert cli.__file__.startswith(sys.prefix), cli.__file__'; \
		status=$$?; rm -rf "$$out"; exit $$status
	# As CI: ragas has broken on import alone, which --validate-only never touches.
	uv run python -c "import ragas"
	uv run python -m eval.ragas_runner eval/datasets/sample_eval.jsonl --validate-only
	uv run python -m eval.harness validate eval/datasets/retrieval_eval.sample.jsonl
	uv run python -m eval.harness validate eval/datasets/handbook_eval.jsonl
	uv run python -m eval.harness validate eval/datasets/multilingual_eval.jsonl
	uv run python -m eval.harness validate eval/datasets/unanswerable_eval.jsonl
	uv run python -m eval.harness validate eval/datasets/followup_eval.jsonl
	uv run python -m eval.harness validate eval/datasets/paraphrase_eval.jsonl
	req=$$(mktemp) && uv export --frozen --extra ocr --extra mcp --no-hashes --no-emit-project > "$$req" \
		&& uvx pip-audit==2.10.1 --disable-pip --no-deps -r "$$req"; \
		status=$$?; rm -f "$$req"; exit $$status

frontend:
	cd frontend && npm run test
	cd frontend && npm run typecheck:tests
	cd frontend && npm run build
	cd frontend && npm run lint
	cd frontend && npm audit --omit=dev --audit-level=high

e2e:
	cd frontend && npm run e2e

# Regenerate the pinned snapshots after an intended change, then review the diff.
golden:
	DOCRAG_UPDATE_GOLDEN=1 uv run pytest tests/test_chunking_golden.py tests/test_openapi_snapshot.py

openapi:
	DOCRAG_UPDATE_GOLDEN=1 uv run pytest tests/test_openapi_snapshot.py

# Live evals (need Qdrant on EVAL_QDRANT_URL, the models, and for answers an LLM).
# The handbook corpus goes into its own collection with the settings its baseline was
# measured with (as CI's qdrant job does), so the app's collection is never touched.
# LITELLM_MODE=PRODUCTION stops LiteLLM loading .env into the process; settings still
# read .env, so a different reranker there makes the baseline check refuse (by design).
EVAL_QDRANT_URL ?= http://127.0.0.1:6333
EVAL_DATASET ?= eval/datasets/handbook_eval.jsonl
EVAL_OUT ?= eval-answers.json
EVAL_ARGS ?=
LLM ?= ollama_chat/qwen2.5
EVAL_ENV = LITELLM_MODE=PRODUCTION QDRANT_URL=$(EVAL_QDRANT_URL) QDRANT_COLLECTION=eval_handbook \
	CHUNK_SIZE_TOKENS=48 CHUNK_OVERLAP_TOKENS=8 MIN_SECTION_WORDS=0

eval-ingest:
	$(EVAL_ENV) uv run python -m eval.harness ingest eval/corpus_handbook

eval-retrieval: eval-ingest
	$(EVAL_ENV) uv run python -m eval.harness retrieval eval/datasets/handbook_eval.jsonl \
		--baseline eval/baselines/handbook_reranked.json --max-regression 0.06

# e.g. make eval-answer EVAL_DATASET=eval/datasets/followup_eval.jsonl EVAL_ARGS="--resume"
eval-answer: eval-ingest
	$(EVAL_ENV) uv run python -m eval.harness answer $(EVAL_DATASET) --output $(EVAL_OUT) $(EVAL_ARGS)

# Scores eval-answer's output with a local judge by default (needs the eval extra).
eval-ragas:
	LITELLM_MODE=PRODUCTION uv run python -m eval.ragas_runner $(EVAL_OUT) --llm $(LLM) \
		--embeddings fastembed --output eval-ragas.json
