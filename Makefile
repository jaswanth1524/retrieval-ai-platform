# Local shortcuts for the checks CI runs (.github/workflows/ci.yml).
# `make check` runs the backend and frontend gates as CI does (coverage floor and
# dependency audits included); e2e and Docker stay separate.

.PHONY: check backend frontend e2e golden openapi

check: backend frontend

backend:
	uv run pytest --cov=api --cov=eval --cov-report=term-missing:skip-covered --cov-fail-under=92
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy api eval
	uv run python -m eval.ragas_runner eval/datasets/sample_eval.jsonl --validate-only
	uv run python -m eval.harness validate eval/datasets/retrieval_eval.sample.jsonl
	uv run python -m eval.harness validate eval/datasets/handbook_eval.jsonl
	req=$$(mktemp) && uv export --frozen --extra ocr --no-hashes --no-emit-project > "$$req" \
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
