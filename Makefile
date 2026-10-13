# Local shortcuts for the checks CI runs (.github/workflows/ci.yml).
# `make check` runs the backend and frontend gates; e2e and Docker stay separate.

.PHONY: check backend frontend e2e golden openapi

check: backend frontend

backend:
	uv run pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy api eval

frontend:
	cd frontend && npm run test
	cd frontend && npm run typecheck:tests
	cd frontend && npm run build
	cd frontend && npm run lint

e2e:
	cd frontend && npm run e2e

# Regenerate the pinned snapshots after an intended change, then review the diff.
golden:
	DOCRAG_UPDATE_GOLDEN=1 uv run pytest tests/test_chunking_golden.py tests/test_openapi_snapshot.py

openapi:
	DOCRAG_UPDATE_GOLDEN=1 uv run pytest tests/test_openapi_snapshot.py
