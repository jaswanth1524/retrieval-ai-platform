# Local shortcuts for the checks CI runs (.github/workflows/ci.yml).
# `make check` runs the backend and frontend gates as CI does (coverage floor and
# dependency audits included); e2e and Docker stay separate.

.PHONY: check backend frontend e2e golden openapi

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
		&& uvx pip-audit==2.10.1 --disable-pip --no-deps -r "$$req" \
			$$(awk '!/^#/ && NF {printf "--ignore-vuln %s ", $$1}' .github/ci/pip-audit-ignore.txt); \
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
