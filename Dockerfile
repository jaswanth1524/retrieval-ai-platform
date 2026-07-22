# ---- frontend build stage ----
FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- python runtime stage ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY api/ ./api/
RUN uv sync --frozen
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Non-root: created after uv sync/COPY so the venv/app dir is owned by root at build
# time (default umask still leaves it world-readable) and only the running process
# drops privilege. HOME must be writable — fastembed/HuggingFace cache under it at
# runtime (WARMUP_MODELS or the first real query both trigger a model download).
RUN useradd --create-home --uid 1000 app
ENV HOME=/home/app
USER app

EXPOSE 8000

# No curl/wget in the slim image — a stdlib urllib check instead. start-period is
# generous because WARMUP_MODELS=true downloads the ~1.1 GB reranker on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Run the venv's uvicorn directly rather than `uv run` — the runtime user has no
# write access to re-verify/re-lock the environment, and doesn't need to.
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
