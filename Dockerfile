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
# Dependency layer first, and on its own: `[tool.uv] package = false` means uv sync
# never reads api/, so copying the source before it only served to invalidate the
# (slow, network-bound) dependency install on every code change.
COPY pyproject.toml uv.lock ./
# Bytecode is compiled at build time because the runtime user can't write
# __pycache__ into this root-owned venv, so every boot recompiled every import:
# `import api.main` measured 5.1-6.8s without it, 2.3-3.7s with it (+93 MB image).
# The cache mount keeps downloaded wheels across builds, so a uv.lock change doesn't
# re-download everything; copy mode because the mount is a different filesystem.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
COPY api/ ./api/
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Non-root: created after uv sync/COPY so the venv/app dir is owned by root at build
# time (default umask still leaves it world-readable) and only the running process
# drops privilege.
RUN useradd --create-home --uid 1000 app
ENV HOME=/home/app
# Model caches under one directory docker-compose mounts as a named volume. fastembed
# otherwise defaults to tempfile.gettempdir()/fastembed_cache — /tmp inside the
# container — so the ~1.2 GB of models re-downloaded on every container recreate.
# Created here, owned by `app`, so a fresh named volume is seeded writable.
ENV FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed HF_HOME=/home/app/.cache/huggingface
RUN mkdir -p /home/app/.cache/fastembed /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache
# /app/data is where JOB_STORE_PATH, RAW_DOCUMENT_DIR and FEEDBACK_STORE_PATH all
# default to, and it must exist in the image owned by `app` for either of the two
# ways it gets used to work: without a mount, the running (non-root) process cannot
# mkdir it inside root-owned /app; with the docker-compose volume mounted here,
# Docker seeds a new named volume from the image directory — including its
# ownership — so a root-owned or absent /app/data yields a volume the process
# cannot write either.
RUN mkdir -p /app/data && chown app:app /app/data
USER app

EXPOSE 8000

# No curl/wget in the slim image — a stdlib urllib check instead. start-period is
# generous because WARMUP_MODELS=true downloads the reranker (~0.28 GB int8, ~1.1 GB fp32)
# on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Run the venv's uvicorn directly rather than `uv run` — the runtime user has no
# write access to re-verify/re-lock the environment, and doesn't need to.
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
