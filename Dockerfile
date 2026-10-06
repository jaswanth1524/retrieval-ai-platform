# Base images are pinned by digest (the tag is kept for reading); Dependabot's docker
# ecosystem (.github/dependabot.yml) proposes the updates.

# ---- frontend build stage ----
# Always the builder's own platform: the output is static files, so a multi-arch image
# build doesn't run npm under QEMU.
FROM --platform=$BUILDPLATFORM node:22-slim@sha256:43ac6c60b8f89723f746e8a92ce91abd5017e627ce1ddfe4238355d3a30b772c AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- python build stage: uv resolves and installs the venv ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS python-build
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
# The venv is built on the image's own /usr/local/bin/python3.12 (never a uv-managed
# download), the same path the runtime image below provides, so it runs there as is.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12
# Space-separated uv extras to install, e.g. `--build-arg UV_EXTRAS=ocr` (compose:
# UV_EXTRAS=ocr in .env).
ARG UV_EXTRAS=""
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen $(for extra in $UV_EXTRAS; do printf -- '--extra %s ' "$extra"; done)
COPY api/ ./api/
# UV_COMPILE_BYTECODE only covers what uv sync installs, and `package = false` keeps
# api/ out of that — compile the app's own modules here, for the same reason.
RUN /app/.venv/bin/python -m compileall -q api

# ---- runtime stage: plain Python, no uv or build tooling ----
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS runtime
WORKDIR /app
# OCR's opencv links against libGL/glib, absent from slim; only installed with it.
ARG UV_EXTRAS=""
RUN if printf '%s\n' $UV_EXTRAS | grep -qx ocr; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi
COPY --from=python-build /app/.venv ./.venv
COPY --from=python-build /app/api ./api
# api/version.py reads the version from here; without it the image reports 0.0.0 (in
# /openapi.json and every /export manifest).
COPY pyproject.toml ./
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Non-root: created after the COPYs so the venv/app dir is owned by root at build
# time (default umask still leaves it world-readable) and only the running process
# drops privilege.
RUN useradd --create-home --uid 1000 app
ENV HOME=/home/app
# Model caches under one directory docker-compose mounts as a named volume. fastembed
# otherwise defaults to tempfile.gettempdir()/fastembed_cache — /tmp inside the
# container — so the models (~0.5 GB; ~1.3 GB with the fp32 reranker) re-downloaded
# on every container recreate.
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
# on first boot, before /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --start-interval=5s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Run the venv's uvicorn directly (there is no uv in this image). One worker, and it
# must stay one: the filename locks, the ingest sequencer, question slots, the answer
# and metadata caches and the job registry are all per process.
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
