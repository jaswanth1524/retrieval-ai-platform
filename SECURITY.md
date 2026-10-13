# Security

DocRAG is self-hosted: you run it, on your machine or network, with your documents.
This page says what it protects, what it leaves to you, and how to report a problem.

## Reporting a vulnerability

Please report it privately through the repository's **Security** tab
(*Report a vulnerability*), not in a public issue. Include the version (`/health` or
`pyproject.toml`), how it is deployed, and the steps that show the problem. Fixes land on
`main` and in the next release; the changelog's **Security** section lists them.

## What DocRAG does

- **Access.** Off by default. With `API_KEY` set, every route that reads or changes the
  corpus, answers questions, or exposes traces, feedback, metrics, exports or the API
  docs needs a matching `X-API-Key` header; only `/health`, `/health/ready` and
  `/config` stay open. `API_READ_KEY` adds a second key that can read and ask but not
  change anything. Keys are compared in constant time. This is one shared key, not user
  accounts.
- **Uploads are untrusted.** Type and name are checked before the body is read; sizes are
  capped (`MAX_UPLOAD_BYTES`, a pending-bytes budget); zip-based formats (DOCX, EPUB) are
  refused past an expanded-size limit; EPUB DTDs are refused; nothing is ever extracted
  to disk; filenames are reduced to a base name. A backup restore reads only the members
  it expects, each through a capped read.
- **Request bodies** are size-capped before FastAPI reads them, and client-sent lists
  are capped in both count and item length.
- **Rate limiting** (opt-in, `RATE_LIMIT_PER_MINUTE`): each client gets that many
  changing requests (questions, searches, uploads, feedback, deletes) per minute, and
  every request with a wrong API key counts too, which slows key guessing. At most
  `MAX_CONCURRENT_QUESTIONS` questions run at once either way. Behind a reverse proxy,
  run uvicorn with `--forwarded-allow-ips` so clients aren't all the proxy's address.
- **The browser UI** is served with a Content-Security-Policy that allows scripts,
  styles, fonts and requests from its own origin only, frames only for the document
  viewer's PDF (a `blob:` URL), and lets no other site frame it (`frame-ancestors
  'self'`, `X-Frame-Options: DENY`). Every response carries
  `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. Answers never
  load images. A frontend served from a different origin (`VITE_API_BASE_URL`) is not
  covered by this policy; set your own.
- **Prompt injection.** Retrieved document text is placed in the prompt as data: the
  system prompt says so, and text anywhere in a source that imitates a source header
  (`[2] filename=…`) is rewritten as `(ref) filename=…`, without the number, so a
  document can't pass its text off as another document's. This reduces, but cannot
  remove, the chance that a model follows instructions planted in a document. Treat
  answers about untrusted documents with the same care as the documents.
- **Containers.** The compose file runs the API read-only, as a non-root user, with no
  Linux capabilities and `no-new-privileges`; Qdrant and the API listen on 127.0.0.1
  unless `API_BIND` says otherwise.

## What is left to you

- **TLS.** DocRAG speaks plain HTTP. Put it behind a reverse proxy that terminates TLS
  before exposing it beyond your machine, especially with `API_KEY` set (the key travels
  in a header).
- **Exposure.** Without `API_KEY`, anyone who can reach the port can read, change and
  delete every document. Keep it on localhost or a trusted network, or set a key.
- **Hosted LLMs.** With `LLM_PROVIDER=openai` (or an OpenAI-compatible server), the
  question and the retrieved passages are sent to that provider. Ingestion and
  embeddings stay local either way.
- **Secrets.** `.env` holds keys; keep it out of version control and backups you share.
- **Updates.** Dependencies, base images and actions are pinned and bumped by hand. CI
  audits the runtime dependencies and scans the image on every push and weekly, and
  reviews the dependencies a pull request adds; a release is built only from a commit on
  `main` that passed CI, and both architectures' images must pass the scan first.
- **Rate limits** are off by default; turn on `RATE_LIMIT_PER_MINUTE` when the instance
  is reachable by people you don't fully trust.
