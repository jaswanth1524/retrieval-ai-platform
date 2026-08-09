# DocRAG frontend

React + Vite + TypeScript UI for DocRAG. In production the API serves this directory's
build output (`dist/`) as static files from the same origin (`api/main.py`); in
development, Vite proxies API calls to `localhost:8000`.

See the top-level [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md) for the full
project overview, running instructions, and architecture. Common commands:

```bash
npm ci                         # install deps (matches CI)
npm run dev                    # Vite dev server, proxies API calls to localhost:8000
npm run test                   # vitest run (all tests)
npm run build                  # tsc -b && vite build -> dist/
npm run lint                   # oxlint
```
