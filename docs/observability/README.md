# Monitoring DocRAG

`GET /metrics` serves Prometheus metrics (with `API_KEY` set, the scrape needs the full key
in `X-API-Key`).

- `grafana-dashboard.json`: import it in Grafana (Dashboards → Import) and pick your
  Prometheus data source. Questions, answer time by stage, grounding, caches, indexing,
  free disk, backups, the watched folder, tokens, failed jobs by kind, overdue URL
  refreshes.
- `alerts.yml`: Prometheus alert rules — failing or stale backups, low disk, indexing
  failures that aren't a bad document, files the watcher gave up on, prompts cut by a
  small context window, questions erroring, URL refreshes falling behind, background
  (URL, watcher, re-index) jobs that keep failing.

`tests/test_observability.py` checks that every metric these files name exists.
