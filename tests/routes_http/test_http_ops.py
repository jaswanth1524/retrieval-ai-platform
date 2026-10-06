"""Health, config, CORS, auth, metrics, export, lifespan and warmup."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_ollama_reachability_checker,
    get_qdrant_reachability_checker,
)
from api.embeddings import EmbeddedText
from api.main import (
    create_app,
    run_model_warmup,
)
from tests.routes_http.support import (
    ApiTestContext,
    FakeEmbeddingProvider,
    FakeReranker,
    cors_app,
    keyed_client,
    make_settings,
    raw_storage_client,
    upload_and_wait,
)


def test_health_endpoint(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_ok_when_qdrant_and_provider_are_reachable(
    api_context: ApiTestContext,
) -> None:
    api_context.app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: True

    response = api_context.client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "qdrant": True,
        "generation_provider": True,
        "llm_provider": "ollama",
        "index_compatible": True,
        "index_detail": None,
    }


def test_health_endpoint_never_probes_dependencies(api_context: ApiTestContext) -> None:
    def raise_if_called(_client: object) -> bool:
        raise AssertionError("check_qdrant_reachable must not run for /health")

    api_context.app.dependency_overrides[get_qdrant_reachability_checker] = lambda: raise_if_called

    response = api_context.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_exposes_non_secret_settings(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["qdrant_collection"] == "api_documents"
    assert payload["dense_embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert payload["llm_provider"] == "ollama"
    assert payload["openai_available"] is False
    assert "openai_api_key" not in payload


def test_lifespan_exit_shuts_the_ingest_pool_down_and_a_new_lifespan_rebuilds_it(
    api_context: ApiTestContext,
) -> None:
    """The pools are lru_cached singletons: shutting one down without clearing its cache
    handed the next lifespan (a reload, the next TestClient) a dead pool."""

    from api.dependencies import get_ingest_executor

    with TestClient(api_context.app) as client:
        assert upload_and_wait(client, "one.txt", b"alpha")["state"] == "done"
        first_pool = get_ingest_executor()
    assert first_pool._shutdown

    assert upload_and_wait(api_context.client, "two.txt", b"beta")["state"] == "done"
    assert get_ingest_executor() is not first_pool


def test_config_reports_openai_available_when_key_present() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key="sk-test")
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    with TestClient(app) as client:
        payload = client.get("/config").json()
    clear_dependency_caches()

    assert payload["openai_available"] is True
    assert "openai_api_key" not in payload


def test_config_reports_ollama_available_true_or_false_from_checker() -> None:
    clear_dependency_caches()
    settings = make_settings()
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings

    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: True
    with TestClient(app) as client:
        assert client.get("/config").json()["ollama_available"] is True

    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    with TestClient(app) as client:
        assert client.get("/config").json()["ollama_available"] is False
    clear_dependency_caches()


def test_config_exposes_override_limit_fields() -> None:
    clear_dependency_caches()
    settings = make_settings(fused_top_n=5)
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    with TestClient(app) as client:
        payload = client.get("/config").json()
    clear_dependency_caches()

    # rerank_top_k_limit is capped by the server's own fused_top_n (5 here), not the
    # global REQUEST_RERANK_TOP_K_MAX (50).
    assert payload["rerank_top_k_limit"] == 5
    assert payload["max_context_chunks_limit"] == 20
    assert payload["llm_temperature_max"] == 2.0
    assert payload["llm_temperature"] == settings.llm_temperature


def test_cors_preflight_allows_the_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cross-origin request carrying X-API-Key must survive its preflight.

    X-API-Key is not a CORS-simple header, so the browser preflights it. When
    allow_headers omitted it, the middleware answered 400 "Disallowed CORS headers"
    and API_KEY + CORS_ALLOW_ORIGINS could never be used together.
    """

    origin = "http://localhost:5173"
    app = cors_app(monkeypatch, origin)

    with TestClient(app) as client:
        response = client.options(
            "/documents",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
    clear_dependency_caches()

    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    assert "x-api-key" in allowed
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_preflight_still_allows_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing Content-Type allowance is not regressed by adding X-API-Key."""

    origin = "http://localhost:5173"
    app = cors_app(monkeypatch, origin)

    with TestClient(app) as client:
        response = client.options(
            "/questions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    clear_dependency_caches()

    assert response.status_code == 200
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_run_model_warmup_touches_both_models() -> None:
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()

    run_model_warmup(embeddings, reranker)

    assert embeddings.seen_text_batches == [["warmup"]]
    assert reranker.seen_documents == ["warmup"]


def test_run_model_warmup_swallows_failures_instead_of_raising() -> None:
    class RaisingEmbeddingProvider:
        def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
            raise RuntimeError("model download failed")

    reranker = FakeReranker()
    # Must not raise — a warmup failure is logged, not fatal to app startup — and one
    # model failing must not leave the others cold.
    run_model_warmup(RaisingEmbeddingProvider(), reranker)
    assert reranker.seen_documents == ["warmup"]


def test_run_model_warmup_names_a_wrong_dense_vector_size_at_boot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fake embeds 3 dimensions; a collection configured for 4 would only fail at
    the first upload, as "retry the upload"."""

    run_model_warmup(
        FakeEmbeddingProvider(),
        FakeReranker(),
        settings=make_settings(qdrant_dense_vector_size=4),
    )

    assert any(
        "QDRANT_DENSE_VECTOR_SIZE=3" in r.getMessage()
        or (r.exc_info and "QDRANT_DENSE_VECTOR_SIZE=3" in str(r.exc_info[1]))
        for r in caplog.records
    )


def test_lifespan_runs_warmup_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lifespan calls get_embedding_provider()/get_reranker() directly (not via
    FastAPI Depends), so dependency_overrides can't intercept them — patch the names
    api.main actually holds instead, same as production code would resolve them."""

    clear_dependency_caches()
    settings = make_settings(warmup_models=True)
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_embedding_provider", lambda: embeddings)
    monkeypatch.setattr("api.main.get_reranker", lambda: reranker)
    counted: list[str] = []

    class RecordingTokenCounter:
        def count(self, text: str) -> int:
            counted.append(text)
            return 1

    monkeypatch.setattr("api.main.get_token_counter", lambda: RecordingTokenCounter())

    app = create_app()
    with TestClient(app):
        pass

    assert embeddings.seen_text_batches == [["warmup"]]
    assert reranker.seen_documents == ["warmup"]
    # The chunking tokenizer is warmed too, so the first upload doesn't fetch it.
    assert counted == ["warmup"]
    clear_dependency_caches()


def test_lifespan_skips_warmup_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_dependency_caches()
    settings = make_settings(warmup_models=False)
    embeddings = FakeEmbeddingProvider()
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_embedding_provider", lambda: embeddings)
    monkeypatch.setattr("api.main.get_reranker", lambda: FakeReranker())

    app = create_app()
    with TestClient(app):
        pass

    assert embeddings.seen_text_batches == []
    clear_dependency_caches()


def test_metrics_endpoint_exposes_prometheus_text(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_api_key_unset_leaves_every_route_open(api_context: ApiTestContext) -> None:
    """Default AppSettings.api_key is "" — the zero-config self-host story must be
    unaffected: no route requires X-API-Key when the operator never set one."""

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    response = api_context.client.post("/questions", json={"question": "alpha"})
    traces_response = api_context.client.get("/traces")
    list_response = api_context.client.get("/documents")
    content_response = api_context.client.get("/documents/guide.txt/content")

    assert response.status_code == 200
    assert traces_response.status_code == 200
    assert list_response.status_code == 200
    assert content_response.status_code == 200


def test_api_key_set_rejects_protected_routes_without_header() -> None:
    with keyed_client("secret-key") as client:
        health_response = client.get("/health")
        ready_response = client.get("/health/ready")
        config_response = client.get("/config")
        questions_response = client.post("/questions", json={"question": "alpha"})
        traces_response = client.get("/traces")
        upload_response = client.post(
            "/documents", files={"file": ("guide.txt", b"Intro\nalpha beta", "text/plain")}
        )
        delete_response = client.delete("/documents/guide.txt")
        # The corpus read path is guarded too: an open GET /documents enumerates
        # filenames and an open /content dumps every chunk's raw text, so a key that
        # only covered writes and Q&A would still leave the documents readable. The
        # job-status route carries the filename as well, and its id lands in browser
        # history and proxy logs, so it is guarded rather than relying on uuid4 secrecy.
        list_response = client.get("/documents")
        content_response = client.get("/documents/guide.txt/content")
        job_response = client.get("/documents/jobs/any-id")
        metrics_response = client.get("/metrics")
        reindex_response = client.post("/documents/guide.txt/reindex")
        reindex_all_response = client.post("/documents/reindex")

    # Open either way: liveness/readiness probes and the UI's boot request, which has to
    # succeed before the user has anywhere to type a key.
    assert health_response.status_code == 200
    assert ready_response.status_code in (200, 503)
    assert config_response.status_code == 200

    assert questions_response.status_code == 401
    assert traces_response.status_code == 401
    assert upload_response.status_code == 401
    assert delete_response.status_code == 401
    assert list_response.status_code == 401
    assert content_response.status_code == 401
    # 401, not the 404 an unguarded route would return for an unknown job id.
    assert job_response.status_code == 401
    # Counters disclose usage volume and corpus growth; an operator who set a key did
    # not intend to publish those.
    assert metrics_response.status_code == 401
    assert reindex_response.status_code == 401
    assert reindex_all_response.status_code == 401


def test_api_key_set_accepts_protected_routes_with_matching_header() -> None:
    headers = {"X-API-Key": "secret-key"}
    with keyed_client("secret-key") as client:
        upload_and_wait(client, "guide.txt", b"Intro\nalpha beta", headers=headers)
        questions_response = client.post("/questions", json={"question": "alpha"}, headers=headers)
        traces_response = client.get("/traces", headers=headers)
        list_response = client.get("/documents", headers=headers)
        content_response = client.get("/documents/guide.txt/content", headers=headers)
        metrics_response = client.get("/metrics", headers=headers)

    assert questions_response.status_code == 200
    assert traces_response.status_code == 200
    assert list_response.status_code == 200
    assert content_response.status_code == 200
    assert metrics_response.status_code == 200


def test_api_key_set_rejects_wrong_header_value() -> None:
    with keyed_client("secret-key") as client:
        response = client.post(
            "/questions", json={"question": "alpha"}, headers={"X-API-Key": "wrong-key"}
        )

    assert response.status_code == 401


def test_api_key_non_ascii_configured_key_matches_utf8_header() -> None:
    """An operator-configured non-ASCII key still authenticates a utf-8 header.

    Guards the bytes comparison against a naive fix that only avoided the TypeError:
    latin-1-encoding the header recovers the exact wire bytes, so a key set as utf-8 in
    the environment matches a client sending those same utf-8 bytes.
    """

    with keyed_client("clé-secrète") as client:
        matching = client.get("/documents", headers={b"X-API-Key": "clé-secrète".encode()})
        mismatched = client.get("/documents", headers={b"X-API-Key": "clé-secrete".encode()})

    assert matching.status_code == 200, matching.text
    assert mismatched.status_code == 401, mismatched.text


def test_metrics_endpoint_exposes_the_question_and_ingest_series(
    api_context: ApiTestContext,
) -> None:
    """The prior test asserted only a 200 and a content-type, so it would have passed
    against every counter being mislabeled or never incremented — which matters now
    that outcome/error_type labels exist on both counters."""

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert api_context.client.post("/questions", json={"question": "alpha"}).status_code == 200

    body = api_context.client.get("/metrics").text

    assert 'docrag_questions_total{error_type="",outcome="ok"}' in body
    assert 'docrag_ingest_jobs_total{error_type="",outcome="done"}' in body
    assert 'docrag_ingest_job_seconds_count{outcome="done"}' in body
    for stage in ("embed", "search", "rerank", "generate", "total"):
        assert f'docrag_question_stage_seconds_sum{{stage="{stage}"}}' in body


def test_config_reports_the_openai_compatible_provider(api_context: ApiTestContext) -> None:
    payload = api_context.client.get("/config").json()
    assert payload["openai_compatible_available"] is False

    configured = make_settings(
        openai_compatible_base_url="http://vllm:8000/v1", openai_compatible_model="qwen"
    )
    api_context.app.dependency_overrides[get_app_settings] = lambda: configured

    payload = api_context.client.get("/config").json()
    assert payload["openai_compatible_available"] is True
    assert payload["openai_compatible_model"] == "qwen"


def test_the_api_version_comes_from_pyproject(api_context: ApiTestContext) -> None:
    import tomllib

    with open("pyproject.toml", "rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]

    assert api_context.client.get("/openapi.json").json()["info"]["version"] == expected


def test_the_read_only_key_can_ask_and_read_but_not_change_anything() -> None:
    full = {"X-API-Key": "full-key"}
    read = {"X-API-Key": "read-key"}
    with keyed_client("full-key", api_read_key="read-key", feedback_enabled=True) as client:
        assert upload_and_wait(client, "a.txt", b"Intro\nalpha", headers=full)["state"] == "done"

        # Reading and asking work with either key.
        assert client.get("/documents", headers=read).status_code == 200
        assert client.get("/documents/a.txt/content", headers=read).status_code == 200
        assert (
            client.post("/questions", json={"question": "alpha"}, headers=read).status_code == 200
        )
        assert client.get("/traces", headers=read).status_code == 200

        # Changing the corpus, and operator data, need the full key.
        upload = client.post("/documents", files={"file": ("b.txt", b"x")}, headers=read)
        assert upload.status_code == 403
        assert "read-only" in upload.json()["detail"]
        assert client.delete("/documents/a.txt", headers=read).status_code == 403
        tags = client.patch("/documents/a.txt/tags", json={"tags": ["x"]}, headers=read)
        assert tags.status_code == 403
        assert client.post("/documents/a.txt/reindex", headers=read).status_code == 403
        assert client.get("/metrics", headers=read).status_code == 403
        assert client.get("/feedback", headers=read).status_code == 403

        assert (
            client.patch("/documents/a.txt/tags", json={"tags": ["x"]}, headers=full).status_code
            == 200
        )
        assert client.get("/metrics", headers=full).status_code == 200
        assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401


def test_a_read_key_without_a_full_key_is_a_configuration_error() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="API_READ_KEY needs API_KEY"):
        make_settings(api_read_key="read-key")
    with pytest.raises(ValidationError, match="must differ"):
        make_settings(api_key="same", api_read_key="same")


def test_export_zips_the_manifest_originals_and_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        assert client.patch("/documents/guide.txt/tags", json={"tags": ["kb"]}).status_code == 200

        response = client.get("/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "docrag-export-" in response.headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["embedding_model_tag"] == "test-embedding:v1"
    [document] = manifest["documents"]
    assert document["filename"] == "guide.txt"
    assert document["tags"] == ["kb"]
    assert document["original"] == "originals/guide.txt"
    # Which chunking produced each document, next to the current one.
    assert document["chunking_fingerprints"] == [manifest["chunking_fingerprint"]]
    assert archive.read("originals/guide.txt") == b"Intro\nalpha beta"
    assert "feedback.jsonl" not in archive.namelist()  # feedback is off here


def test_export_includes_feedback_when_it_is_on(tmp_path: Path) -> None:
    import io
    import zipfile

    from api.dependencies import get_feedback_store
    from api.feedback import FeedbackStore

    store = FeedbackStore(str(tmp_path / "f.db"))
    with keyed_client(
        "full-key",
        extra_overrides={get_feedback_store: lambda: store},
        feedback_enabled=True,
    ) as client:
        headers = {"X-API-Key": "full-key"}
        client.post(
            "/feedback",
            headers=headers,
            json={
                "trace_id": None,
                "question": "q",
                "answer_excerpt": "a",
                "cited_filenames": [],
                "rating": "up",
                "citation_source_number": None,
            },
        )
        assert client.get("/export").status_code == 401
        response = client.get("/export", headers=headers)

    rows = (
        zipfile.ZipFile(io.BytesIO(response.content)).read("feedback.jsonl").decode().splitlines()
    )
    assert [json.loads(row)["rating"] for row in rows] == ["up"]


def test_every_response_carries_a_request_id_and_keeps_a_sane_supplied_one(
    api_context: ApiTestContext,
) -> None:
    client = api_context.client
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32

    kept = client.get("/health", headers={"X-Request-ID": "lb-123.abc"})
    assert kept.headers["X-Request-ID"] == "lb-123.abc"
    replaced = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert replaced.headers["X-Request-ID"] != "bad id\nwith newline"


def test_a_trace_records_the_request_id_of_its_question(api_context: ApiTestContext) -> None:
    client = api_context.client
    assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"

    answer = client.post(
        "/questions", json={"question": "alpha"}, headers={"X-Request-ID": "ask-1"}
    ).json()

    assert client.get(f"/traces/{answer['trace_id']}").json()["request_id"] == "ask-1"


def test_a_reranker_that_fails_to_load_names_the_setting_not_the_library_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.dependencies as dependencies
    from api.reranking import RerankingError

    def broken(settings: object) -> object:
        raise ValueError("Model x is not supported in TextCrossEncoder; /home/app/.cache/...")

    clear_dependency_caches()
    monkeypatch.setattr(dependencies, "LocalCrossEncoderReranker", broken)
    try:
        with pytest.raises(RerankingError) as raised:
            dependencies.get_reranker()
    finally:
        clear_dependency_caches()

    assert "RERANKER_MODEL" in str(raised.value)
    assert "/home/app" not in str(raised.value)
