"""Fixtures for the HTTP route tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.dependencies import (
    clear_dependency_caches,
    get_embedding_provider,
    get_generator,
    get_reranker,
)
from tests.routes_http.support import (
    ApiTestContext,
    FakeEmbeddingProvider,
    FakeGenerator,
    FakeReranker,
    hermetic_app,
    make_settings,
)


@pytest.fixture
def api_context() -> Generator[ApiTestContext]:
    clear_dependency_caches()
    settings = make_settings()
    qdrant = QdrantClient(":memory:")
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()
    generator = FakeGenerator()
    app = hermetic_app(settings, qdrant)
    app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_generator] = lambda: generator

    with TestClient(app) as client:
        yield ApiTestContext(
            client=client,
            app=app,
            settings=settings,
            qdrant=qdrant,
            embeddings=embeddings,
            reranker=reranker,
            generator=generator,
        )

    clear_dependency_caches()
