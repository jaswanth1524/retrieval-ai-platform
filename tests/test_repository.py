from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from api.repository import DocumentMetadata, VectorRepository, reciprocal_rank_fusion
from api.settings import AppSettings
from tests.factories import in_memory_qdrant, make_test_settings, wait_until


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        **in_memory_qdrant("repository_documents"),
    }
    defaults.update(overrides)
    return make_test_settings(**defaults)


def scored_point(chunk_id: str, score: float) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, chunk_id)),
        version=0,
        score=score,
        payload={
            "filename": "guide.md",
            "page": 1,
            "section": "Setup",
            "chunk_id": chunk_id,
            "text": f"text {chunk_id}",
        },
    )


def make_point(
    point_id: str,
    filename: str,
    page: int = 1,
    byte_size: int | None = None,
    uploaded_at: float | None = None,
) -> models.PointStruct:
    payload: dict[str, str | int | float] = {
        "filename": filename,
        "page": page,
        "section": "Intro",
        "chunk_id": point_id,
        "text": "hello",
    }
    if byte_size is not None:
        payload["byte_size"] = byte_size
    if uploaded_at is not None:
        payload["uploaded_at"] = uploaded_at
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, point_id)),
        vector={
            "dense": [0.1, 0.2, 0.3],
            "sparse": models.SparseVector(indices=[1], values=[0.5]),
        },
        payload=payload,
    )


def test_reciprocal_rank_fusion_combines_duplicate_points() -> None:
    point_a = scored_point("a", 0.9)
    point_b = scored_point("b", 0.8)
    point_c = scored_point("c", 0.7)

    fused = reciprocal_rank_fusion(
        [[point_a, point_b], [point_b, point_c]],
        k=60,
        limit=3,
    )

    assert [str(point.id) for point in fused] == [str(point_b.id), str(point_a.id), str(point_c.id)]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 60)


def test_vector_repository_upsert_indexes_and_ensures_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)

    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert client.collection_exists(settings.qdrant_collection)
    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["filename"] == "guide.md"


def test_vector_repository_point_ids_for_filename_returns_only_that_files_ids() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    guide_point = make_point("p1", "guide.md")
    repository.upsert(settings, [guide_point, make_point("p2", "other.md")])

    ids = repository.point_ids_for_filename(settings, "guide.md")

    assert ids == [str(guide_point.id)]


def test_vector_repository_point_ids_for_filename_empty_for_unknown_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert repository.point_ids_for_filename(settings, "missing.md") == []


def test_vector_repository_delete_by_ids_removes_only_those_points() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    guide_point = make_point("p1", "guide.md")
    other_point = make_point("p2", "other.md")
    repository.upsert(settings, [guide_point, other_point])

    repository.delete_by_ids(settings, [str(guide_point.id)])

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert [record.payload["filename"] for record in records if record.payload] == ["other.md"]


def test_vector_repository_delete_by_ids_is_a_noop_for_empty_list() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    repository.delete_by_ids(settings, [])

    records, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False
    )
    assert len(records) == 1


def test_vector_repository_filename_metadata_counts_points_per_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [make_point("p1", "b.md"), make_point("p2", "a.md"), make_point("p3", "b.md")],
    )

    metadata = repository.filename_metadata(settings)

    assert {name: meta.chunk_count for name, meta in metadata.items()} == {"a.md": 1, "b.md": 2}


def test_vector_repository_filename_metadata_aggregates_page_count_and_stamped_fields() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_point("p1", "guide.md", page=1, byte_size=2048, uploaded_at=1700000000.0),
            make_point("p2", "guide.md", page=3, byte_size=2048, uploaded_at=1700000000.0),
            make_point("p3", "guide.md", page=2, byte_size=2048, uploaded_at=1700000000.0),
        ],
    )

    metadata = repository.filename_metadata(settings)

    assert metadata == {
        "guide.md": DocumentMetadata(
            chunk_count=3, page_count=3, byte_size=2048, uploaded_at=1700000000.0
        )
    }


def test_vector_repository_filename_metadata_nulls_stamped_fields_for_legacy_points() -> None:
    """A point ingested before byte_size/uploaded_at existed has neither key in its
    payload at all — this must not crash, and both fields must come back None rather
    than some default like 0."""

    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "old.md", page=1)])

    metadata = repository.filename_metadata(settings)

    assert metadata["old.md"] == DocumentMetadata(
        chunk_count=1, page_count=1, byte_size=None, uploaded_at=None
    )


def test_vector_repository_filename_metadata_empty_for_empty_collection() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.ensure_ready(settings)

    assert repository.filename_metadata(settings) == {}


def make_ordinal_point(point_id: str, filename: str, ordinal: int, text: str) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, point_id)),
        vector={
            "dense": [0.1, 0.2, 0.3],
            "sparse": models.SparseVector(indices=[1], values=[0.5]),
        },
        payload={
            "filename": filename,
            "page": 1,
            "section": "Intro",
            "chunk_id": point_id,
            "text": text,
            "chunk_ordinal": ordinal,
        },
    )


def test_vector_repository_fetch_neighbors_returns_matching_ordinals() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_ordinal_point("p1", "guide.md", 1, "one"),
            make_ordinal_point("p2", "guide.md", 2, "two"),
            make_ordinal_point("p3", "guide.md", 3, "three"),
            make_ordinal_point("p4", "other.md", 2, "other-two"),
        ],
    )

    neighbors = repository.fetch_neighbors(settings, "guide.md", [1, 3])

    texts = sorted(point.payload["text"] for point in neighbors if point.payload)
    assert texts == ["one", "three"]


def test_vector_repository_fetch_neighbors_empty_for_no_ordinals() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)

    assert repository.fetch_neighbors(settings, "guide.md", []) == []


def test_vector_repository_chunks_for_filename_sorted_by_ordinal() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(
        settings,
        [
            make_ordinal_point("p3", "guide.md", 3, "three"),
            make_ordinal_point("p1", "guide.md", 1, "one"),
            make_ordinal_point("p2", "guide.md", 2, "two"),
            make_ordinal_point("x1", "other.md", 1, "other-one"),
        ],
    )

    chunks = repository.chunks_for_filename(settings, "guide.md")

    assert [payload["chunk_ordinal"] for payload in chunks] == [1, 2, 3]
    assert all(payload["filename"] == "guide.md" for payload in chunks)


def test_vector_repository_chunks_for_filename_empty_for_unknown_file() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [make_point("p1", "guide.md")])

    assert repository.chunks_for_filename(settings, "missing.md") == []


def test_vector_repository_scrolls_page_by_page_across_every_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every whole-collection/per-file reader shares one paging loop; with a page size
    of 2 and 5 points, a reader that stopped after the first page would under-count."""

    import api.repository as repository_module

    monkeypatch.setattr(repository_module, "_SCROLL_PAGE_SIZE", 2)
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(
        settings,
        [make_ordinal_point(f"g{n}", "guide.md", n, f"text {n}") for n in range(1, 6)]
        + [make_ordinal_point("o1", "other.md", 1, "other")],
    )

    assert len(repository.point_ids_for_filename(settings, "guide.md")) == 5
    assert [p["chunk_ordinal"] for p in repository.chunks_for_filename(settings, "guide.md")] == [
        1,
        2,
        3,
        4,
        5,
    ]
    metadata = repository.filename_metadata(settings)
    assert {name: meta.chunk_count for name, meta in metadata.items()} == {
        "guide.md": 5,
        "other.md": 1,
    }


def test_filename_metadata_reports_the_oldest_chunker_version_and_legacy_as_none() -> None:
    """Staleness uses the minimum: a re-ingest that failed partway leaves old-version
    points beside new ones, and any point without the field is legacy."""

    def versioned(point_id: str, filename: str, version: int | None) -> models.PointStruct:
        point = make_point(point_id, filename, uploaded_at=100.0)
        assert point.payload is not None
        if version is not None:
            point.payload["chunker_version"] = version
        return point

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(
        settings,
        [
            versioned("a1", "current.md", 3),
            versioned("a2", "current.md", 3),
            versioned("b1", "mixed.md", 3),
            versioned("b2", "mixed.md", 2),
            versioned("c1", "legacy.md", 3),
            versioned("c2", "legacy.md", None),
        ],
    )

    metadata = repository.filename_metadata(settings)

    assert metadata["current.md"].chunker_version == 3
    assert metadata["mixed.md"].chunker_version == 2
    assert metadata["legacy.md"].chunker_version is None
    assert repository.uploaded_at_for_filename(settings, "mixed.md") == 100.0
    assert repository.uploaded_at_for_filename(settings, "absent.md") is None


def test_a_collection_deleted_under_a_running_process_is_recreated_not_a_500() -> None:
    """Readiness is cached per process: a collection dropped out of band used to fail
    every later call until a restart."""

    from api.embeddings import EmbeddedText
    from api.qdrant_schema import clear_readiness_cache

    clear_readiness_cache()
    client = QdrantClient(":memory:")
    settings = make_settings()
    repository = VectorRepository(client)
    repository.ensure_ready(settings)
    client.delete_collection(settings.qdrant_collection)

    assert repository.filename_metadata(settings) == {}
    assert client.collection_exists(settings.qdrant_collection)
    client.delete_collection(settings.qdrant_collection)
    query = EmbeddedText(
        dense=[1.0] + [0.0] * (settings.qdrant_dense_vector_size - 1),
        sparse=models.SparseVector(indices=[1], values=[1.0]),
    )
    assert repository.hybrid_search(settings, query) == []
    clear_readiness_cache()


def test_the_document_listing_is_cached_until_the_corpus_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.corpus import bump_corpus_generation
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    client = QdrantClient(":memory:")
    settings = make_settings()
    repository = VectorRepository(client)
    scans: list[int] = []
    scan = repository._scan_filename_metadata

    def counting_scan(settings: AppSettings) -> dict[str, DocumentMetadata]:
        scans.append(1)
        return scan(settings)

    monkeypatch.setattr(repository, "_scan_filename_metadata", counting_scan)

    repository.filename_metadata(settings)
    VectorRepository(client).filename_metadata(settings)  # a new repository, same client
    assert len(scans) == 1
    bump_corpus_generation()
    repository.filename_metadata(settings)
    assert len(scans) == 2
    clear_metadata_cache()


def _record_scans(
    monkeypatch: pytest.MonkeyPatch, repository: VectorRepository
) -> list[frozenset[str] | None]:
    """Records each metadata scan: None for a full one, else the filenames it read."""

    scans: list[frozenset[str] | None] = []
    scan = repository._scan_filename_metadata

    def counting_scan(
        settings: AppSettings, only: frozenset[str] | None = None
    ) -> dict[str, DocumentMetadata]:
        scans.append(only)
        return scan(settings, only=only)

    monkeypatch.setattr(repository, "_scan_filename_metadata", counting_scan)
    return scans


def test_a_change_to_named_documents_rereads_only_those(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.corpus import bump_corpus_generation
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(settings, [make_point("a1", "a.md"), make_point("b1", "b.md", page=2)])
    scans = _record_scans(monkeypatch, repository)
    assert set(repository.filename_metadata(settings)) == {"a.md", "b.md"}

    # An upload of c.md and another page for a.md: one filtered read of just those.
    repository.upsert(settings, [make_point("c1", "c.md"), make_point("a2", "a.md", page=3)])
    bump_corpus_generation(["c.md"])
    bump_corpus_generation(["a.md"])
    listing = repository.filename_metadata(settings)
    assert set(listing) == {"a.md", "b.md", "c.md"}
    assert listing["a.md"].chunk_count == 2 and listing["a.md"].page_count == 3
    assert listing["b.md"].page_count == 2
    assert scans == [None, frozenset({"a.md", "c.md"})]

    # A delete takes the document out of the listing.
    repository.delete_by_ids(settings, repository.point_ids_for_filename(settings, "b.md"))
    bump_corpus_generation(["b.md"])
    assert set(repository.filename_metadata(settings)) == {"a.md", "c.md"}
    assert scans[-1] == frozenset({"b.md"})
    clear_metadata_cache()


def test_an_unnamed_change_an_old_entry_or_many_changes_rescan_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.corpus import bump_corpus_generation
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(settings, [make_point("a1", "a.md")])
    scans = _record_scans(monkeypatch, repository)
    repository.filename_metadata(settings)

    bump_corpus_generation()  # names nothing: anything may have changed
    repository.filename_metadata(settings)
    assert scans == [None, None]

    bump_corpus_generation([f"doc{i}.md" for i in range(51)])  # past the patch limit
    repository.filename_metadata(settings)
    assert scans[-1] is None

    for index in range(300):  # more bumps than the change log keeps
        bump_corpus_generation([f"doc{index}.md"])
    repository.filename_metadata(settings)
    assert scans[-1] is None

    # Past the TTL the whole collection is read again (another process's writes),
    # even when the change names its document.
    expired = settings.model_copy(update={"document_listing_ttl_seconds": 1e-9})
    bump_corpus_generation(["a.md"])
    repository.filename_metadata(expired)
    assert scans[-1] is None
    assert len(scans) == 5
    clear_metadata_cache()


def test_changed_since_names_what_moved_or_says_it_cannot() -> None:
    from api.corpus import bump_corpus_generation, changed_since, corpus_generation

    start = corpus_generation()
    assert changed_since(start) == frozenset()
    bump_corpus_generation(["a.md"])
    bump_corpus_generation(["b.md", "a.md"])
    assert changed_since(start) == frozenset({"a.md", "b.md"})
    assert changed_since(start + 1) == frozenset({"b.md", "a.md"})
    assert changed_since(start + 99) is None
    bump_corpus_generation()
    assert changed_since(start) is None


def test_concurrent_listings_share_one_scan_of_the_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After each finished job every open UI asked at once, and each scrolled the
    whole collection."""

    import threading

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(settings, [make_point("p1", "a.md")])
    scan = repository._scan_filename_metadata
    scans: list[int] = []
    release = threading.Event()

    def slow_scan(settings: AppSettings) -> dict[str, DocumentMetadata]:
        scans.append(1)
        release.wait(5)
        return scan(settings)

    monkeypatch.setattr(repository, "_scan_filename_metadata", slow_scan)
    results: list[dict[str, DocumentMetadata]] = []
    workers = [
        threading.Thread(target=lambda: results.append(repository.filename_metadata(settings)))
        for _ in range(5)
    ]
    for worker in workers:
        worker.start()
    wait_until(lambda: scans, message="no scan started")
    release.set()
    for worker in workers:
        worker.join(timeout=10)

    assert len(scans) == 1
    assert len(results) == 5 and all(set(result) == {"a.md"} for result in results)


def test_a_scan_slower_than_the_ttl_still_stores_a_fresh_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.repository as repository_module

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(settings, [make_point("p1", "a.md")])
    clock = [1000.0]
    monkeypatch.setattr(repository_module.time, "monotonic", lambda: clock[0])
    scan = repository._scan_filename_metadata
    scans: list[int] = []

    def slow_scan(settings: AppSettings) -> dict[str, DocumentMetadata]:
        scans.append(1)
        clock[0] += 60  # longer than the 30 s TTL
        return scan(settings)

    monkeypatch.setattr(repository, "_scan_filename_metadata", slow_scan)
    repository.filename_metadata(settings)
    repository.filename_metadata(settings)

    assert len(scans) == 1  # the entry was stored as of the scan's end, not its start


def test_staleness_follows_the_chunking_fingerprint_and_the_token_counter() -> None:
    from api.documents import CHUNKER_VERSION

    def meta(**values: Any) -> DocumentMetadata:
        base: dict[str, Any] = {
            "chunk_count": 1,
            "page_count": 1,
            "byte_size": None,
            "uploaded_at": None,
            "chunker_version": CHUNKER_VERSION,
        }
        return DocumentMetadata(**{**base, **values})

    current = "fp-now"
    # Indexed before fingerprints existed but by today's chunker: not flagged on upgrade.
    assert not meta().is_stale(current, "hf")
    assert not meta(chunking_fingerprints=frozenset({current})).is_stale(current, "hf")
    # Another CHUNK_SIZE_TOKENS/overlap/model: one old point is enough.
    assert meta(chunking_fingerprints=frozenset({current, "fp-old"})).is_stale(current, "hf")
    assert meta(chunker_version=CHUNKER_VERSION - 1).is_stale(current, "hf")
    assert meta(chunker_version=None).is_stale(current, "hf")
    # Sized by the heuristic: stale once the real tokenizer is there, not before.
    heuristic = meta(chunking_fingerprints=frozenset({current}), heuristic_chunked=True)
    assert heuristic.is_stale(current, "hf")
    assert not heuristic.is_stale(current, "heuristic")
    assert not heuristic.is_stale(current, "unknown")


def _tagged_point(point_id: str, filename: str, tags: list[str]) -> models.PointStruct:
    point = make_point(point_id, filename)
    assert point.payload is not None
    point.payload["tags"] = tags
    return point


def _counting_scans(repository: VectorRepository, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    scans: list[int] = []
    scan = repository._scan_filename_metadata

    def counting_scan(settings: AppSettings) -> dict[str, DocumentMetadata]:
        scans.append(1)
        return scan(settings)

    monkeypatch.setattr(repository, "_scan_filename_metadata", counting_scan)
    return scans


def test_stored_tags_come_from_one_facet_call_not_a_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tag-scoped question used to scroll the whole collection whenever the listing
    cache was cold; the tags keyword index answers it in one call."""

    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(
        settings,
        [
            _tagged_point("a", "a.md", ["Finance", "q3"]),
            _tagged_point("b", "b.md", ["finance"]),
        ],
    )
    scans = _counting_scans(repository, monkeypatch)

    assert repository.stored_tags(settings) == {"Finance", "finance", "q3"}
    assert scans == []
    clear_metadata_cache()


def test_stored_tags_fall_back_to_the_scan_when_the_facet_may_be_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A facet that returns as many values as it was asked for may have cut some off,
    and a missing stored spelling would silently search the wrong documents."""

    from api import repository as repository_module
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    monkeypatch.setattr(repository_module, "TAG_FACET_LIMIT", 2)
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    repository.upsert(
        settings,
        [
            _tagged_point("a", "a.md", ["alpha", "beta"]),
            _tagged_point("b", "b.md", ["gamma"]),
        ],
    )
    scans = _counting_scans(repository, monkeypatch)

    assert repository.stored_tags(settings) == {"alpha", "beta", "gamma"}
    assert scans == [1]
    clear_metadata_cache()


def test_stored_tags_fall_back_to_the_scan_when_the_server_cannot_facet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [_tagged_point("a", "a.md", ["alpha"])])

    def no_facet(*args: Any, **kwargs: Any) -> object:
        raise ValueError("facet is not supported")

    monkeypatch.setattr(client, "facet", no_facet)
    assert repository.stored_tags(settings) == {"alpha"}
    clear_metadata_cache()


def test_stored_tags_use_a_warm_listing_without_calling_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api.repository import clear_metadata_cache

    clear_metadata_cache()
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    repository.upsert(settings, [_tagged_point("a", "a.md", ["alpha"])])
    repository.filename_metadata(settings)

    def no_facet(*args: Any, **kwargs: Any) -> object:
        raise AssertionError("the warm listing already has every tag")

    monkeypatch.setattr(client, "facet", no_facet)
    assert repository.stored_tags(settings) == {"alpha"}
    clear_metadata_cache()


def test_search_hits_carry_only_the_fields_a_candidate_is_built_from() -> None:
    """Search reads six payload fields; tags, sizes, stamps and fingerprints stay on
    the server. Every field the candidate needs must still be there."""

    from api.embeddings import EmbeddedText
    from api.retrieval import points_to_chunks

    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    point = _tagged_point("a", "a.md", ["alpha"])
    assert point.payload is not None
    point.payload.update({"chunk_ordinal": 3, "byte_size": 10, "uploaded_at": 1.0})
    repository.upsert(settings, [point])
    query = EmbeddedText(dense=[0.1, 0.2, 0.3], sparse=models.SparseVector(indices=[1], values=[1]))

    for points in (
        repository.hybrid_search(settings, query),
        repository.search_leg(settings, query, "dense"),
        repository.search_leg(settings, query, "sparse"),
        repository._manual_hybrid_query(settings, query, None),
    ):
        assert set(points[0].payload or {}) == {
            "filename",
            "page",
            "section",
            "chunk_id",
            "text",
            "chunk_ordinal",
        }
        chunk = points_to_chunks(points)[0]
        assert (chunk.filename, chunk.page, chunk.section, chunk.chunk_id) == (
            "a.md",
            1,
            "Intro",
            "a",
        )
        assert chunk.text == "hello" and chunk.chunk_ordinal == 3
