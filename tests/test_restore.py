"""restore_backup's guards against an untrusted zip: entry count, names, damage, sizes."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pytest

from api import restore
from api.documents import DocumentSource
from api.export import MANIFEST_VERSION
from api.feedback import Feedback
from api.restore import BackupImportError, restore_backup
from api.schemas import DocumentJobAcceptedResponse


def _document(name: str, original: str | None = None) -> dict[str, Any]:
    return {"filename": name, "original": original if original is not None else f"originals/{name}"}


def _zip(documents: list[dict[str, Any]], members: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"manifest_version": MANIFEST_VERSION, "documents": documents}),
        )
        for name, data in members.items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


class Recorder:
    def __init__(self) -> None:
        self.queued: list[tuple[str, bytes]] = []
        self.sources: dict[str, DocumentSource | None] = {}

    def enqueue(
        self,
        name: str,
        content: bytes,
        uploaded_at: float | None,
        tags: tuple[str, ...],
        source: DocumentSource | None,
    ) -> DocumentJobAcceptedResponse:
        self.queued.append((name, content))
        self.sources[name] = source
        return DocumentJobAcceptedResponse(job_id=f"job-{name}", filename=name, state="queued")


def _restore(upload: io.BytesIO, recorder: Recorder, **kwargs: Any) -> restore.ImportOutcome:
    return restore_backup(
        upload,
        existing_filenames=kwargs.pop("existing", set()),
        max_original_bytes=kwargs.pop("max_bytes", 1_000_000),
        has_room=lambda size: True,
        enqueue=recorder.enqueue,
        feedback_store=kwargs.pop("feedback_store", None),
    )


def test_an_archive_with_too_many_entries_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restore, "MAX_MEMBERS", 3)
    upload = _zip([], {f"originals/{i}.txt": b"x" for i in range(3)})

    with pytest.raises(BackupImportError, match="more than 3 entries"):
        _restore(upload, Recorder())


def test_names_that_would_land_elsewhere_are_rejected_not_written() -> None:
    documents = [
        _document("../evil.txt"),
        _document("nested/b.txt"),
        _document("bad\x00name.txt"),
        _document("ok.txt"),
        {"filename": 42},
    ]
    members = {
        "originals/../evil.txt": b"evil",
        "originals/nested/b.txt": b"b",
        "originals/ok.txt": b"fine",
    }
    recorder = Recorder()

    outcome = _restore(_zip(documents, members), recorder)

    assert recorder.queued == [("ok.txt", b"fine")]
    assert set(outcome.rejected) == {"../evil.txt", "nested/b.txt", "bad\x00name.txt"}


def test_a_damaged_or_oversized_original_is_rejected_and_the_rest_restored() -> None:
    upload = _zip(
        [_document("broken.txt"), _document("huge.txt"), _document("ok.txt")],
        {
            "originals/broken.txt": b"A" * 4096,
            "originals/huge.txt": b"B" * 200,
            "originals/ok.txt": b"fine",
        },
    )
    data = bytearray(upload.getvalue())
    # Corrupt broken.txt's compressed bytes (its CRC no longer matches).
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as archive:
        info = archive.getinfo("originals/broken.txt")
    start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    data[start + 2] ^= 0xFF
    recorder = Recorder()

    outcome = _restore(io.BytesIO(bytes(data)), recorder, max_bytes=100)

    assert recorder.queued == [("ok.txt", b"fine")]
    assert set(outcome.rejected) == {"broken.txt", "huge.txt"}


class FeedbackSink:
    def __init__(self) -> None:
        self.rows: list[Feedback] = []

    def import_rows(self, rows: list[Feedback], workspace: str) -> int:
        self.rows.extend(rows)
        return len(rows)


def _feedback_line(index: int) -> str:
    return (
        json.dumps(
            {
                "id": f"row-{index}",
                "question": f"Question {index}?",
                "rating": "up",
                "created_at": 1_700_000_000.0 + index,
            }
        )
        + "\n"
    )


def test_feedback_past_the_byte_cap_stops_and_keeps_the_rows_before_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [_feedback_line(index) for index in range(10)]
    monkeypatch.setattr(restore, "_FEEDBACK_MAX_BYTES", len(lines[0]) * 3 + 1)
    sink = FeedbackSink()

    outcome = _restore(
        _zip([], {"feedback.jsonl": "".join(lines).encode()}),
        Recorder(),
        feedback_store=sink,
    )

    assert [row.id for row in sink.rows] == ["row-0", "row-1", "row-2"]
    assert outcome.feedback_imported == 3


def test_a_name_listed_twice_in_the_manifest_is_queued_once() -> None:
    recorder = Recorder()

    outcome = _restore(
        _zip([_document("a.txt"), _document("a.txt")], {"originals/a.txt": b"alpha"}), recorder
    )

    assert recorder.queued == [("a.txt", b"alpha")]
    assert outcome.skipped == ["a.txt"]


def test_a_fetched_document_keeps_where_it_came_from() -> None:
    documents = [
        {
            **_document("page.html"),
            "source_url": "https://example.com/page",
            "source_etag": '"v1"',
            "source_last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "source_refresh_minutes": 30,
        },
        # Not a URL the server could have fetched, and an older manifest's row.
        {**_document("odd.html"), "source_url": "file:///etc/passwd"},
        _document("plain.txt"),
        # A validator no request could carry (non-ASCII): the URL is kept, it isn't.
        {
            **_document("latin.html"),
            "source_url": "https://example.com/latin",
            "source_etag": '"caf\xe9"',
            # Not a number of minutes the API takes: left out.
            "source_refresh_minutes": -5,
        },
    ]
    upload = _zip(
        documents,
        {
            "originals/page.html": b"<p>a</p>",
            "originals/odd.html": b"<p>b</p>",
            "originals/plain.txt": b"c",
            "originals/latin.html": b"<p>d</p>",
        },
    )
    recorder = Recorder()
    _restore(upload, recorder)
    assert recorder.sources == {
        "page.html": DocumentSource(
            "https://example.com/page", '"v1"', "Mon, 01 Jan 2024 00:00:00 GMT", 30
        ),
        "odd.html": None,
        "plain.txt": None,
        "latin.html": DocumentSource("https://example.com/latin"),
    }


def test_the_manifest_writes_a_fetched_documents_source_and_refresh_interval() -> None:
    from api.export import _source_fields
    from api.repository import DocumentMetadata

    meta = DocumentMetadata(
        chunk_count=1,
        page_count=1,
        byte_size=1,
        uploaded_at=1.0,
        chunker_version=9,
        source_url="https://example.com/page",
        source_etag='"v1"',
        source_refresh_minutes=0,
    )
    assert _source_fields(meta) == {
        "source_url": "https://example.com/page",
        "source_etag": '"v1"',
        "source_refresh_minutes": 0,
    }
    assert _source_fields(DocumentMetadata(1, 1, 1, 1.0, 9)) == {}
