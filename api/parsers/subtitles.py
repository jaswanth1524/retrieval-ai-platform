"""SRT and WebVTT subtitles: the spoken text, a section per minute.

Cue numbers, timings, styling tags and VTT headers/notes are dropped; the text of the
cues starting in each minute of the recording becomes one section, labelled with that
minute's start ("12:00") and given the minute as its page (minute 0 is page 1), so a
citation points at the place in the recording.
"""

from __future__ import annotations

import re

from api.parsers.common import DocumentSection, EmptyDocumentError

# 00:01:02,345 --> 00:01:05,000 (SRT) or 01:02.345 --> 01:05.000 (VTT, hours optional).
_TIMING_RE = re.compile(
    r"^\s*(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:\d{1,3}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
)
_TAG_RE = re.compile(r"<[^>]{0,200}>|\{\\[^}]{0,200}\}")


def _minute_label(minute: int) -> str:
    hours, minutes = divmod(minute, 60)
    return f"{hours}:{minutes:02d}:00" if hours else f"{minutes}:00"


def parse_subtitles_document(filename: str, text: str) -> list[DocumentSection]:
    """The cues of ``text`` grouped by the minute they start in."""

    minutes: dict[int, list[str]] = {}
    current: int | None = None
    in_note = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            current = None
            in_note = False
            continue
        if in_note:
            continue
        if line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            in_note = line.startswith(("NOTE", "STYLE", "REGION"))
            continue
        timing = _TIMING_RE.match(line)
        if timing is not None:
            hours = int(timing.group(1) or 0)
            current = hours * 60 + int(timing.group(2))
            continue
        if current is None:
            continue  # a cue number (SRT) or identifier (VTT) before its timing
        spoken = " ".join(_TAG_RE.sub("", line).split())
        if spoken:
            minutes.setdefault(current, []).append(spoken)
    sections = [
        DocumentSection(
            filename=filename,
            page=minute + 1,
            section=_minute_label(minute),
            text=" ".join(lines),
            # A minute stays its own passage, however short: a citation names it.
            boundary="isolated",
        )
        for minute, lines in sorted(minutes.items())
    ]
    if not sections:
        raise EmptyDocumentError(f"'{filename}' has no subtitle text.")
    return sections
