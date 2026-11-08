"""Mailbox files (.mbox): every message of an export, each parsed as an .eml is."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
)
from api.parsers.eml import parse_eml_document

logger = logging.getLogger(__name__)

# Past this many messages the rest are left out (and logged): one upload is one
# document, and a whole archive belongs in several.
MAX_MBOX_MESSAGES = 5000
# A message starts at a "From " line at the start of the file or after a blank line.
_FROM_LINE = re.compile(rb"(?:\A|\n)(?=From [^\n]*\n)")
# Body lines that begin "From " are escaped with ">" (mboxrd); one level is taken off.
_ESCAPED_FROM = re.compile(rb"^>(>*From )", re.MULTILINE)


def parse_mbox_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Each message's headers and body (``parse_eml_document``), one page per message
    in mailbox order, so a citation names the message by its position. A message that
    can't be read is skipped; attachments are not indexed."""

    if not content.startswith(b"From "):
        raise DocumentParseError(f"'{filename}' is not an mbox file (it must start with 'From ').")
    messages = [part for part in _FROM_LINE.split(content) if part.strip()]
    if len(messages) > MAX_MBOX_MESSAGES:
        logger.info(
            "Mailbox %s: %d messages; only the first %d are indexed.",
            filename,
            len(messages),
            MAX_MBOX_MESSAGES,
        )
        messages = messages[:MAX_MBOX_MESSAGES]
    sections: list[DocumentSection] = []
    skipped = 0
    for page, raw in enumerate(messages, start=1):
        # The envelope line ("From sender date") isn't part of the message.
        _, _, body = raw.partition(b"\n")
        try:
            parsed = parse_eml_document(filename, _ESCAPED_FROM.sub(rb"\1", body))
        except (DocumentParseError, EmptyDocumentError):
            skipped += 1
            continue
        sections.extend(replace(section, page=page, boundary="isolated") for section in parsed)
    if skipped:
        logger.info("Mailbox %s: %d message(s) had no readable text.", filename, skipped)
    if not sections:
        raise EmptyDocumentError(f"Mailbox '{filename}' has no extractable text.")
    return sections
