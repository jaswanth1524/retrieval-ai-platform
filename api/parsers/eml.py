"""Email (.eml) parsing: headers plus the message body; attachments are skipped."""

from __future__ import annotations

import logging
from email import policy
from email.message import EmailMessage, MIMEPart
from email.parser import BytesParser

from api.parsers.common import (
    DEFAULT_SECTION,
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    cap_section,
    normalize_text,
)
from api.parsers.html import parse_html_document

logger = logging.getLogger(__name__)

_HEADERS = ("Subject", "From", "To", "Cc", "Date")


def parse_eml_document(filename: str, content: bytes) -> list[DocumentSection]:
    """Extract an email's headers and body.

    The headers come first, as their own section, so "who sent the invoice" and "when"
    are answerable and citable. The body prefers the plain-text part and falls back to
    the HTML one (through the HTML parser). The subject is the section label of a
    plain-text body. Attachments are not indexed; their count is logged.
    """

    try:
        message = BytesParser(policy=policy.default).parsebytes(content)
    except Exception as exc:
        raise DocumentParseError(f"Could not parse email '{filename}'.") from exc
    if not isinstance(message, EmailMessage):
        raise DocumentParseError(f"Could not parse email '{filename}'.")

    subject = str(message.get("Subject") or "").strip()
    label = cap_section(subject) if subject else DEFAULT_SECTION
    header_text = normalize_text(
        "\n".join(f"{name}: {message.get(name)}" for name in _HEADERS if message.get(name))
    )
    sections: list[DocumentSection] = []
    if header_text:
        sections.append(
            DocumentSection(
                filename=filename, page=1, section=label, text=header_text, boundary="semantic"
            )
        )

    body = message.get_body(preferencelist=("plain", "html"))
    if body is not None:
        text = _part_text(body)
        if body.get_content_subtype() == "html":
            try:
                sections.extend(parse_html_document(filename, text))
            except EmptyDocumentError:
                pass
        elif normalize_text(text):
            sections.append(
                DocumentSection(filename=filename, page=1, section=label, text=normalize_text(text))
            )

    attachments = sum(1 for _ in message.iter_attachments())
    if attachments:
        logger.info("Email %s: %d attachment(s) not indexed.", filename, attachments)
    if not sections:
        raise EmptyDocumentError(f"Email '{filename}' has no extractable text.")
    return sections


def _part_text(part: MIMEPart) -> str:
    """A text part's content, decoded; an unknown charset falls back to UTF-8."""

    try:
        content = part.get_content()
    except (LookupError, UnicodeError):
        payload = part.get_payload(decode=True)
        return payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else ""
    return content if isinstance(content, str) else ""
