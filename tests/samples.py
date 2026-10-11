"""Builders for small, real sample documents used across tests."""

from __future__ import annotations

import io


def pdf_bytes(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""

    objects: list[bytes] = []
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for index, text in enumerate(pages):
        content_id = 4 + 2 * index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def epub_bytes(chapters: list[tuple[str, str]], *, spine_order: list[int] | None = None) -> bytes:
    """A minimal valid EPUB: one XHTML file per ``(title, body)`` chapter, read in
    ``spine_order`` (chapter indexes; default: as given)."""

    import zipfile

    order = spine_order if spine_order is not None else list(range(len(chapters)))
    manifest = "".join(
        f'<item id="c{index}" href="text/chapter%20{index}.xhtml" '
        'media-type="application/xhtml+xml"/>'
        for index in range(len(chapters))
    )
    spine = "".join(f'<itemref idref="c{index}"/>' for index in order)
    package = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f"<manifest>{manifest}"
        '<item id="css" href="style.css" media-type="text/css"/></manifest>'
        f"<spine>{spine}</spine></package>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", package)
        archive.writestr("OEBPS/style.css", "body { margin: 0 }")
        for index, (title, body) in enumerate(chapters):
            archive.writestr(
                f"OEBPS/text/chapter {index}.xhtml",
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title}</title></head>"
                f"<body><h1>{title}</h1><p>{body}</p></body></html>",
            )
    return buffer.getvalue()


def eml_bytes(
    *, subject: str, body: str, html: str | None = None, attachment: bool = False
) -> bytes:
    """A small RFC 5322 email with a plain-text body, optionally an HTML alternative
    and a PDF attachment."""

    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "Ada Lovelace <ada@example.com>"
    message["To"] = "team@example.com"
    message["Date"] = "Mon, 05 Oct 2026 09:30:00 +0000"
    message.set_content(body)
    if html is not None:
        message.add_alternative(html, subtype="html")
    if attachment:
        message.add_attachment(
            b"%PDF-1.4 not indexed", maintype="application", subtype="pdf", filename="a.pdf"
        )
    return bytes(message)
