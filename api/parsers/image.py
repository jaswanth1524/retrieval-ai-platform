"""Images (scans, photos of pages, screenshots), read with OCR — only when the
optional ``ocr`` extra is installed; without it the formats aren't offered at all."""

from __future__ import annotations

import importlib.util
import logging
from io import BytesIO

from api.parsers.common import (
    DocumentParseError,
    DocumentSection,
    EmptyDocumentError,
    infer_section,
    normalize_text,
)
from api.parsers.pdf import OCR_MAX_PAGE_PIXELS, _get_ocr_engine
from api.settings import AppSettings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})

# An image declaring more pixels than this is refused before it is decoded: a small
# file can declare a huge canvas (a decompression bomb).
MAX_IMAGE_PIXELS = 200_000_000


def ocr_available() -> bool:
    """Whether the ``ocr`` extra is installed, so images can be read at all."""

    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def parse_image_document(
    filename: str, content: bytes, settings: AppSettings | None = None
) -> list[DocumentSection]:
    """OCR an image; each frame of a multi-page TIFF is a page, up to ``max_ocr_pages``.

    A frame larger than a PDF page is allowed to be (``OCR_MAX_PAGE_PIXELS``) is scaled
    down before OCR, the same bound the PDF path keeps.
    """

    try:
        import numpy as np
        from PIL import Image, ImageSequence
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise EmptyDocumentError(
            f"Image '{filename}' can only be read with OCR. Install it with: "
            "uv sync --extra ocr (or pip install 'docrag[ocr]')."
        ) from exc

    limits = settings if settings is not None else AppSettings.model_construct()
    max_pages = int(limits.max_ocr_pages)
    try:
        image = Image.open(BytesIO(content))
        width, height = image.size
    except Exception as exc:  # UnidentifiedImageError, a truncated header, ...
        raise DocumentParseError(f"Could not read image '{filename}'.") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise DocumentParseError(
            f"Image '{filename}' is {width} x {height} pixels, over the "
            f"{MAX_IMAGE_PIXELS // 1_000_000} megapixel limit."
        )

    engine = _get_ocr_engine(RapidOCR)
    sections: list[DocumentSection] = []
    with image:
        for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            if number > max_pages:
                logger.warning(
                    "Image %s: only its first %d pages were read (MAX_OCR_PAGES).",
                    filename,
                    max_pages,
                )
                break
            try:
                page = frame.convert("RGB")
                pixels = page.width * page.height
                if pixels > OCR_MAX_PAGE_PIXELS:
                    scale = (OCR_MAX_PAGE_PIXELS / pixels) ** 0.5
                    page = page.resize((int(page.width * scale), int(page.height * scale)))
                result, _ = engine(np.array(page))
            except Exception:
                logger.warning(
                    "Image %s: OCR failed on page %d; skipping it.", filename, number, exc_info=True
                )
                continue
            text = normalize_text("\n".join(line[1] for line in (result or [])))
            if text:
                sections.append(
                    DocumentSection(
                        filename=filename, page=number, section=infer_section(text), text=text
                    )
                )
    if not sections:
        raise EmptyDocumentError(f"OCR found no text in image '{filename}'.")
    return sections
