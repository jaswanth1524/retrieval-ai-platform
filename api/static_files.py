"""Serving the built frontend: precompressed assets and long-lived caching."""

from __future__ import annotations

from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

# Encodings the build writes next to each text asset (frontend/vite.config.ts), most
# compact first.
PRECOMPRESSED = (("br", ".br"), ("gzip", ".gz"))

# Vite content-hashes every file under assets/: a changed file gets a new name, so a
# browser may keep one forever. index.html keeps its name across deploys and must be
# revalidated, or a returning browser would keep loading the previous build's chunks.
IMMUTABLE = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"


class FrontendFiles(StaticFiles):
    """``StaticFiles`` that serves a precompressed copy when the browser accepts it.

    The API compressed nothing, so every visit downloaded about 480 KB of JS and CSS
    as-is. Compressing at build time costs nothing per request, and unlike a GZip
    middleware it never touches the API's responses, so the answer stream can't be
    buffered by it.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code in (200, 304):
            response.headers["cache-control"] = (
                IMMUTABLE if path.startswith("assets/") else REVALIDATE
            )
        if response.status_code != 200 or not isinstance(response, FileResponse):
            return response
        accepted = _accepted_encodings(Headers(scope=scope).get("accept-encoding", ""))
        for encoding, suffix in PRECOMPRESSED:
            compressed = Path(f"{response.path}{suffix}")
            if encoding in accepted and compressed.is_file():
                return FileResponse(
                    compressed,
                    # The original file's type: the browser decodes, then runs it as JS.
                    media_type=response.media_type,
                    headers={
                        "cache-control": response.headers["cache-control"],
                        "content-encoding": encoding,
                        "vary": "Accept-Encoding",
                    },
                )
        if any(Path(f"{response.path}{suffix}").is_file() for _, suffix in PRECOMPRESSED):
            # A cache between here and a browser must not hand this copy to one that
            # does accept the compressed one, or the reverse.
            response.headers["vary"] = "Accept-Encoding"
        return response


def _accepted_encodings(header: str) -> set[str]:
    """Encodings an ``Accept-Encoding`` header allows (a ``q=0`` entry refuses one)."""

    accepted: set[str] = set()
    for item in header.split(","):
        name, _, params = item.strip().partition(";")
        quality = params.strip().removeprefix("q=").strip()
        if name and quality not in ("0", "0.0", "0.00", "0.000"):
            accepted.add(name.strip().lower())
    return accepted
