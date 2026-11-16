"""Read FITS headers over HTTP without downloading the pixels.

A SPHEREx Level-2 frame is ~63 MB, but its WCS lives in the first few tens of
kilobytes. FITS is block-structured (2880-byte blocks, ASCII 80-column cards,
terminated by an ``END`` card), so a single HTTP range request for the head of
the file yields the complete astrometric solution.

Measured: 46 KB in 0.47 s, versus 63 MB for the full frame -- ~1400x less data.
That is what makes it practical to characterise thousands of frames before
deciding which ones to actually pull.

The 46 KB is exact, not approximate: ``16 * BLOCK`` = 46,080 bytes. The 1400x is
against a ``qr3/`` v27 frame, 65,577,600 B, which divides out at 1423x. This
module mostly does not meet those -- 29 of the 30 epochs the bake has landed are
``qr2/``, 20 of them 71,634,240 B and 9 of them 71,637,120 B, and both round to
**1555x** less data for the same request. Both
frame sizes are real and neither supersedes the other; the release table in
``docs/FEASIBILITY.md`` reconciles them.
"""

from __future__ import annotations

import random
import time
import urllib.error
import urllib.request

BLOCK = 2880
CARD = 80
#: 16 blocks holds the primary header plus the first extension header for every
#: QR3 frame observed so far. Bumped automatically by ``read_header_cards``.
DEFAULT_BLOCKS = 16

#: Transient-failure retries for every HTTP call in this module. A cutout bake
#: issues hundreds of range requests over minutes, and both S3 and IRSA will
#: reset a connection occasionally; without this, one reset in three hundred
#: aborts the whole run. Exponential backoff with jitter, because a fixed delay
#: across concurrent workers just re-collides.
HTTP_ATTEMPTS = 5
HTTP_BACKOFF_BASE = 0.6


def _is_transient(exc: BaseException) -> bool:
    """Worth retrying? Network faults and 5xx yes; 4xx no.

    416 in particular must not be retried -- callers walking an object of
    unknown length rely on it arriving promptly as end-of-file.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (urllib.error.URLError, OSError, TimeoutError))


def http_retry(fn, *, attempts: int = HTTP_ATTEMPTS):
    """Call ``fn()``, retrying transient failures with jittered backoff."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # re-raised below unless transient
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            last = exc
            time.sleep(HTTP_BACKOFF_BASE * (2**attempt) * (0.5 + random.random()))
    raise last  # pragma: no cover -- unreachable, the loop always raises first


class HeaderError(RuntimeError):
    """The byte range did not contain a complete, parsable FITS header."""


class RangeBeyondEOF(HeaderError):
    """The requested range starts at or past the end of the object (HTTP 416)."""


def fetch_range(url: str, start: int, end: int, *, timeout: float = 60.0) -> bytes:
    """HTTP GET of an inclusive byte range.

    Raises :class:`RangeBeyondEOF` on 416 so callers walking an unknown-length
    object can treat it as a clean end-of-file rather than a hard failure.
    """
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})

    def once() -> bytes:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # 206 Partial Content expected; a 200 means the server ignored
            # Range and is streaming the whole object, so stop after what we
            # asked for.
            return resp.read(end - start + 1)

    try:
        return http_retry(once)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            raise RangeBeyondEOF(f"{url}: range {start}-{end} past EOF") from exc
        raise


def content_length(url: str, *, timeout: float = 60.0) -> int | None:
    """Object size via HEAD, or None if the server does not report it."""
    req = urllib.request.Request(url, method="HEAD")

    def once() -> str | None:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.headers.get("Content-Length")

    raw = http_retry(once)
    return int(raw) if raw else None


def parse_cards(raw: bytes) -> dict[str, str]:
    """Parse 80-column FITS cards up to the first ``END``.

    Returns raw string values; callers coerce. Keys appearing in more than one
    header (e.g. ``NAXIS``) resolve to the first occurrence, which is the
    primary header -- use :func:`tracklet.archive.cutout.map_layout` when that
    matters, since it walks the HDU chain and keeps each extension separately.

    (This pointed at a ``split_hdus`` that does not exist and never did, in this
    module or any other.)
    """
    out: dict[str, str] = {}
    text = raw.decode("ascii", "replace")
    for i in range(0, len(text) - CARD + 1, CARD):
        card = text[i : i + CARD]
        key = card[:8].strip()
        if key == "END":
            break
        if not key or card[8:10] != "= ":
            continue
        value = card[10:]
        # strip an inline comment, but not a '/' inside a quoted string
        if value.lstrip().startswith("'"):
            close = value.find("'", value.find("'") + 1)
            value = value[: close + 1] if close > 0 else value
        else:
            value = value.split("/")[0]
        out.setdefault(key, value.strip().strip("'").strip())
    else:
        raise HeaderError("no END card within the fetched range")
    return out


def read_header_cards(
    url: str, *, blocks: int = DEFAULT_BLOCKS, max_blocks: int = 128
) -> dict[str, str]:
    """Fetch and parse the **primary** header, growing the range until ``END``.

    Byte 0 onward, and ``parse_cards`` stops at the first ``END`` -- so this
    returns HDU 0 and nothing else, whatever ``blocks`` is set to. A SPHEREx
    Level-2 primary header has no astrometry in it; the WCS is in the IMAGE
    extension. For that, use :func:`tracklet.archive.cutout.map_layout`, which
    walks the HDU chain and keeps each extension's own cards.

    (``DEFAULT_BLOCKS`` reaches past the primary header into the first
    extension's, but that sizing is for ``map_layout``'s benefit -- this
    function cannot see those cards.)
    """
    while blocks <= max_blocks:
        raw = fetch_range(url, 0, blocks * BLOCK - 1)
        try:
            return parse_cards(raw)
        except HeaderError:
            blocks *= 2
    raise HeaderError(f"{url}: no END card in first {max_blocks} blocks")


def _f(cards: dict[str, str], key: str, default: float | None = None) -> float:
    raw = cards.get(key)
    if raw is None:
        if default is None:
            raise HeaderError(f"missing required card {key}")
        return default
    return float(raw)


def matrix_from_cards(
    cards: dict[str, str],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Linear WCS matrix in deg/pixel, from either CD or PC+CDELT conventions.

    SPHEREx QR3 frames carry PC with CDELT scaling; older products may use CD.
    Both collapse to one matrix so downstream code need not branch.
    """
    if "CD1_1" in cards:
        return (
            (_f(cards, "CD1_1"), _f(cards, "CD1_2", 0.0)),
            (_f(cards, "CD2_1", 0.0), _f(cards, "CD2_2")),
        )
    d1, d2 = _f(cards, "CDELT1", 1.0), _f(cards, "CDELT2", 1.0)
    return (
        (d1 * _f(cards, "PC1_1"), d1 * _f(cards, "PC1_2", 0.0)),
        (d2 * _f(cards, "PC2_1", 0.0), d2 * _f(cards, "PC2_2")),
    )


def pixel_scale_arcsec(
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Plate scale from the matrix determinant.

    Must use the determinant, not a diagonal term: PC1_1 alone is
    ``scale * cos(rotation)``, which understates the scale for any rotated
    frame. The error flows straight into pixel solid angle and therefore into
    every derived flux and exclusion limit.
    """
    (a, b), (c, d) = matrix
    return abs(a * d - b * c) ** 0.5 * 3600.0


# `frame_wcs`, `frame_wcs_many` and the `FrameWCS` dataclass were here, and they
# could not have worked on a single frame this project reads.
#
# `read_header_cards` above always fetches from byte 0 and `parse_cards` stops at
# the first END, so it returns the *primary* header and there is no argument that
# changes that. A SPHEREx Level-2 primary header carries no astrometry -- the WCS
# lives in the IMAGE extension -- so `frame_wcs` raised
# `HeaderError: missing required card PC1_1` on any real frame. Called on the NEP
# field's epoch 0 it did exactly that.
#
# Nothing outside this module called it. `frame_wcs_many` was its only caller and
# had no callers itself, and no test touched either. `scripts/bake_field.py` says
# why in its own comment: it uses `map_layout`, which walks the HDU chain and
# keeps each extension's cards, "deliberately -- the primary header carries no
# WCS". So the working path was already the one in use and this was a parallel
# API that looked usable, wasn't, and failed with an error naming a missing card
# rather than the wrong HDU.
#
# `read_header_cards` stays: it does what it says and says what it does. The
# limitation is now written on it.
